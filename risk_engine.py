"""Moteur de calcul du Portfolio Risk Engine (utilisé par l'application web app.py).

Mêmes méthodes que le notebook Portfolio_Risk_Engine.ipynb : cours ajustés Yahoo Finance convertis dans
la devise de référence, rendements simples pour le P&L (poids constants), rendements log pour les
statistiques annualisées, VaR / ES historique, paramétrique et Monte Carlo, backtest (Kupiec, feux
tricolores) et module de stress.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.special import xlogy

TRADING_DAYS = 252
MINOR_UNITS = {"GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01), "ILA": ("ILS", 0.01), "ZAc": ("ZAR", 0.01)}
QUOTE_TYPE_CLASS = {"EQUITY": "Actions", "ETF": "ETF", "MUTUALFUND": "Fonds", "CRYPTOCURRENCY": "Crypto",
                    "CURRENCY": "Devises", "FUTURE": "Matières premières", "INDEX": "Indice"}
EPISODES = {
    "Crise financière mondiale 2007-2009": ("2007-10-09", "2009-03-09"),
    "Faillite de Lehman Brothers (sept.-nov. 2008)": ("2008-09-12", "2008-11-20"),
    "Krach du Covid (févr.-mars 2020)": ("2020-02-19", "2020-03-23"),
    "Choc de taux 2022": ("2022-01-03", "2022-10-12"),
    "Éclatement de la bulle internet 2000-2002": ("2000-03-24", "2002-10-09"),
    "Black Monday 1987": ("1987-10-16", "1987-10-19"),
}


class EngineError(ValueError):
    """Erreur affichée telle quelle à l'utilisateur."""


@dataclass
class Config:
    tickers: list
    classes: list
    weights: np.ndarray                     # fractions, somme = 1
    base_ccy: str = "EUR"
    value: float = 1_000_000.0
    start: pd.Timestamp = pd.Timestamp("2018-01-01")
    end: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.today().normalize())
    alphas: list = field(default_factory=lambda: [0.95, 0.99])
    horizon: int = 10
    window: int = 756
    n_sims: int = 10_000
    rf: float = 0.03
    market: str = "SPY"
    seed: int = 42
    episodes: list = field(default_factory=lambda: ["Crise financière mondiale 2007-2009",
                                                    "Krach du Covid (févr.-mars 2020)", "Choc de taux 2022"])
    custom_episode: tuple | None = None    # (début, fin) facultatif
    market_shock: float = -0.20
    custom_shocks: dict = field(default_factory=dict)
    rho_star: float = 0.9
    k_vol: float = 2.0
    loss_limit: float = 0.20

    @property
    def a_ref(self):
        return 0.99 if 0.99 in self.alphas else max(self.alphas)


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------
def validate(cfg: Config, strict: bool = True):
    """Renvoie (erreurs, avertissements). Avec strict=True, les règles du projet sont des erreurs."""
    errors, warnings = [], []
    rule = errors if strict else warnings
    t = cfg.tickers
    if len(t) < 2:
        errors.append("Il faut au moins 2 tickers.")
    dup = sorted({x for x in t if t.count(x) > 1})
    if dup:
        errors.append(f"Tickers en double : {', '.join(dup)}.")
    bad = [x for x in t if not re.fullmatch(r"[A-Z0-9.\-^=]{1,20}", x)]
    if bad:
        errors.append(f"Tickers invalides : {', '.join(bad)}.")
    if len(t) < 6:
        rule.append(f"Le portefeuille contient {len(t)} tickers ; le projet en exige au moins 6.")
    if len({c.strip().lower() for c in cfg.classes if c.strip()}) < 3:
        rule.append("Le projet exige au moins 3 classes d'actifs différentes.")
    if any(not c.strip() for c in cfg.classes):
        errors.append("Chaque ticker doit avoir une classe d'actifs.")
    w = np.asarray(cfg.weights, dtype=float)
    if np.any(np.isnan(w)):
        errors.append("Chaque ticker doit avoir un poids.")
    elif np.any(w < 0):
        errors.append("Les poids doivent être positifs ou nuls (portefeuille long-only).")
    elif abs(w.sum() - 1) > 1e-4:
        errors.append(f"Les poids doivent totaliser 100 % (somme actuelle : {w.sum() * 100:g} %).")
    if not re.fullmatch(r"[A-Z]{3}", cfg.base_ccy):
        errors.append("Devise : code ISO à 3 lettres attendu (EUR, USD...).")
    if cfg.value <= 0:
        errors.append("La valeur du portefeuille doit être positive.")
    if cfg.end <= cfg.start:
        errors.append("La date de fin doit suivre la date de début.")
    else:
        years = (cfg.end - cfg.start).days / 365.25
        if years < 5:
            rule.append(f"La période couvre {years:.1f} ans ; le projet en exige au moins 5.")
        if not (cfg.start <= pd.Timestamp("2020-01-31") and cfg.end >= pd.Timestamp("2022-12-30")):
            rule.append("La période doit inclure 2020 et 2022.")
    if not cfg.alphas:
        errors.append("Choisissez au moins un niveau de confiance.")
    if any(not 0.5 < a < 1 for a in cfg.alphas):
        errors.append("Les niveaux de confiance doivent être compris entre 50 % et 100 %.")
    for k in cfg.custom_shocks:
        if k not in t:
            errors.append(f"Choc personnalisé sur {k}, qui n'est pas dans le portefeuille.")
    if any(v <= -1 for v in cfg.custom_shocks.values()):
        errors.append("Un choc ne peut pas dépasser −100 %.")
    if not 0.8 <= cfg.rho_star <= 1:
        errors.append("ρ* doit être compris entre 0,8 et 1.")
    if not 1.5 <= cfg.k_vol <= 3:
        errors.append("k doit être compris entre 1,5 et 3.")
    if cfg.custom_episode and cfg.custom_episode[1] <= cfg.custom_episode[0]:
        errors.append("Fenêtre de stress personnalisée : la fin doit suivre le début.")
    return errors, warnings


# ---------------------------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------------------------
def _yf_close(tickers, start, end, threads):
    data = yf.download(tickers, start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1),
                       auto_adjust=True, progress=False, threads=threads)
    if data is None or data.empty:
        return pd.DataFrame(columns=tickers, dtype=float)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]].set_axis(tickers[:1], axis=1)
    close = close.reindex(columns=tickers)
    idx = pd.to_datetime(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    return close[~close.index.duplicated(keep="last")].astype(float).sort_index()


def download_close(tickers, start, end, retries=2):
    """Cours ajustés ; les séries vides sont retéléchargées une par une (échecs ponctuels de Yahoo)."""
    tickers = list(tickers)
    close = _yf_close(tickers, start, end, threads=True)
    for _ in range(retries):
        empty = [t for t in tickers if close[t].dropna().empty]
        if not empty:
            break
        for t in empty:
            s = _yf_close([t], start, end, threads=False)[t]
            if not s.dropna().empty:
                close = close.reindex(close.index.union(s.index))
                close[t] = s
    return close


def yahoo_info(ticker):
    try:
        fi = yf.Ticker(ticker).fast_info
        return fi.get("currency"), fi.get("quoteType")
    except Exception:
        return None, None


def normalise_currency(ccy):
    if ccy in MINOR_UNITS:
        return MINOR_UNITS[ccy]
    return (ccy or "USD").upper(), 1.0


def fx_series(ccy, base, start, end, cache):
    """Unités de `base` pour 1 unité de `ccy`."""
    pair = f"{ccy}{base}=X"
    if pair not in cache:
        s = download_close([pair], start, end)[pair]
        if s.dropna().empty:
            inv = f"{base}{ccy}=X"
            s_inv = download_close([inv], start, end)[inv]
            if s_inv.dropna().empty:
                raise EngineError(f"Taux de change {pair} introuvable sur Yahoo Finance.")
            s = 1 / s_inv
        cache[pair] = s.dropna()
    return cache[pair]


def align_calendar(prices, max_ffill=5):
    p = prices[prices.index.dayofweek < 5]
    p = p[~p.isna().all(axis=1)]
    first = max(p[c].first_valid_index() for c in p.columns)
    last = min(p[c].last_valid_index() for c in p.columns)
    youngest = max(p.columns, key=lambda c: p[c].first_valid_index())
    p = p.loc[first:last]
    gaps = p.isna().sum()
    p = p.ffill(limit=max_ffill)
    n_before = len(p)
    p = p.dropna()
    return p, {"first": p.index[0], "last": p.index[-1], "n_obs": len(p),
               "years": (p.index[-1] - p.index[0]).days / 365.25, "youngest": youngest,
               "filled_by_ticker": gaps[gaps > 0].astype(int).to_dict(), "dropped_rows": n_before - len(p)}


def load_prices(tickers, market, start, end, base):
    """Télécharge, convertit dans la devise `base` et aligne les cours. Renvoie un dict."""
    universe = list(dict.fromkeys(list(tickers) + [market]))
    raw = download_close(universe, start, end)
    missing = [t for t in universe if raw[t].dropna().empty]
    if missing:
        raise EngineError(f"Aucune donnée Yahoo Finance pour {', '.join(missing)}. Vérifiez l'orthographe "
                          "(ex. MC.PA pour LVMH, ^GSPC pour le S&P 500).")
    currencies, quote_types, notes = {}, {}, []
    for t in universe:
        ccy, qt = yahoo_info(t)
        if ccy is None:
            notes.append(f"Devise de {t} introuvable : USD supposé.")
        currencies[t], quote_types[t] = ccy or "USD", qt
    cache = {}
    prices = raw.copy()
    for t in universe:
        ccy, factor = normalise_currency(currencies[t])
        if ccy == base:
            prices[t] = raw[t] * factor
        else:
            fx = fx_series(ccy, base, start, end, cache)
            fx = fx.reindex(fx.index.union(raw.index)).ffill().reindex(raw.index)
            prices[t] = raw[t] * factor * fx
    prices, cal = align_calendar(prices)
    meta = {
        "source": "Yahoo Finance via yfinance (cours de clôture ajustés, auto_adjust=True)",
        "yfinance_version": yf.__version__,
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "requested_period": [f"{pd.Timestamp(start):%Y-%m-%d}", f"{pd.Timestamp(end):%Y-%m-%d}"],
        "aligned_period": [f"{cal['first']:%Y-%m-%d}", f"{cal['last']:%Y-%m-%d}"],
        "n_obs": cal["n_obs"], "base_currency": base, "currencies": currencies, "quote_types": quote_types,
        "fx_pairs": sorted(cache), "tickers": list(tickers), "market": market,
        "calendar": {"ffill_max_days": 5, "filled_cells_by_ticker": cal["filled_by_ticker"],
                     "dropped_rows": cal["dropped_rows"]},
    }
    return {"prices": prices, "raw": raw, "currencies": currencies, "quote_types": quote_types,
            "calendar": cal, "meta": meta, "notes": notes}


# ---------------------------------------------------------------------------------------------
# Mesures de risque
# ---------------------------------------------------------------------------------------------
def drawdown(r):
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def sample_var_es(r, alpha):
    r = np.asarray(r)
    q = np.quantile(r, 1 - alpha)
    return -q, -r[r <= q].mean()


def param_var_es(mu_p, sig_p, alpha):
    z = stats.norm.ppf(alpha)
    return z * sig_p - mu_p, sig_p * stats.norm.pdf(z) / (1 - alpha) - mu_p


def cov_root(c):
    """L telle que LLᵀ = Σ : Cholesky, ou décomposition spectrale si Σ est singulière."""
    try:
        return np.linalg.cholesky(c)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh((c + c.T) / 2)
        return vecs * np.sqrt(np.clip(vals, 0, None))


def simulate_assets(mu, c, n, h=1, seed=42):
    z = np.random.default_rng(seed).standard_normal((n, len(mu)))
    return mu * h + z @ cov_root(c * h).T


def model_risk(mu, c, w, h, alphas, n, seed):
    mu_p, sig_p = w @ mu * h, np.sqrt(w @ c @ w * h)
    sims_p = simulate_assets(mu, c, n, h, seed) @ w
    res = {}
    for a in alphas:
        res[("Paramétrique", a)] = param_var_es(mu_p, sig_p, a)
        res[("Monte Carlo", a)] = sample_var_es(sims_p, a)
    return res, sims_p


def kupiec_pof(x, n, p):
    phat = x / n
    lr = -2 * ((xlogy(n - x, 1 - p) + xlogy(x, p)) - (xlogy(n - x, 1 - phat) + xlogy(x, phat)))
    return lr, 1 - stats.chi2.cdf(lr, 1)


def traffic_light(x, n, p):
    c = stats.binom.cdf(x, n, p)
    return "🟢 Vert" if c < 0.95 else ("🟠 Orange" if c < 0.9999 else "🔴 Rouge")


def rolling_var(asset_returns, w, win, alphas, n_bt, seed):
    R = asset_returns.values
    rp = R @ w
    T = len(R)
    Z = np.random.default_rng(seed).standard_normal((n_bt, R.shape[1]))
    zs = {a: stats.norm.ppf(a) for a in alphas}
    out = {f"{m} {a:.1%}": np.empty(T - win) for m in ["Historique", "Paramétrique", "Monte Carlo"] for a in alphas}
    for i, t in enumerate(range(win, T)):
        Rw, rw = R[t - win:t], rp[t - win:t]
        mu, C = Rw.mean(axis=0), np.cov(Rw, rowvar=False)
        mu_p, s_p = mu @ w, np.sqrt(w @ C @ w)
        sim_p = mu_p + Z @ (cov_root(C).T @ w)
        for a in alphas:
            out[f"Historique {a:.1%}"][i] = -np.quantile(rw, 1 - a)
            out[f"Paramétrique {a:.1%}"][i] = zs[a] * s_p - mu_p
            out[f"Monte Carlo {a:.1%}"][i] = -np.quantile(sim_p, 1 - a)
    df = pd.DataFrame(out, index=asset_returns.index[win:])
    df["Rendement réalisé"] = rp[win:]
    return df


def _window_return(series, start, end):
    s = series.dropna()
    if s.empty or s.index[0] > start + pd.Timedelta(days=5) or s.index[-1] < end - pd.Timedelta(days=5):
        return np.nan
    return s.asof(end) / s.asof(start) - 1


def replay_episode(cfg, currencies, betas, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    proxies = list(dict.fromkeys([cfg.market, "^GSPC"]))
    universe = list(dict.fromkeys(cfg.tickers + proxies))
    lo, hi = start - pd.Timedelta(days=10), end + pd.Timedelta(days=5)
    raw = download_close(universe, lo, hi)
    ccys = dict(currencies) | {"^GSPC": "USD"}

    def ret_base(t):
        r = _window_return(raw[t], start, end)
        if np.isnan(r):
            return np.nan, ""
        ccy, _ = normalise_currency(ccys.get(t))
        if ccy == cfg.base_ccy:
            return r, ""
        try:
            fx_r = _window_return(fx_series(ccy, cfg.base_ccy, lo, hi, {}), start, end)
        except EngineError:
            fx_r = np.nan
        if np.isnan(fx_r):
            return r, f" (en {ccy}, change indisponible)"
        return (1 + r) * (1 + fx_r) - 1, ""

    proxy_ret, proxy_name = np.nan, None
    for p in proxies:
        proxy_ret, _ = ret_base(p)
        if not np.isnan(proxy_ret):
            proxy_name = p
            break
    rows = []
    for t, w in zip(cfg.tickers, cfg.weights):
        r, note = ret_base(t)
        source = "Historique" + note
        if np.isnan(r):
            if proxy_name is None:
                return None
            r = max(betas[t] * proxy_ret, -1.0)
            source = f"Proxy : β={betas[t]:.2f} × {proxy_name}"
        rows.append({"Actif": t, "Poids": w, "Rendement": r, "Contribution": w * r, "Source": source})
    return pd.DataFrame(rows).set_index("Actif")


# ---------------------------------------------------------------------------------------------
# Analyse complète
# ---------------------------------------------------------------------------------------------
def run_analysis(cfg: Config, data: dict, progress=lambda frac, msg: None):
    w = np.asarray(cfg.weights, dtype=float)
    V, H, a_ref, ccy = cfg.value, cfg.horizon, cfg.a_ref, cfg.base_ccy
    prices_all, cal, currencies = data["prices"], data["calendar"], data["currencies"]
    res = {"cfg": cfg}

    years = cal["years"]
    if years < 4.95:
        res.setdefault("data_warnings", []).append(
            f"Après alignement, l'historique commun ne couvre que {years:.1f} ans (actif le plus récent : {cal['youngest']}).")

    asset_prices = prices_all[cfg.tickers]
    asset_ret = asset_prices.pct_change().dropna()
    asset_log = np.log(asset_prices).diff().dropna()
    mkt_ret = prices_all[cfg.market].pct_change().dropna()
    port_ret = (asset_ret @ w).rename("Portefeuille")
    port_log = np.log1p(port_ret)
    if len(asset_ret) - cfg.window < 250:
        raise EngineError(f"Fenêtre de {cfg.window} jours trop longue : {len(asset_ret)} rendements disponibles, "
                          f"il en faut fenêtre + 250 pour le backtest. Choisissez au plus {len(asset_ret) - 250} jours.")
    period_note = (f"Source : Yahoo Finance (cours ajustés convertis en {ccy}) — période "
                   f"{cal['first']:%d/%m/%Y} → {cal['last']:%d/%m/%Y}, {cal['n_obs']} jours de bourse.")

    # --- Statistiques descriptives ---
    progress(0.10, "Statistiques descriptives")
    betas = pd.Series({t: np.cov(asset_ret[t], mkt_ret)[0, 1] / mkt_ret.var() for t in cfg.tickers})
    beta_p = float(w @ betas.values)

    def row(lr, r, beta, cls, wt):
        ann_ret, ann_vol = lr.mean() * TRADING_DAYS, lr.std() * np.sqrt(TRADING_DAYS)
        return {"Classe": cls, "Poids": wt, "Rendement ann.": ann_ret, "Volatilité ann.": ann_vol,
                "Sharpe": (ann_ret - cfg.rf) / ann_vol, "Max drawdown": drawdown(r).min(), "Bêta": beta,
                "Skewness": stats.skew(r), "Kurtosis (excès)": stats.kurtosis(r), "Pire jour": r.min()}

    stats_tbl = pd.DataFrame({t: row(asset_log[t], asset_ret[t], betas[t], c, wt)
                              for t, c, wt in zip(cfg.tickers, cfg.classes, w)}).T
    stats_tbl.loc["Portefeuille"] = row(port_log, port_ret, beta_p, "—", 1.0)
    num_cols = stats_tbl.columns.drop("Classe")
    stats_tbl[num_cols] = stats_tbl[num_cols].astype(float)
    cov_full = asset_ret.cov().values
    vol_p_ann = float(np.sqrt(w @ cov_full @ w * TRADING_DAYS))
    vol_w_avg = float(w @ (asset_ret.std().values * np.sqrt(TRADING_DAYS)))
    res.update(asset_ret=asset_ret, port_ret=port_ret, mkt_ret=mkt_ret, betas=betas, beta_p=beta_p,
               stats_tbl=stats_tbl, vol_p_ann=vol_p_ann, vol_w_avg=vol_w_avg, corr=asset_ret.corr(),
               cov_ann=asset_ret.cov() * TRADING_DAYS, wealth=V * (1 + port_ret).cumprod(),
               drawdown=drawdown(port_ret), period_note=period_note)

    # --- VaR / ES ---
    progress(0.20, "VaR et Expected Shortfall")
    R_win, r_win = asset_ret.iloc[-cfg.window:], port_ret.iloc[-cfg.window:]
    mu_vec, cov = R_win.mean().values, R_win.cov().values
    est_note = f"Fenêtre d'estimation : {R_win.index[0]:%d/%m/%Y} → {R_win.index[-1]:%d/%m/%Y} ({cfg.window} jours)."
    horizons = sorted({1, H})
    risk, mc_samples = {}, {}
    for h in horizons:
        mres, sims_p = model_risk(mu_vec, cov, w, h, cfg.alphas, cfg.n_sims, cfg.seed)
        mc_samples[h] = sims_p
        overlap = np.expm1(np.log1p(r_win).rolling(h).sum().dropna()) if h > 1 else None
        for a in cfg.alphas:
            v1, e1 = sample_var_es(r_win, a)
            risk[("Historique", h, a)] = (v1 * np.sqrt(h), e1 * np.sqrt(h))
            if h > 1:
                risk[(f"Historique ({h} j chevauchants)", h, a)] = sample_var_es(overlap, a)
            risk[("Paramétrique", h, a)] = mres[("Paramétrique", a)]
            risk[("Monte Carlo", h, a)] = mres[("Monte Carlo", a)]
    risk_long = pd.DataFrame([{"Méthode": m, "Horizon (j)": h, "Confiance": a, "VaR (%)": v, "ES (%)": e,
                               f"VaR ({ccy})": v * V, f"ES ({ccy})": e * V} for (m, h, a), (v, e) in risk.items()])
    lab = lambda a: f"{a:.0%}" if round(a * 100, 6) == int(round(a * 100, 6)) else f"{a:.1%}"
    var_tables = {}
    for h in horizons:
        sub = risk_long[risk_long["Horizon (j)"] == h]
        tbl = pd.concat([
            sub.pivot(index="Méthode", columns="Confiance", values=f"VaR ({ccy})").rename(columns=lambda a: f"VaR {lab(a)}"),
            sub.pivot(index="Méthode", columns="Confiance", values=f"ES ({ccy})").rename(columns=lambda a: f"ES {lab(a)}"),
        ], axis=1)
        order = [m for m in ["Historique", f"Historique ({h} j chevauchants)", "Paramétrique", "Monte Carlo"] if m in tbl.index]
        var_tables[h] = tbl.loc[order]
    ratio = pd.DataFrame({m: {lab(a): risk[(m, 1, a)][1] / risk[(m, 1, a)][0] for a in cfg.alphas}
                          for m in ["Historique", "Paramétrique", "Monte Carlo"]})

    # Contributions au risque
    z_ref = stats.norm.ppf(a_ref)
    sig_p1 = np.sqrt(w @ cov @ w)
    sims1 = simulate_assets(mu_vec, cov, cfg.n_sims, 1, cfg.seed)
    p1 = sims1 @ w
    tail_mc = p1 <= np.quantile(p1, 1 - a_ref)
    tail_h = r_win.values <= np.quantile(r_win.values, 1 - a_ref)
    contrib = pd.DataFrame({
        "Poids": w,
        "Contribution à la variance": w * (cov @ w) / sig_p1**2,
        f"VaR param. {lab(a_ref)} ({ccy})": w * (z_ref * (cov @ w) / sig_p1 - mu_vec) * V,
        f"ES MC {lab(a_ref)} ({ccy})": -(sims1[tail_mc] * w).mean(axis=0) * V,
        f"ES hist. {lab(a_ref)} ({ccy})": -(R_win.values[tail_h] * w).mean(axis=0) * V,
    }, index=cfg.tickers)
    contrib.loc["Total"] = contrib.sum()
    res.update(risk=risk, risk_long=risk_long, var_tables=var_tables, ratio=ratio, contrib=contrib,
               hist_sample=r_win.values, mc_sample=mc_samples[1], est_note=est_note, horizons=horizons, lab=lab)

    # --- Backtest ---
    progress(0.30, "Backtest glissant (le plus long)")
    bt = rolling_var(asset_ret, w, cfg.window, cfg.alphas, min(cfg.n_sims, 5000), cfg.seed)
    rows = []
    for m in ["Historique", "Paramétrique", "Monte Carlo"]:
        for a in cfg.alphas:
            exc = bt["Rendement réalisé"] < -bt[f"{m} {a:.1%}"]
            n, x = len(exc), int(exc.sum())
            lr, pv = kupiec_pof(x, n, 1 - a)
            x250 = int(exc.iloc[-250:].sum())
            verdict = ("✓ non rejeté" if pv >= 0.05 else
                       ("✗ rejeté (trop d'exceptions)" if x > n * (1 - a) else "✗ rejeté (trop peu : VaR conservatrice)"))
            rows.append({"Méthode": m, "Confiance": f"{a:.1%}", "Jours testés": n, "Exceptions": x,
                         "Attendues": n * (1 - a), "Taux observé": x / n, "LR Kupiec": lr, "p-value": pv,
                         "Kupiec (5 %)": verdict, "Exceptions 250 derniers j": x250,
                         "Feu (250 derniers j)": traffic_light(x250, 250, 1 - a),
                         "Feu (échantillon)": traffic_light(x, n, 1 - a)})
    bt_tbl = pd.DataFrame(rows).set_index(["Méthode", "Confiance"])
    res.update(bt=bt, bt_tbl=bt_tbl)

    # --- Stress tests ---
    progress(0.70, "Stress tests")
    stress_rows = []
    ref_var1, ref_es1 = risk[("Historique", 1, a_ref)]
    ref_varh, ref_esh = risk[("Historique", H, a_ref)]

    def record(name, family, loss, driver="—", measure="Perte du scénario"):
        stress_rows.append({"Scénario": name, "Famille": family, "Mesure": measure, "Perte (%)": loss,
                            f"Perte ({ccy})": loss * V, "Actif moteur": driver})

    episodes = [(n, *EPISODES[n]) for n in cfg.episodes]
    if cfg.custom_episode:
        s_, e_ = cfg.custom_episode
        episodes.append((f"Fenêtre personnalisée {pd.Timestamp(s_):%Y-%m-%d} → {pd.Timestamp(e_):%Y-%m-%d}", s_, e_))
    episode_tables, skipped = [], []
    for name, s_, e_ in episodes:
        tbl = replay_episode(cfg, currencies, betas, s_, e_)
        if tbl is None:
            skipped.append(name)
            continue
        loss = -tbl["Contribution"].sum()
        driver = tbl["Contribution"].idxmin()
        record(name, "Historique", loss, driver)
        episode_tables.append((name, pd.Timestamp(s_), pd.Timestamp(e_), loss, driver, tbl))

    hypo_tables = []
    if cfg.market_shock != 0:
        shocks = np.maximum(betas.values * cfg.market_shock, -1)
        tbl = pd.DataFrame({"Poids": w, "Bêta": betas.values, "Choc": shocks, "Contribution": w * shocks},
                           index=cfg.tickers)
        loss = -tbl["Contribution"].sum()
        name = f"Choc de marché {cfg.market_shock:+.0%} via les bêtas"
        record(name, "Hypothétique", loss, tbl["Contribution"].idxmin())
        hypo_tables.append((name, loss, tbl))
    if cfg.custom_shocks:
        shocks = np.array([cfg.custom_shocks.get(t, 0.0) for t in cfg.tickers])
        tbl = pd.DataFrame({"Poids": w, "Choc": shocks, "Contribution": w * shocks,
                            f"Perte ({ccy})": -w * shocks * V}, index=cfg.tickers)
        loss = -tbl["Contribution"].sum()
        name = "Chocs personnalisés par actif"
        record(name, "Hypothétique", loss, tbl["Contribution"].idxmin())
        hypo_tables.append((name, loss, tbl))

    progress(0.85, "Stress de corrélation, de volatilité et inverse")
    vols = np.sqrt(np.diag(cov))
    corr_star = np.full((len(w), len(w)), cfg.rho_star)
    np.fill_diagonal(corr_star, 1.0)
    corr_label, vol_label = f"Corrélations ρ* = {cfg.rho_star:.2f}", f"Volatilités × {cfg.k_vol:g}"
    model_rows = []
    for label, c in [("Base (estimée)", cov), (corr_label, np.outer(vols, vols) * corr_star),
                     (vol_label, cov * cfg.k_vol**2)]:
        for h in horizons:
            mres, _ = model_risk(mu_vec, c, w, h, cfg.alphas, cfg.n_sims, cfg.seed)
            r_ = {"Scénario": label, "Horizon (j)": h, "Vol. ann.": np.sqrt(w @ c @ w * TRADING_DAYS)}
            for a in cfg.alphas:
                for m, short in [("Paramétrique", "param."), ("Monte Carlo", "MC")]:
                    v, e = mres[(m, a)]
                    r_[f"VaR {lab(a)} {short}"] = v * V
                    r_[f"ES {lab(a)} {short}"] = e * V
            model_rows.append(r_)
    model_tbl = pd.DataFrame(model_rows).set_index(["Scénario", "Horizon (j)"])
    base_param = model_tbl.loc[("Base (estimée)", 1), f"VaR {lab(a_ref)} param."]
    for label, fam in [(corr_label, "Corrélation"), (vol_label, "Volatilité")]:
        record(label, fam, model_tbl.loc[(label, 1), f"VaR {lab(a_ref)} param."] / V,
               measure=f"VaR {lab(a_ref)} param. 1 j stressée")

    L = cfg.loss_limit
    w_cov = cov @ w
    var_p = w @ w_cov
    mu_ph, sig_ph = w @ mu_vec * H, np.sqrt(var_p * H)
    r_star = mu_vec * H - (L + mu_ph) * w_cov / var_p
    d_sigma = (L + mu_ph) / sig_ph
    roll_h = np.expm1(np.log1p(port_ret).rolling(H).sum().dropna())
    rev = pd.DataFrame({
        "Poids": w, "Choc uniforme": -L,
        "Choc isolé nécessaire": [(-L / x if x > 0 and L / x <= 1 else np.nan) for x in w],
        f"Via le marché (β × choc {cfg.market})": betas.values * (-L / beta_p) if beta_p > 0 else np.nan,
        f"Le plus vraisemblable ({H} j)": r_star,
    }, index=cfg.tickers)
    reverse_notes = [f"Choc uniforme : tous les actifs à −{L:.0%}."]
    if beta_p > 0:
        reverse_notes.append(f"Via le marché : il faut que {cfg.market} perde {L / beta_p:.1%} (bêta du portefeuille {beta_p:.2f}).")
    reverse_notes += [
        f"Scénario le plus vraisemblable sous la loi normale à {H} j : à {d_sigma:.1f} σ, probabilité ≈ {stats.norm.cdf(-d_sigma):.2e}.",
        f"Fréquence historique d'une perte ≥ {L:.0%} sur {H} jours glissants : {float((roll_h <= -L).mean()):.2%} "
        f"(pire perte observée sur {H} j : {-roll_h.min():.1%}).",
        "« Choc isolé nécessaire » vide : l'actif seul ne peut pas causer cette perte (il faudrait plus de −100 %).",
    ]
    record(f"Stress inverse : limite {L:.0%}", "Inverse", L, cfg.tickers[int(np.argmin(w * r_star))], measure="Limite de perte")

    stress_tbl = pd.DataFrame(stress_rows).set_index("Scénario")
    stress_tbl[f"÷ VaR {lab(a_ref)} 1 j"] = stress_tbl["Perte (%)"] / ref_var1
    stress_tbl[f"÷ ES {lab(a_ref)} 1 j"] = stress_tbl["Perte (%)"] / ref_es1
    stress_tbl[f"÷ VaR {lab(a_ref)} {H} j"] = stress_tbl["Perte (%)"] / ref_varh
    stress_tbl[f"÷ ES {lab(a_ref)} {H} j"] = stress_tbl["Perte (%)"] / ref_esh
    res.update(stress_tbl=stress_tbl, episode_tables=episode_tables, skipped_episodes=skipped, hypo_tables=hypo_tables,
               model_tbl=model_tbl, rev=rev, reverse_notes=reverse_notes, base_param=base_param,
               ref=(ref_var1, ref_es1, ref_varh, ref_esh), corr_label=corr_label)

    # --- Analyses complémentaires ---
    progress(0.95, "Corrélations de crise et concentration")

    def regime(R):
        c = R.corr().values
        C = R.cov().values
        vp = np.sqrt(w @ C @ w * TRADING_DAYS)
        return {"Jours": len(R), "Corrélation moyenne": c[np.triu_indices_from(c, 1)].mean(), "Vol. portefeuille": vp,
                "Ratio de diversification": float(w @ (R.std().values * np.sqrt(TRADING_DAYS))) / vp}

    regimes = {
        "Tout l'historique": asset_ret,
        "Krach Covid (19/02 – 30/04/2020)": asset_ret.loc["2020-02-19":"2020-04-30"],
        "Année 2022": asset_ret.loc["2022-01-01":"2022-12-31"],
        f"5 % des pires jours de {cfg.market}": asset_ret[mkt_ret <= mkt_ret.quantile(0.05)],
        f"Jours calmes (|{cfg.market}| < médiane)": asset_ret[mkt_ret.abs() < mkt_ret.abs().median()],
    }
    regime_tbl = pd.DataFrame({k: regime(v) for k, v in regimes.items() if len(v) > len(w) + 5}).T

    most_vol = stats_tbl.drop("Portefeuille")["Volatilité ann."].idxmax()
    i_mv = cfg.tickers.index(most_vol)

    def tilt(wv, i, target):
        wv = wv.copy()
        others = np.arange(len(wv)) != i
        wv[others] = wv[others] / wv[others].sum() * (1 - target) if wv[others].sum() > 0 else (1 - target) / others.sum()
        wv[i] = target
        return wv

    conc = {}
    for label, wv in {"Équipondéré": np.full(len(w), 1 / len(w)), "Portefeuille actuel": w,
                      f"50 % sur {most_vol}": tilt(w, i_mv, 0.5), f"100 % sur {most_vol}": tilt(w, i_mv, 1.0)}.items():
        hv, he = sample_var_es(R_win.values @ wv, a_ref)
        pv_, _ = param_var_es(wv @ mu_vec, np.sqrt(wv @ cov @ wv), a_ref)
        conc[label] = {"HHI": float(wv @ wv), "N effectif (1/HHI)": 1 / float(wv @ wv),
                       "Vol. ann.": np.sqrt(wv @ cov @ wv * TRADING_DAYS),
                       f"VaR hist. {lab(a_ref)} 1 j ({ccy})": hv * V, f"ES hist. {lab(a_ref)} 1 j ({ccy})": he * V,
                       f"VaR param. {lab(a_ref)} 1 j ({ccy})": pv_ * V}
    conc_tbl = pd.DataFrame(conc).T
    res.update(regime_tbl=regime_tbl, conc_tbl=conc_tbl, most_vol=most_vol)

    # --- Réponses chiffrées aux 8 questions ---
    eur = lambda x: f"{x:,.0f} {ccy}"
    bt_ref = bt_tbl.xs(f"{a_ref:.1%}", level="Confiance")
    sp = stats_tbl.loc["Portefeuille"]
    no_rev = stress_tbl[stress_tbl["Famille"] != "Inverse"]
    worst = no_rev["Perte (%)"].astype(float).idxmax()
    h_v, h_e = risk[("Historique", 1, a_ref)]
    p_v, p_e = risk[("Paramétrique", 1, a_ref)]
    m_v, _ = risk[("Monte Carlo", 1, a_ref)]
    vol_rank = stats_tbl.drop("Portefeuille")["Volatilité ann."].sort_values(ascending=False)
    k_corr = model_tbl.loc[(corr_label, 1), f"VaR {lab(a_ref)} param."] / base_param
    res["worst"] = worst
    res["answers"] = [
        ("Actif le plus volatil", f"{vol_rank.index[0]} ({vol_rank.iloc[0]:.1%} par an), devant {vol_rank.index[1]} "
                                  f"({vol_rank.iloc[1]:.1%}). Portefeuille : {sp['Volatilité ann.']:.1%}."),
        (f"Backtest à {lab(a_ref)}", " ; ".join(
            f"{m} : {r['Exceptions']} exceptions pour {r['Attendues']:.1f} attendues, Kupiec p = {r['p-value']:.3f} "
            f"({r['Kupiec (5 %)']}), feu {r['Feu (250 derniers j)']}" for m, r in bt_ref.iterrows()) + "."),
        ("Diversification en crise", f"corrélation moyenne {regime_tbl.iloc[0]['Corrélation moyenne']:.2f} sur tout "
                                     f"l'historique, contre {regime_tbl['Corrélation moyenne'].iloc[1:].max():.2f} au pire régime. "
                                     f"Avec ρ* = {cfg.rho_star:.2f}, la VaR paramétrique est multipliée par {k_corr:.2f}."),
        ("Historique vs paramétrique", f"{eur(h_v * V)} vs {eur(p_v * V)}. Skewness du portefeuille {sp['Skewness']:.2f}, "
                                       f"excès de kurtosis {sp['Kurtosis (excès)']:.2f} : les rendements s'écartent de la loi normale."),
        ("Monte Carlo vs historique", f"{eur(m_v * V)} vs {eur(h_v * V)}. Le Monte Carlo suppose une loi normale multivariée "
                                      f"N(μ, Σ) : pas de queues épaisses, mais {cfg.n_sims:,} scénarios et une erreur en 1/√N."),
        ("Apport de l'ES", f"ES/VaR historique = {h_e / h_v:.2f}, contre {p_e / p_v:.2f} sous la loi normale. L'ES mesure la "
                           "gravité des pertes au-delà de la VaR et il est sous-additif (mesure cohérente)."),
        ("Concentration", f"passer à 50 % sur {most_vol} porte la VaR historique de "
                          f"{eur(conc_tbl.loc['Portefeuille actuel', f'VaR hist. {lab(a_ref)} 1 j ({ccy})'])} à "
                          f"{eur(conc_tbl.loc[f'50 % sur {most_vol}', f'VaR hist. {lab(a_ref)} 1 j ({ccy})'])}."),
        ("Pire scénario de stress", f"{worst} ({stress_tbl.loc[worst, 'Perte (%)']:.1%}, {eur(stress_tbl.loc[worst, f'Perte ({ccy})'])}, "
                                    f"soit {stress_tbl.loc[worst, f'÷ VaR {lab(a_ref)} 1 j']:.1f}× la VaR à 1 jour). "
                                    f"Actif moteur : {stress_tbl.loc[worst, 'Actif moteur']}."),
    ]
    progress(1.0, "Terminé")
    return res


def metadata_json(meta):
    return json.dumps(meta, indent=2, ensure_ascii=False, default=str)
