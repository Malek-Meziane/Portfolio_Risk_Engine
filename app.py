"""Portfolio Risk Engine — application web interactive (Streamlit).

Lancement :  python -m streamlit run app.py      (ou double-clic sur Lancer_app.bat)
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import risk_engine as eng

st.set_page_config(page_title="Portfolio Risk Engine", page_icon="📉", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
C_TAIL, C_BODY, INK2, GRID = "#d03b3b", "#9ec5f4", "#52514e", "#e4e3df"

DEFAULT_PORTFOLIO = pd.DataFrame({
    "Ticker": ["MC.PA", "ASML.AS", "TTE.PA", "JPM", "MSFT", "TLT", "LQD", "GLD", "VNQ"],
    "Classe": ["Actions"] * 5 + ["Obligations", "Obligations", "Matières premières", "Immobilier"],
    "Poids (%)": [10.0, 10, 10, 10, 10, 15, 15, 10, 10],
    "Choc perso (%)": [-20.0, -25, 5, -12, -15, -8, -6, 8, -15],
})
CLASS_HINT = "Actions, Obligations, Matières premières, Immobilier, Crypto, Devises…"


# ---------------------------------------------------------------------------------------------
# Mise en forme
# ---------------------------------------------------------------------------------------------
def eur(x, ccy):
    return "—" if pd.isna(x) else f"{x:,.0f} {ccy}"


def pct(d=1):
    return lambda x: "—" if pd.isna(x) else f"{x:.{d}%}"


def num(d=2):
    return lambda x: "—" if pd.isna(x) else f"{x:,.{d}f}"


def show_table(df, formats=None, default=None, height=None):
    formats = dict(formats or {})
    if default is not None:
        for c in df.columns:
            if c not in formats and pd.api.types.is_numeric_dtype(df[c]):
                formats[c] = default
    kwargs = {"height": height} if height else {}
    st.dataframe(df.style.format(formats, na_rep="—"), width="stretch", **kwargs)


def money_cols(df, ccy):
    return {c: (lambda x, c_=ccy: eur(x, c_)) for c in df.columns if f"({ccy})" in c or str(c).startswith(("VaR", "ES"))}


def style_fig(fig, height=380, title=None):
    fig.update_layout(height=height, title=dict(text=title, x=0, xanchor="left"), template="plotly_white",
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10), font=dict(size=12),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
                      hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ---------------------------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------------------------
def fig_wealth(res, ccy):
    dd = res["drawdown"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                        subplot_titles=(f"Valeur du portefeuille ({ccy})", "Drawdown"))
    fig.add_trace(go.Scatter(x=res["wealth"].index, y=res["wealth"], name="Valeur", line=dict(color=C1, width=2),
                             hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.0f} " + ccy + "<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="Drawdown", fill="tozeroy", line=dict(color=C1, width=1),
                             fillcolor="rgba(42,120,214,0.2)", hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1%}<extra></extra>"),
                  row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    return style_fig(fig, 480).update_layout(showlegend=False)


def fig_corr(corr):
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                               colorscale=[[0, "#1c5cab"], [0.5, "#f0efec"], [1, "#e34948"]],
                               text=corr.values, texttemplate="%{text:.2f}", xgap=2, ygap=2,
                               hovertemplate="%{y} / %{x}<br>ρ = %{z:.2f}<extra></extra>", colorbar=dict(title="ρ")))
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, 460, "Corrélations des rendements quotidiens")


def fig_distribution(sample, var_, es_, title, V, ccy, a_lab):
    pnl = np.asarray(sample) * V / 1e3
    counts, edges = np.histogram(pnl, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2
    v, e = var_ * V / 1e3, es_ * V / 1e3
    fig = go.Figure(go.Bar(x=centers, y=counts, width=(edges[1] - edges[0]) * 0.92,
                           marker_color=[C_TAIL if c <= -v else C_BODY for c in centers],
                           hovertemplate="P&L ≈ %{x:,.1f}k<br>%{y} observations<extra></extra>", showlegend=False))
    fig.add_vline(x=-v, line=dict(color="#0b0b0b", width=2),
                  annotation=dict(text=f"VaR {a_lab} : {v:,.1f}k", font_size=11), annotation_position="top right")
    fig.add_vline(x=-e, line=dict(color="#0b0b0b", width=2, dash="dash"),
                  annotation=dict(text=f"ES {a_lab} : {e:,.1f}k", font_size=11), annotation_position="top left")
    fig.update_xaxes(title=f"P&L quotidien (k{ccy})")
    return style_fig(fig, 340, title)


def fig_contrib(contrib):
    c = contrib.drop("Total")
    fig = go.Figure([
        go.Bar(x=c.index, y=c["Poids"], name="Poids", marker_color=C1, hovertemplate="%{x}<br>Poids %{y:.1%}<extra></extra>"),
        go.Bar(x=c.index, y=c["Contribution à la variance"], name="Contribution au risque", marker_color=C2,
               hovertemplate="%{x}<br>Contribution %{y:.1%}<extra></extra>"),
    ])
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(tickformat=".0%")
    return style_fig(fig, 360, "Poids vs contribution au risque")


def fig_backtest(res, a_ref, V, ccy, a_lab):
    bt = res["bt"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["Rendement réalisé"] * V, name="P&L réalisé",
                             line=dict(color="#b7b6b0", width=1), hovertemplate="%{y:,.0f}<extra>P&L</extra>"))
    for m, c in zip(["Historique", "Paramétrique", "Monte Carlo"], [C1, C2, C4]):
        fig.add_trace(go.Scatter(x=bt.index, y=-bt[f"{m} {a_ref:.1%}"] * V, name=f"−VaR {a_lab} {m.lower()}",
                                 line=dict(color=c, width=1.6), hovertemplate="%{y:,.0f}<extra>" + m + "</extra>"))
    exc = bt["Rendement réalisé"] < -bt[f"Historique {a_ref:.1%}"]
    fig.add_trace(go.Scatter(x=bt.index[exc], y=bt.loc[exc, "Rendement réalisé"] * V, mode="markers",
                             name=f"Exceptions (VaR historique) : {int(exc.sum())}",
                             marker=dict(color=C_TAIL, size=8, line=dict(color="white", width=1.5)),
                             hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.0f}<extra>Exception</extra>"))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title=ccy)
    return style_fig(fig, 440, f"P&L quotidien vs VaR {a_lab} à 1 jour (fenêtre glissante)")


def fig_stress(res, V, ccy, a_lab, H):
    s = res["stress_tbl"]
    vals = s[f"Perte ({ccy})"] / 1e3
    fig = go.Figure(go.Bar(y=s.index, x=vals, orientation="h", marker_color=C1, showlegend=False,
                           text=[f"{v:,.0f}k" for v in vals], textposition="outside",
                           hovertemplate="%{y}<br>Perte %{x:,.1f}k<extra></extra>"))
    n = len(s)
    for ref, dash, lab in zip(res["ref"], ["solid", "dash", "dashdot", "dot"],
                              [f"VaR {a_lab} 1 j", f"ES {a_lab} 1 j", f"VaR {a_lab} {H} j", f"ES {a_lab} {H} j"]):
        x = ref * V / 1e3
        fig.add_trace(go.Scatter(x=[x, x], y=[-0.5, n - 0.5], mode="lines", name=f"{lab} historique : {x:,.0f}k",
                                 line=dict(color=INK2, width=1.5, dash=dash), hoverinfo="name", yaxis="y2"))
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      yaxis2=dict(overlaying="y", range=[n - 0.5, -0.5], visible=False))
    fig.update_xaxes(title=f"Perte (k{ccy})")
    return style_fig(fig, 90 + 48 * n, "Pertes de stress vs VaR et ES historiques")


# ---------------------------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------------------------
st.title("📉 Portfolio Risk Engine")
st.caption("VaR, Expected Shortfall, Monte Carlo, backtest et stress tests sur données Yahoo Finance · "
           "MSc FBD · NEOMA Business School")

with st.sidebar:
    st.header("Paramètres")
    st.subheader("Données")
    base_ccy = st.text_input("Devise de référence", "EUR", max_chars=3).strip().upper()
    value = st.number_input(f"Valeur du portefeuille ({base_ccy})", min_value=1_000.0, value=1_000_000.0,
                            step=100_000.0, format="%.0f")
    c1, c2 = st.columns(2)
    start = c1.date_input("Début", pd.Timestamp("2018-01-01"), min_value=pd.Timestamp("1990-01-01"), format="DD/MM/YYYY")
    end = c2.date_input("Fin", pd.Timestamp.today(), format="DD/MM/YYYY")
    market = st.text_input("Indice de marché (bêtas, proxys)", "SPY").strip().upper()

    st.subheader("Mesures de risque")
    conf = st.multiselect("Niveaux de confiance (%)", [90.0, 95.0, 97.5, 99.0, 99.5], default=[95.0, 99.0])
    horizon = st.slider("Horizon (jours de bourse)", 1, 60, 10, help="La VaR à 1 jour est toujours calculée.")
    window = st.select_slider("Fenêtre d'estimation (jours)", [250, 500, 756, 1000, 1260], value=756,
                              help="756 jours ≈ 3 ans. Le backtest a besoin de fenêtre + 250 jours de données.")
    n_sims = st.select_slider("Simulations Monte Carlo", [1_000, 5_000, 10_000, 50_000, 100_000], value=10_000)
    rf = st.number_input("Taux sans risque annuel (%)", -5.0, 20.0, 3.0, 0.25)

    st.subheader("Stress tests")
    episodes = st.multiselect("Épisodes historiques à rejouer", list(eng.EPISODES), default=[
        "Crise financière mondiale 2007-2009", "Krach du Covid (févr.-mars 2020)", "Choc de taux 2022"])
    use_custom_ep = st.checkbox("Ajouter une fenêtre personnalisée")
    custom_ep = None
    if use_custom_ep:
        e1, e2 = st.columns(2)
        ce_s = e1.date_input("Début fenêtre", pd.Timestamp("2018-10-01"), format="DD/MM/YYYY")
        ce_e = e2.date_input("Fin fenêtre", pd.Timestamp("2018-12-24"), format="DD/MM/YYYY")
        custom_ep = (pd.Timestamp(ce_s), pd.Timestamp(ce_e))
    mkt_shock = st.slider(f"Choc de marché sur {market} via les bêtas (%)", -60, 30, -20)
    rho_star = st.slider("Corrélation de crise ρ*", 0.80, 1.00, 0.90, 0.01)
    k_vol = st.slider("Multiplicateur de volatilité k", 1.5, 3.0, 2.0, 0.1)
    loss_limit = st.slider("Limite de perte du stress inverse (%)", 5, 60, 20)

    st.divider()
    strict = st.toggle("Règles du projet bloquantes", True,
                       help="≥ 6 tickers, ≥ 3 classes d'actifs, ≥ 5 ans incluant 2020 et 2022.")
    run = st.button("▶ Lancer l'analyse", type="primary", width="stretch")

st.subheader("1 · Votre portefeuille")
st.caption(f"Modifiez le tableau : ajoutez ou supprimez des lignes, changez les tickers Yahoo Finance (ex. MC.PA, "
           f"BTC-USD, ^GSPC), les classes ({CLASS_HINT}) et les poids. La colonne « Choc perso » alimente le stress "
           "hypothétique par actif (laisser vide ou 0 pour ignorer).")
portfolio = st.data_editor(
    DEFAULT_PORTFOLIO, num_rows="dynamic", width="stretch", key="portfolio", hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", required=True, help="Symbole Yahoo Finance"),
        "Classe": st.column_config.TextColumn("Classe d'actifs", required=True),
        "Poids (%)": st.column_config.NumberColumn("Poids (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f %%"),
        "Choc perso (%)": st.column_config.NumberColumn("Choc perso (%)", min_value=-99.0, max_value=1000.0, step=1.0,
                                                        format="%+.0f %%"),
    })
portfolio = portfolio[portfolio["Ticker"].fillna("").str.strip() != ""].copy()
portfolio["Ticker"] = portfolio["Ticker"].str.strip().str.upper()
portfolio["Classe"] = portfolio["Classe"].fillna("").astype(str).str.strip()
total = portfolio["Poids (%)"].fillna(0).sum()
n_cls = portfolio["Classe"].str.lower().nunique()
msg = f"**{len(portfolio)} actifs · {n_cls} classes · total des poids : {total:g} %**"
if abs(total - 100) < 1e-6:
    st.success(msg + " ✓")
else:
    st.warning(msg + " — le total doit faire 100 %.")

cfg = eng.Config(
    tickers=portfolio["Ticker"].tolist(), classes=portfolio["Classe"].tolist(),
    weights=portfolio["Poids (%)"].to_numpy(dtype=float) / 100, base_ccy=base_ccy, value=float(value),
    start=pd.Timestamp(start), end=pd.Timestamp(end), alphas=sorted(a / 100 for a in conf), horizon=int(horizon),
    window=int(window), n_sims=int(n_sims), rf=rf / 100, market=market, episodes=episodes, custom_episode=custom_ep,
    market_shock=mkt_shock / 100,
    custom_shocks={t: s / 100 for t, s in zip(portfolio["Ticker"], portfolio["Choc perso (%)"]) if pd.notna(s) and s != 0},
    rho_star=float(rho_star), k_vol=float(k_vol), loss_limit=loss_limit / 100,
)
signature = repr(cfg)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(tickers, market_, start_, end_, base_):
    return eng.load_prices(list(tickers), market_, start_, end_, base_)


# ?autorun=1 dans l'URL : lance l'analyse au premier chargement (pratique pour une démonstration)
if st.query_params.get("autorun") == "1" and "res" not in st.session_state:
    run = True

if run:
    errors, warnings = eng.validate(cfg, strict)
    if errors:
        st.error("**Impossible de lancer l'analyse :**\n\n" + "\n".join(f"- {e}" for e in errors))
        st.stop()
    for wmsg in warnings:
        st.warning(wmsg)
    try:
        with st.spinner("Téléchargement des cours Yahoo Finance…"):
            data = cached_prices(tuple(cfg.tickers), cfg.market, cfg.start, cfg.end, cfg.base_ccy)
        DATA_DIR.mkdir(exist_ok=True)
        data["prices"].to_csv(DATA_DIR / f"prices_{cfg.base_ccy}.csv")
        (DATA_DIR / "metadata.json").write_text(eng.metadata_json(data["meta"]), encoding="utf-8")
        bar = st.progress(0.0, "Calcul…")
        res = eng.run_analysis(cfg, data, progress=lambda f, m: bar.progress(f, m))
        bar.empty()
    except eng.EngineError as exc:
        st.error(str(exc))
        st.stop()
    st.session_state.update(res=res, data=data, signature=signature)

if "res" not in st.session_state:
    st.info("👈 Réglez les paramètres puis cliquez sur **Lancer l'analyse**. Un calcul complet prend 10 à 30 secondes.")
    st.stop()

# ---------------------------------------------------------------------------------------------
# Résultats
# ---------------------------------------------------------------------------------------------
res, data = st.session_state["res"], st.session_state["data"]
rc = res["cfg"]
ccy, V, H, a_ref, lab = rc.base_ccy, rc.value, rc.horizon, rc.a_ref, res["lab"]
a_lab = lab(a_ref)
EUR = lambda x: eur(x, ccy)

st.subheader("2 · Résultats")
if st.session_state.get("signature") != signature:
    st.info("Les paramètres ont changé depuis le dernier calcul : cliquez sur **Lancer l'analyse** pour mettre à jour.")
st.caption(" · ".join(f"{t} {w:.0%}" for t, w in zip(rc.tickers, rc.weights)) + "  —  " + res["period_note"])
for note in data.get("notes", []) + res.get("data_warnings", []) + [
        f"Épisode ignoré faute de données : {n}" for n in res["skipped_episodes"]]:
    st.warning(note)

sp = res["stats_tbl"].loc["Portefeuille"]
ref_var1, ref_es1, ref_varh, _ = res["ref"]
bt_ref = res["bt_tbl"].loc[("Historique", f"{a_ref:.1%}")]
worst = res["worst"]
k = st.columns(4)
k[0].metric("Rendement annualisé", f"{sp['Rendement ann.']:.1%}", f"Sharpe {sp['Sharpe']:.2f}", delta_color="off", delta_arrow="off", border=True)
k[1].metric("Volatilité annualisée", f"{sp['Volatilité ann.']:.1%}", f"actifs seuls : {res['vol_w_avg']:.1%}",
            delta_color="off", delta_arrow="off", border=True)
k[2].metric(f"VaR {a_lab} · 1 jour", EUR(ref_var1 * V), f"{H} jours : {EUR(ref_varh * V)}", delta_color="off", delta_arrow="off", border=True)
k[3].metric(f"ES {a_lab} · 1 jour", EUR(ref_es1 * V), f"ES / VaR = {ref_es1 / ref_var1:.2f}", delta_color="off", delta_arrow="off", border=True)
k = st.columns(4)
k[0].metric("Max drawdown", f"{sp['Max drawdown']:.1%}", f"le {res['drawdown'].idxmin():%d/%m/%Y}", delta_color="off", delta_arrow="off", border=True)
k[1].metric(f"Backtest VaR {a_lab}", bt_ref["Feu (250 derniers j)"],
            f"{bt_ref['Exceptions']} exceptions / {bt_ref['Attendues']:.1f} attendues", delta_color="off", delta_arrow="off", border=True)
k[2].metric("Pire stress", f"−{res['stress_tbl'].loc[worst, 'Perte (%)']:.1%}", worst, delta_color="off", delta_arrow="off", border=True)
k[3].metric("Bêta du portefeuille", f"{res['beta_p']:.2f}", f"vs {rc.market}", delta_color="off", delta_arrow="off", border=True)

tabs = st.tabs(["🧭 Synthèse", "📈 Performance", "🎯 VaR & ES", "🧪 Backtest", "🌪️ Stress tests", "🔗 Analyses",
                "💾 Données & export"])

with tabs[0]:
    st.markdown("#### Réponses chiffrées aux 8 questions du projet")
    st.caption("Tirées automatiquement des résultats. L'interprétation reste à rédiger dans le rapport.")
    for i, (q, a) in enumerate(res["answers"], 1):
        st.markdown(f"**{i}. {q}** : {a}")

with tabs[1]:
    show_table(res["stats_tbl"], {"Poids": pct(1), "Rendement ann.": pct(1), "Volatilité ann.": pct(1), "Sharpe": num(),
                                  "Max drawdown": pct(1), "Bêta": num(), "Skewness": num(), "Kurtosis (excès)": num(),
                                  "Pire jour": pct(2)})
    st.caption(f"Diversification : volatilité du portefeuille {res['vol_p_ann']:.1%} contre {res['vol_w_avg']:.1%} pour la "
               "moyenne pondérée des volatilités. Rendements et volatilités annualisés à partir des rendements log (× 252, × √252).")
    st.plotly_chart(fig_wealth(res, ccy), width="stretch")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig_corr(res["corr"]), width="stretch")
    with c2:
        st.markdown("**Matrice de covariance annualisée Σ**")
        show_table(res["cov_ann"], default=num(4))

with tabs[2]:
    st.caption(res["est_note"])
    for h, tbl in res["var_tables"].items():
        st.markdown(f"**Horizon {h} jour{'s' if h > 1 else ''}**")
        show_table(tbl, default=EUR)
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_distribution(res["hist_sample"], *res["risk"][("Historique", 1, a_ref)],
                                     "Distribution historique", V, ccy, a_lab), width="stretch")
    c2.plotly_chart(fig_distribution(res["mc_sample"], *res["risk"][("Monte Carlo", 1, a_ref)],
                                     f"Distribution Monte Carlo ({rc.n_sims:,} scénarios)", V, ccy, a_lab), width="stretch")
    st.markdown("**Ratio ES / VaR à 1 jour** (loi normale : 1,25 à 95 %, 1,15 à 99 %)")
    show_table(res["ratio"], default=num())
    st.markdown(f"**Contribution au risque par actif** (1 jour, {a_lab})")
    show_table(res["contrib"], {"Poids": pct(1), "Contribution à la variance": pct(1)} | money_cols(res["contrib"], ccy))
    st.plotly_chart(fig_contrib(res["contrib"]), width="stretch")

with tabs[3]:
    bt = res["bt"]
    st.caption(f"VaR à 1 jour réestimée chaque jour sur les {rc.window} jours précédents "
               f"({bt.index[0]:%d/%m/%Y} → {bt.index[-1]:%d/%m/%Y}). Kupiec : rejet si p-value < 5 %. "
               "Feux tricolores : loi binomiale, référence bâloise 250 jours à 99 % (vert 0-4, orange 5-9, rouge ≥ 10).")
    show_table(res["bt_tbl"], {"Attendues": num(1), "Taux observé": pct(2), "LR Kupiec": num(), "p-value": num(3)})
    st.plotly_chart(fig_backtest(res, a_ref, V, ccy, a_lab), width="stretch")

with tabs[4]:
    st.caption(f"Chaque perte est rapportée à la VaR et à l'ES historiques à {a_lab}, à 1 jour et à {H} jours.")
    s = res["stress_tbl"]
    show_table(s, {"Perte (%)": pct(2), f"Perte ({ccy})": EUR} | {c: num(1) for c in s.columns if c.startswith("÷")})
    st.plotly_chart(fig_stress(res, V, ccy, a_lab, H), width="stretch")
    st.markdown("#### Détail des scénarios")
    for name, s_, e_, loss, driver, tbl in res["episode_tables"]:
        with st.expander(f"📅 {name} : −{loss:.2%} ({EUR(loss * V)}) · actif moteur {driver}"):
            st.caption(f"Du {s_:%d/%m/%Y} au {e_:%d/%m/%Y}, achat-conservation sur les poids actuels.")
            show_table(tbl, {"Poids": pct(1), "Rendement": pct(2), "Contribution": pct(2)})
    for name, loss, tbl in res["hypo_tables"]:
        with st.expander(f"✍️ {name} : −{loss:.2%} ({EUR(loss * V)})"):
            show_table(tbl, {"Poids": pct(1), "Bêta": num(), "Choc": pct(2), "Contribution": pct(2)} | money_cols(tbl, ccy))
    with st.expander("🔗 Stress de corrélation et de volatilité"):
        st.caption("Moyenne μ inchangée, mêmes tirages aléatoires que le cas de base.")
        show_table(res["model_tbl"], {"Vol. ann.": pct(1)}, default=EUR)
    with st.expander(f"↩️ Stress inverse : perte de {rc.loss_limit:.0%} = {EUR(rc.loss_limit * V)}"):
        show_table(res["rev"], default=pct(1))
        for n_ in res["reverse_notes"]:
            st.markdown(f"- {n_}")

with tabs[5]:
    st.markdown("**La diversification selon le régime de marché**")
    show_table(res["regime_tbl"], {"Jours": num(0), "Corrélation moyenne": num(), "Vol. portefeuille": pct(1),
                                   "Ratio de diversification": num()})
    st.caption("Ratio de diversification = moyenne pondérée des volatilités / volatilité du portefeuille (1 = aucune diversification).")
    st.markdown(f"**Concentration sur l'actif le plus volatil ({res['most_vol']})**")
    show_table(res["conc_tbl"], {"HHI": num(3), "N effectif (1/HHI)": num(1), "Vol. ann.": pct(1)} | money_cols(res["conc_tbl"], ccy))

with tabs[6]:
    meta = data["meta"]
    st.markdown(f"**Source** : {meta['source']}  \n**Téléchargé le** : {meta['downloaded_at']}  \n"
                f"**Période alignée** : {meta['aligned_period'][0]} → {meta['aligned_period'][1]} ({meta['n_obs']} jours)  \n"
                f"**Devises de cotation** : {', '.join(f'{t} {c}' for t, c in meta['currencies'].items())}  \n"
                f"**Paires de change** : {', '.join(meta['fx_pairs']) or 'aucune'}")
    st.caption(f"Le jeu de données est aussi enregistré dans {DATA_DIR / f'prices_{ccy}.csv'} et {DATA_DIR / 'metadata.json'}.")
    tables = {"prices.csv": data["prices"], "statistiques.csv": res["stats_tbl"], "var_es.csv": res["risk_long"],
              "contributions.csv": res["contrib"], "backtest_resume.csv": res["bt_tbl"], "backtest_series.csv": res["bt"],
              "stress_tests.csv": res["stress_tbl"], "stress_correlation_volatilite.csv": res["model_tbl"],
              "stress_inverse.csv": res["rev"], "regimes.csv": res["regime_tbl"], "concentration.csv": res["conc_tbl"]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            z.writestr(name, df.to_csv())
        z.writestr("metadata.json", eng.metadata_json(meta))
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ Cours (CSV)", data["prices"].to_csv().encode(), f"prices_{ccy}.csv", "text/csv", width="stretch")
    c2.download_button("⬇ Métadonnées (JSON)", eng.metadata_json(meta).encode(), "metadata.json", "application/json",
                       width="stretch")
    c3.download_button("⬇ Tous les résultats (ZIP)", buf.getvalue(), "portfolio_risk_results.zip", "application/zip",
                       width="stretch")
    st.dataframe(data["prices"].tail(10), width="stretch")
