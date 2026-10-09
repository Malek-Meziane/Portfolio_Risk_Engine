# Portfolio Risk Engine

Moteur de risque de portefeuille : VaR / ES historique, paramétrique et Monte Carlo, backtesting (Kupiec, feux tricolores) et module de stress (scénarios historiques, hypothétiques, de corrélation et inverses).

## Contenu

| Fichier | Rôle |
|---|---|
| `Portfolio_Risk_Engine.ipynb` | Notebook principal (reproductible, `seed = 42`) |
| `risk_engine.py` | Moteur de calcul (mêmes méthodes que le notebook) |
| `app.py` / `Lancer_app.bat` | Application web interactive (Streamlit) |
| `data/` | Jeu de données sauvegardé (CSV) + `metadata.json` (source, dates, devises) |
| `outputs/` | Graphiques et tableaux de résultats |

## Données

Cours de clôture ajustés téléchargés depuis Yahoo Finance via `yfinance` (`auto_adjust=True`), convertis en EUR.
Période : 2018-01-02 → 2026-10-09 (2 268 observations). Détails dans `data/metadata.json`.

## Lancer

```bash
pip install -r requirements.txt
jupyter notebook Portfolio_Risk_Engine.ipynb   # notebook
streamlit run app.py                           # application interactive
```
