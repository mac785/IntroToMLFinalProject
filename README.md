# ERCOT Load Forecasting
**Predicting Texas Grid Demand with Machine Learning**

Final project for Intro to Machine Learning · Prof. Varun Rai · Spring 2026
Jamie Moseley · EID jbm4577

---

## Overview

Hourly electricity load forecasting for the ERCOT grid (Texas), covering 41,992 hours of data from January 2021 through December 2025. The project trains 8 regression models and 5 classification models on weather and calendar features, then addresses a significant distribution-shift problem caused by Texas demand growth (~2.1 GW/year from data center expansion).

**Key result:** Adding lag features (`load_lag_24h`) and a days-elapsed trend counter reduces RMSE by **83%** on the Jan–Apr 2026 holdout (7,895 MW → 1,378 MW, SVR).

---

## Data Sources

| Source | Description |
|---|---|
| [ERCOT Native Load](https://www.ercot.com/gridinfo/load/load_hist) | Hourly system-wide load (MW), 2021–2025, downloaded as annual CSVs |
| [ERCOT API (NP6-345-CD)](https://api.ercot.com) | 2026 load data fetched via OAuth2, paginated |
| [Open-Meteo Historical Forecast API](https://open-meteo.com) | ERA5 reanalysis weather for Austin, Dallas, Houston, San Antonio |

ERCOT API credentials are required for 2026 data fetching. Store them in a `.env` file at the project root:
```
ERCOT_USERNAME=...
ERCOT_PASSWORD=...
ERCOT_PRIMARY_KEY=...
```

---

## Requirements

```bash
pip install numpy pandas scikit-learn statsmodels torch joblib matplotlib requests python-dotenv ISLP
```

Python 3.10+ recommended.

---

## Project Structure

```
final_project/
├── data/
│   ├── raw/                  # Downloaded ERCOT and weather CSVs
│   └── processed/            # Engineered feature files and result CSVs
├── models/                   # Saved .pkl (sklearn) and .pt (PyTorch) model files
├── notebooks/                # All pipeline scripts (run in order)
│   ├── 01_data_acquisition.py
│   ├── 02_feature_engineering.py
│   ├── 02b–02d_*.py          # Lag, trend, and 2026 feature variants
│   ├── 03_eda.py
│   ├── 04_regression_models.py
│   ├── 04b–04e_*.py          # Lag, trend, bias, and retrain variants
│   ├── 05_classification_models.py
│   ├── 06_live_predict.py    # Run on presentation day
│   ├── 07_approach_comparison.py
│   └── 08_2026_rmse_figure.py
├── figures/                  # Live and 2026 evaluation figures (23–26)
├── figures_archive_prefix/   # EDA, model, and classification figures (01–22)
├── reference_files/          # Syllabus, project guidance, reference decks
├── SLIDES.md                 # Marp slide deck (final deliverable)
└── SLIDES.pdf                # Exported PDF
```

---

## How to Run

Scripts are numbered and intended to run in order from `notebooks/`. Each script saves its outputs (CSVs, model files, figures) so subsequent scripts can load them.

```bash
cd final_project

# Build features
python notebooks/02_feature_engineering.py
python notebooks/02b_feature_engineering_lag.py
python notebooks/02c_feature_engineering_trend.py

# Train models
python notebooks/04_regression_models.py
python notebooks/04b_regression_lag.py
python notebooks/04e_regression_2025train.py
python notebooks/05_classification_models.py

# Generate figures
python notebooks/07_approach_comparison.py   # live evaluation figures
python notebooks/08_2026_rmse_figure.py      # 2026 holdout comparison
```

To fetch live 2026 data (requires `.env` credentials):
```bash
python notebooks/02d_fetch_2026_data.py
```

---

## Key Results

### Regression (2021–2024 train → 2025 test, weather-only)
Best model: **SVR**, RMSE = 6,461 MW, R² = 0.60. All 8 models fall within a ~420 MW band.

### Classification — peak detection (top 10% of hourly load)
At F1-optimal threshold with lag + trend features: **~93% precision, ~93% recall** (Logistic Regression, F1 = 0.939, AUC = 0.997).

### Distribution shift fix (2025 retrain + lag + trend → Jan–Apr 2026 test)
| Model | RMSE | R² |
|---|---|---|
| SVR | 1,378 MW | 0.952 |
| Gradient Boosting | 1,447 MW | 0.947 |
| MLP | 1,475 MW | 0.945 |

**83% RMSE reduction** vs. weather-only baseline.

---

## Slides

The final presentation is in `SLIDES.md` (Marp format). To export to PDF:

```bash
npx @marp-team/marp-cli@latest SLIDES.md --pdf --allow-local-files -o SLIDES.pdf
```
