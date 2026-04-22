# ERCOT Load Forecasting — Project Outline
**Course:** Intro to Machine Learning, Prof. Varun Rai, UT Austin Spring 2026
**Student:** Jamie Moseley (EID: jbm4577)
**Presentation:** April 22, 2026 | **Final Report Due:** April 29, 2026

---

## Phase 1 — Data Acquisition
- Download ERCOT hourly native load (2020–2025, CSV from ercot.com)
- Download NOAA weather data for 4 key Texas stations: Austin (KAUS), Dallas (KDFW), Houston (KHOU), San Antonio (KSAT)
- Merge on timestamp, handle missing values

## Phase 2 — Feature Engineering
- **Time features:** hour of day, day of week, month, season, is_weekend, is_holiday
- **Weather features:** temperature (dominant driver), humidity, wind speed, dew point
- **Derived features:** Cooling Degree Days (CDD), Heating Degree Days (HDD), temp × hour interactions
- **Lag features:** prior hour load, prior day same-hour load
- **Classification label:** `is_peak` = top 10% of hourly load values

## Phase 3 — Exploratory Data Analysis (EDA)
- Load profiles by hour, weekday, season
- Temperature vs. load scatter (the classic "U-shape")
- Correlation heatmap

## Phase 4 — Regression Models (Primary Task: Predict MW)
1. **Linear Regression** — baseline, interpretable
2. **Ridge + Lasso** — regularization comparison, feature selection via coefficient shrinkage
3. **Decision Tree** — simple nonlinear baseline
4. **Random Forest** — ensemble, yields feature importance
5. **Gradient Boosting** (sklearn GradientBoostingRegressor or XGBoost) — typically best on tabular data
6. **Support Vector Regression** — RBF kernel for nonlinear patterns
7. **Neural Network (MLP)** — dense layers via sklearn MLPRegressor or PyTorch

## Phase 5 — Classification Models (Secondary Task: Predict Peak/Non-Peak)
1. **Logistic Regression** (with regularization)
2. **Random Forest Classifier**
- Metrics: confusion matrix, ROC-AUC, precision/recall

## Phase 6 — Model Comparison & Evaluation
- Time-series cross-validation (rolling window — NOT random split; temporal order matters)
- Metrics: RMSE, MAE, R² for regression models
- Feature importance plots (Random Forest, Gradient Boosting)
- Residual analysis to check for systematic bias

## Phase 7 — Live Prediction Demo (Day of Presentation)
- Script (`live_predict.py`) pulls day-of weather forecast from NOAA's free API
- Feeds forecast into best-performing model → outputs predicted hourly load curve for April 22
- Compare live against ERCOT's real-time dashboard during the presentation

---

## File Structure
```
final_project/
  data/
    raw/            ← downloaded CSVs (ERCOT + NOAA)
    processed/      ← merged, cleaned dataset
  notebooks/
    01_data_acquisition.ipynb
    02_feature_engineering.ipynb
    03_eda.ipynb
    04_regression_models.ipynb
    05_classification_models.ipynb
    06_model_comparison.ipynb
  models/           ← saved model objects (.pkl)
  project_notes.md  ← anecdotes, facts, revelations for the presentation
  project_outline.md
  live_predict.py   ← day-of prediction script
```

---

## Deliverables Checklist
- [ ] Project proposal (submitted 2/6/2026)
- [ ] In-class presentation slides (3–5 slides, due 4/22/2026)
- [ ] Final report slide deck (10–15 slides + appendix, due 4/29/2026)
- [ ] Code + data submitted with final report
- [ ] AI usage disclosure statement included in final report
