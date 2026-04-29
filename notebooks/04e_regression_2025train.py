"""
Phase 4e: Regression Models — 2025 Retrain (Approach 2C)

Trains all 8 models on the FULL 2021–2025 dataset (features_nolag.csv,
features_lag.csv, features_trend.csv) and evaluates on 2026 test data
(features_nolag_2026.csv, features_lag_2026.csv, features_trend_2026.csv).

Models saved with suffixes:
  _2025train         — trained on nolag 2021-2025, tested on 2026
  _2025train_lag     — trained on lag   2021-2025, tested on 2026
  _2025train_trend   — trained on trend 2021-2025, tested on 2026

Results: data/processed/regression_2025train_results.csv
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import (RandomForestRegressor as RF,
                              GradientBoostingRegressor as GBR)
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ISLP.models import ModelSpec as MS
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset
from torchmetrics import MeanAbsoluteError, R2Score
from ISLP.torch import SimpleDataModule, SimpleModule, ErrorTracker, rec_num_workers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path
import joblib
import warnings

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

ROOT   = Path(__file__).resolve().parent.parent
PROC   = ROOT / "data" / "processed"
MODELS = ROOT / "models"

FEATURE_COLS = [
    'temperature_2m_Austin', 'temperature_2m_Dallas',
    'temperature_2m_Houston', 'temperature_2m_SanAntonio',
    'temperature_2m_avg', 'apparent_temperature_avg',
    'relative_humidity_2m_avg', 'dew_point_2m_avg',
    'wind_speed_10m_avg', 'shortwave_radiation_avg',
    'CDH', 'HDH', 'temp_sq', 'apparent_temp_delta_avg',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'is_weekend', 'is_holiday', 'is_workday', 'season',
    'temp_x_hour_sin', 'CDH_x_hour_sin', 'temp_x_is_weekend',
]
LAG_COLS   = ['load_lag_24h', 'load_lag_168h']
TREND_COLS = ['days_elapsed']
TARGET     = 'Load_MW'


def metrics(y_true, y_pred, label="", variant=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"  {label:30s}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return dict(model=label, variant=variant, RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)


class ERCOTModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.flatten    = nn.Flatten()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1))
    def forward(self, x):
        return torch.flatten(self.sequential(self.flatten(x)))


def train_and_eval(train_df, test_df, feat_cols, suffix, variant_name):
    """Train all 8 models on train_df, evaluate on test_df. Save with given suffix."""
    tscv    = skm.TimeSeriesSplit(n_splits=5)
    results = []

    X_train = train_df[feat_cols].values
    y_train = train_df[TARGET].values
    X_test  = test_df[feat_cols].values
    y_test  = test_df[TARGET].values

    print(f"  Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows  |"
          f"  Features: {len(feat_cols)}")

    # ── 1. OLS ────────────────────────────────────────────────────────────────
    print("  --- OLS ---")
    design   = MS(feat_cols).fit(train_df)
    ols_res  = sm.OLS(train_df[TARGET], design.transform(train_df)).fit()
    ols_pred = ols_res.predict(design.transform(test_df)).values
    results.append(metrics(y_test, ols_pred, "OLS", variant_name))
    print(f"    R² (train): {ols_res.rsquared:.4f}")

    # ── 2. Ridge ──────────────────────────────────────────────────────────────
    print("  --- Ridge ---")
    lambdas    = 10 ** np.linspace(6, -3, 100)
    ridge_pipe = Pipeline([('scaler', StandardScaler()),
                           ('ridge', skl.ElasticNetCV(alphas=lambdas, l1_ratio=0,
                                                      cv=tscv, max_iter=10000))])
    ridge_pipe.fit(X_train, y_train)
    print(f"    Best λ: {ridge_pipe.named_steps['ridge'].alpha_:.4f}")
    results.append(metrics(y_test, ridge_pipe.predict(X_test), "Ridge", variant_name))
    joblib.dump(ridge_pipe, MODELS / f"ridge{suffix}.pkl")

    # ── 3. Lasso ──────────────────────────────────────────────────────────────
    print("  --- Lasso ---")
    lasso_pipe = Pipeline([('scaler', StandardScaler()),
                           ('lasso', skl.ElasticNetCV(n_alphas=100, l1_ratio=1,
                                                      cv=tscv, max_iter=10000))])
    lasso_pipe.fit(X_train, y_train)
    n_zero = np.sum(lasso_pipe.named_steps['lasso'].coef_ == 0)
    print(f"    Best λ: {lasso_pipe.named_steps['lasso'].alpha_:.4f}  Zeroed: {n_zero}/{len(feat_cols)}")
    results.append(metrics(y_test, lasso_pipe.predict(X_test), "Lasso", variant_name))
    joblib.dump(lasso_pipe, MODELS / f"lasso{suffix}.pkl")

    # ── 4. Decision Tree ──────────────────────────────────────────────────────
    print("  --- Decision Tree ---")
    grid_dtr = skm.GridSearchCV(
        DTR(random_state=0),
        {'max_depth': [3, 5, 8, 12, None], 'min_samples_leaf': [10, 50, 100]},
        cv=tscv, scoring='neg_mean_squared_error', refit=True)
    grid_dtr.fit(X_train, y_train)
    print(f"    Best params: {grid_dtr.best_params_}")
    results.append(metrics(y_test, grid_dtr.predict(X_test), "Decision Tree", variant_name))
    joblib.dump(grid_dtr.best_estimator_, MODELS / f"decision_tree{suffix}.pkl")

    # ── 5. Random Forest ──────────────────────────────────────────────────────
    print("  --- Random Forest ---")
    rf = RF(n_estimators=300, max_features='sqrt', min_samples_leaf=5,
            random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(metrics(y_test, rf.predict(X_test), "Random Forest", variant_name))
    joblib.dump(rf, MODELS / f"random_forest{suffix}.pkl")

    # ── 6. Gradient Boosting ──────────────────────────────────────────────────
    print("  --- Gradient Boosting ---")
    gbr = GBR(n_estimators=500, learning_rate=0.05, max_depth=5,
              min_samples_leaf=10, subsample=0.8, random_state=0)
    gbr.fit(X_train, y_train)
    results.append(metrics(y_test, gbr.predict(X_test), "Gradient Boosting", variant_name))
    joblib.dump(gbr, MODELS / f"gradient_boosting{suffix}.pkl")

    # ── 7. SVR ────────────────────────────────────────────────────────────────
    print("  --- SVR ---")
    svr_pipe = Pipeline([('scaler', StandardScaler()),
                         ('svr', SVR(kernel='rbf', C=1e4, epsilon=500, gamma='scale'))])
    svr_pipe.fit(X_train, y_train)
    results.append(metrics(y_test, svr_pipe.predict(X_test), "SVR", variant_name))
    joblib.dump(svr_pipe, MODELS / f"svr{suffix}.pkl")

    # ── 8. MLP ────────────────────────────────────────────────────────────────
    print("  --- MLP ---")
    mlp_scaler  = StandardScaler()
    X_train_np  = mlp_scaler.fit_transform(X_train)
    X_test_np   = mlp_scaler.transform(X_test)
    X_train_t   = torch.tensor(X_train_np.astype(np.float32))
    y_train_t   = torch.tensor(y_train.astype(np.float32))
    X_test_t    = torch.tensor(X_test_np.astype(np.float32))
    y_test_t    = torch.tensor(y_test.astype(np.float32))

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds  = TensorDataset(X_test_t,  y_test_t)
    mlp_dm   = SimpleDataModule(train_ds, test_ds,
                                batch_size=1024,
                                num_workers=min(4, rec_num_workers()),
                                validation=test_ds)

    mlp_model  = ERCOTModel(X_train_t.shape[1])
    mlp_module = SimpleModule.regression(
        mlp_model,
        optimizer=Adam(mlp_model.parameters(), lr=1e-3, weight_decay=1e-5),
        metrics={'mae': MeanAbsoluteError(), 'r2': R2Score()})
    mlp_trainer = Trainer(
        max_epochs=200, accelerator='gpu', devices=1,
        precision='bf16-mixed', gradient_clip_val=1.0,
        log_every_n_steps=20,
        logger=CSVLogger('logs', name=f'ERCOT_MLP{suffix}'),
        callbacks=[ErrorTracker()], enable_progress_bar=True)
    mlp_trainer.fit(mlp_module, datamodule=mlp_dm)
    mlp_trainer.test(mlp_module, datamodule=mlp_dm)

    mlp_model.eval()
    with torch.no_grad():
        mlp_pred = mlp_module(X_test_t).numpy()
    results.append(metrics(y_test, mlp_pred, "MLP", variant_name))

    torch.save(mlp_model.state_dict(), MODELS / f"mlp_state{suffix}.pt")
    joblib.dump(mlp_scaler, MODELS / f"mlp_scaler{suffix}.pkl")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
print("=== Phase 4e: Regression Models — 2025 Retrain (Approach 2C) ===\n")

all_results = []

# ── Variant A: No-lag ─────────────────────────────────────────────────────────
print("=== Variant A: No-lag  (suffix: _2025train) ===")
train_a = pd.read_csv(PROC / "features_nolag.csv", parse_dates=["HourEnding"])
test_a  = pd.read_csv(PROC / "features_nolag_2026.csv", parse_dates=["HourEnding"])
train_a = train_a.sort_values("HourEnding").reset_index(drop=True)
test_a  = test_a.sort_values("HourEnding").reset_index(drop=True)
print(f"  Train: {train_a['HourEnding'].min()} → {train_a['HourEnding'].max()}")
print(f"  Test:  {test_a['HourEnding'].min()} → {test_a['HourEnding'].max()}\n")

res_a = train_and_eval(train_a, test_a, FEATURE_COLS, "_2025train", "2025train_nolag")
all_results.extend(res_a)

# ── Variant B: Lag features ───────────────────────────────────────────────────
print("\n=== Variant B: Lag features  (suffix: _2025train_lag) ===")
train_b = pd.read_csv(PROC / "features_lag.csv", parse_dates=["HourEnding"])
test_b  = pd.read_csv(PROC / "features_lag_2026.csv", parse_dates=["HourEnding"])
train_b = train_b.sort_values("HourEnding").reset_index(drop=True)
test_b  = test_b.sort_values("HourEnding").reset_index(drop=True)
print(f"  Train: {train_b['HourEnding'].min()} → {train_b['HourEnding'].max()}")
print(f"  Test:  {test_b['HourEnding'].min()} → {test_b['HourEnding'].max()}\n")

res_b = train_and_eval(train_b, test_b, FEATURE_COLS + LAG_COLS, "_2025train_lag", "2025train_lag")
all_results.extend(res_b)

# ── Variant C: Trend feature ──────────────────────────────────────────────────
print("\n=== Variant C: Days-elapsed trend  (suffix: _2025train_trend) ===")
train_c = pd.read_csv(PROC / "features_trend.csv", parse_dates=["HourEnding"])
test_c  = pd.read_csv(PROC / "features_trend_2026.csv", parse_dates=["HourEnding"])
train_c = train_c.sort_values("HourEnding").reset_index(drop=True)
test_c  = test_c.sort_values("HourEnding").reset_index(drop=True)
print(f"  Train: {train_c['HourEnding'].min()} → {train_c['HourEnding'].max()}")
print(f"  Test:  {test_c['HourEnding'].min()} → {test_c['HourEnding'].max()}\n")

res_c = train_and_eval(train_c, test_c, FEATURE_COLS + TREND_COLS, "_2025train_trend", "2025train_trend")
all_results.extend(res_c)

# ── Variant D: Lag + Trend (combined) ─────────────────────────────────────────
print("\n=== Variant D: Lag + Trend  (suffix: _2025train_lag_trend) ===")

train_d = train_b.merge(
    train_c[["HourEnding", "days_elapsed"]], on="HourEnding", how="inner")
test_d  = test_b.merge(
    test_c[["HourEnding", "days_elapsed"]], on="HourEnding", how="inner")

train_d = train_d.sort_values("HourEnding").reset_index(drop=True)
test_d  = test_d.sort_values("HourEnding").reset_index(drop=True)

train_d.to_csv(PROC / "features_lag_trend.csv", index=False)
test_d.to_csv(PROC / "features_lag_trend_2026.csv", index=False)
print(f"  Saved features_lag_trend.csv ({len(train_d):,} rows)")
print(f"  Saved features_lag_trend_2026.csv ({len(test_d):,} rows)")
print(f"  Train: {train_d['HourEnding'].min()} → {train_d['HourEnding'].max()}")
print(f"  Test:  {test_d['HourEnding'].min()} → {test_d['HourEnding'].max()}\n")

res_d = train_and_eval(train_d, test_d, FEATURE_COLS + LAG_COLS + TREND_COLS,
                       "_2025train_lag_trend", "2025train_lag_trend")
all_results.extend(res_d)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== 2025 Retrain Results Summary (Test Set: Jan–Apr 2026) ===")
res_df = pd.DataFrame(all_results)
for variant_name, grp in res_df.groupby("variant"):
    print(f"\n--- {variant_name} ---")
    print(grp.drop(columns="variant").set_index("model").sort_values("RMSE").round(2).to_string())

res_df.to_csv(PROC / "regression_2025train_results.csv", index=False)
print(f"\nSaved → {PROC / 'regression_2025train_results.csv'}")
print("Phase 4e complete.")
