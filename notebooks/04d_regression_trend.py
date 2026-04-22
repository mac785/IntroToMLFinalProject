"""
Phase 4d: Regression Models — Days-Elapsed Trend Feature (Approach 2A)

Same model suite as 04_regression_models.py, trained on features_trend.csv
which adds days_elapsed (fractional days since 2021-01-01) to the original
27 weather features.

All models saved with _trend suffix:
  ridge_trend.pkl, lasso_trend.pkl, decision_tree_trend.pkl,
  random_forest_trend.pkl, gradient_boosting_trend.pkl, svr_trend.pkl,
  mlp_state_trend.pt, mlp_scaler_trend.pkl, regression_trend_results.csv
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
TREND_COLS        = ['days_elapsed']
FEATURE_COLS_TREND = FEATURE_COLS + TREND_COLS
TARGET = 'Load_MW'

def metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"  {label:30s}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return dict(model=label, RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)


print("=== Phase 4d: Regression Models — Days-Elapsed Trend ===\n")

df = pd.read_csv(PROC / "features_trend.csv", parse_dates=["HourEnding"])
df = df.sort_values("HourEnding").reset_index(drop=True)

train = df[df["HourEnding"].dt.year < 2025]
test  = df[df["HourEnding"].dt.year == 2025]

X_train = train[FEATURE_COLS_TREND].values
y_train = train[TARGET].values
X_test  = test[FEATURE_COLS_TREND].values
y_test  = test[TARGET].values

print(f"Training: {len(train):,} rows  |  Features: {len(FEATURE_COLS_TREND)} (27 weather + days_elapsed)")
print(f"Test:     {len(test):,} rows\n")

tscv    = skm.TimeSeriesSplit(n_splits=5)
results = []


# ══════════════════════════════════════════════════════════════════════════════
# 1. OLS
# ══════════════════════════════════════════════════════════════════════════════
print("--- 1. OLS ---")
design  = MS(FEATURE_COLS_TREND).fit(train)
ols_res = sm.OLS(train[TARGET], design.transform(train)).fit()
ols_pred = ols_res.predict(design.transform(test)).values
results.append(metrics(y_test, ols_pred, "OLS"))
print(f"  R² (train): {ols_res.rsquared:.4f}")
trend_coef = ols_res.params.get("days_elapsed", float("nan"))
print(f"  days_elapsed coefficient: {trend_coef:+.2f} MW/day  "
      f"({trend_coef * 365 / 1000:+.2f} GW/year implied)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Ridge
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 2. Ridge ---")
lambdas    = 10 ** np.linspace(6, -3, 100)
ridge_pipe = Pipeline([('scaler', StandardScaler()),
                       ('ridge', skl.ElasticNetCV(alphas=lambdas, l1_ratio=0,
                                                  cv=tscv, max_iter=10000))])
ridge_pipe.fit(X_train, y_train)
print(f"  Best λ: {ridge_pipe.named_steps['ridge'].alpha_:.4f}")
results.append(metrics(y_test, ridge_pipe.predict(X_test), "Ridge"))
joblib.dump(ridge_pipe, MODELS / "ridge_trend.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Lasso
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 3. Lasso ---")
lasso_pipe = Pipeline([('scaler', StandardScaler()),
                       ('lasso', skl.ElasticNetCV(n_alphas=100, l1_ratio=1,
                                                  cv=tscv, max_iter=10000))])
lasso_pipe.fit(X_train, y_train)
lasso_coef = lasso_pipe.named_steps['lasso'].coef_
n_zero     = np.sum(lasso_coef == 0)
trend_kept = lasso_coef[FEATURE_COLS_TREND.index('days_elapsed')] != 0
print(f"  Best λ: {lasso_pipe.named_steps['lasso'].alpha_:.4f}  "
      f"  Zeroed: {n_zero}/{len(FEATURE_COLS_TREND)}  "
      f"  days_elapsed retained: {trend_kept}")
results.append(metrics(y_test, lasso_pipe.predict(X_test), "Lasso"))
joblib.dump(lasso_pipe, MODELS / "lasso_trend.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Decision Tree
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 4. Decision Tree ---")
grid_dtr = skm.GridSearchCV(
    DTR(random_state=0),
    {'max_depth': [3, 5, 8, 12, None], 'min_samples_leaf': [10, 50, 100]},
    cv=tscv, scoring='neg_mean_squared_error', refit=True)
grid_dtr.fit(X_train, y_train)
print(f"  Best params: {grid_dtr.best_params_}")
results.append(metrics(y_test, grid_dtr.predict(X_test), "Decision Tree"))
joblib.dump(grid_dtr.best_estimator_, MODELS / "decision_tree_trend.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Random Forest
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 5. Random Forest ---")
rf = RF(n_estimators=300, max_features='sqrt', min_samples_leaf=5,
        random_state=0, n_jobs=-1)
rf.fit(X_train, y_train)
results.append(metrics(y_test, rf.predict(X_test), "Random Forest"))
imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS_TREND).sort_values(ascending=False)
days_rank = list(imp.index).index('days_elapsed') + 1
print(f"  days_elapsed importance: {imp['days_elapsed']:.4f}  (rank {days_rank}/{len(FEATURE_COLS_TREND)})")
joblib.dump(rf, MODELS / "random_forest_trend.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Gradient Boosting
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 6. Gradient Boosting ---")
gbr = GBR(n_estimators=500, learning_rate=0.05, max_depth=5,
          min_samples_leaf=10, subsample=0.8, random_state=0)
gbr.fit(X_train, y_train)
results.append(metrics(y_test, gbr.predict(X_test), "Gradient Boosting"))
imp_gbr = pd.Series(gbr.feature_importances_, index=FEATURE_COLS_TREND)
days_rank_gbr = list(imp_gbr.sort_values(ascending=False).index).index('days_elapsed') + 1
print(f"  days_elapsed importance: {imp_gbr['days_elapsed']:.4f}  (rank {days_rank_gbr}/{len(FEATURE_COLS_TREND)})")
joblib.dump(gbr, MODELS / "gradient_boosting_trend.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SVR
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 7. SVR ---")
svr_pipe = Pipeline([('scaler', StandardScaler()),
                     ('svr', SVR(kernel='rbf', C=1e4, epsilon=500, gamma='scale'))])
svr_pipe.fit(X_train, y_train)
results.append(metrics(y_test, svr_pipe.predict(X_test), "SVR"))
joblib.dump(svr_pipe, MODELS / "svr_trend.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MLP (PyTorch) — 28-feature input
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 8. MLP (PyTorch) ---")

mlp_scaler = StandardScaler()
X_train_np = mlp_scaler.fit_transform(X_train)
X_test_np  = mlp_scaler.transform(X_test)
X_train_t  = torch.tensor(X_train_np.astype(np.float32))
y_train_t  = torch.tensor(y_train.astype(np.float32))
X_test_t   = torch.tensor(X_test_np.astype(np.float32))
y_test_t   = torch.tensor(y_test.astype(np.float32))

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)
mlp_dm   = SimpleDataModule(train_ds, test_ds,
                             batch_size=1024,
                             num_workers=min(4, rec_num_workers()),
                             validation=test_ds)


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


mlp_model  = ERCOTModel(X_train_t.shape[1])
mlp_module = SimpleModule.regression(
    mlp_model,
    optimizer=Adam(mlp_model.parameters(), lr=1e-3, weight_decay=1e-5),
    metrics={'mae': MeanAbsoluteError(), 'r2': R2Score()})
mlp_trainer = Trainer(
    max_epochs=200, accelerator='gpu', devices=1,
    precision='bf16-mixed', gradient_clip_val=1.0,
    log_every_n_steps=20, logger=CSVLogger('logs', name='ERCOT_MLP_trend'),
    callbacks=[ErrorTracker()], enable_progress_bar=True)
mlp_trainer.fit(mlp_module, datamodule=mlp_dm)
mlp_trainer.test(mlp_module, datamodule=mlp_dm)

mlp_model.eval()
with torch.no_grad():
    mlp_pred = mlp_module(X_test_t).numpy()
results.append(metrics(y_test, mlp_pred, "MLP (PyTorch)"))

torch.save(mlp_model.state_dict(), MODELS / "mlp_state_trend.pt")
joblib.dump(mlp_scaler, MODELS / "mlp_scaler_trend.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Trend Model Results (Test Set: 2025) ===")
res_df = pd.DataFrame(results).set_index('model').sort_values('RMSE')
print(res_df.round(2).to_string())
res_df.to_csv(PROC / "regression_trend_results.csv")
print("\nPhase 4d complete.")
