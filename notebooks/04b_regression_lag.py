"""
Phase 4b: Regression Models — Lag Features

Same model suite as 04_regression_models.py, trained on features_lag.csv
which adds load_lag_24h and load_lag_168h to the original 27 weather features.

All models saved with _lag suffix to coexist with the original set:
  ridge_lag.pkl, lasso_lag.pkl, decision_tree_lag.pkl,
  random_forest_lag.pkl, gradient_boosting_lag.pkl, svr_lag.pkl,
  mlp_state_lag.pt, regression_lag_results.csv

No figures generated — run 07_approach_comparison.py for side-by-side plots.
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
LAG_COLS     = ['load_lag_24h', 'load_lag_168h']
FEATURE_COLS_LAG = FEATURE_COLS + LAG_COLS
TARGET = 'Load_MW'

def metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"  {label:30s}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return dict(model=label, RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)


print("=== Phase 4b: Regression Models — Lag Features ===\n")

df = pd.read_csv(PROC / "features_lag.csv", parse_dates=["HourEnding"])
df = df.sort_values("HourEnding").reset_index(drop=True)

train = df[df["HourEnding"].dt.year < 2025]
test  = df[df["HourEnding"].dt.year == 2025]

X_train = train[FEATURE_COLS_LAG].values
y_train = train[TARGET].values
X_test  = test[FEATURE_COLS_LAG].values
y_test  = test[TARGET].values

print(f"Training: {len(train):,} rows  |  Features: {len(FEATURE_COLS_LAG)}")
print(f"Test:     {len(test):,} rows\n")

tscv = skm.TimeSeriesSplit(n_splits=5)
results_list = []
scaler = StandardScaler()


# ══════════════════════════════════════════════════════════════════════════════
# 1. OLS  (re-fit on-the-fly in 07; saved here for offline reference only)
# ══════════════════════════════════════════════════════════════════════════════
print("--- 1. OLS ---")
design  = MS(FEATURE_COLS_LAG).fit(train)
ols_res = sm.OLS(train[TARGET], design.transform(train)).fit()
ols_pred = ols_res.predict(design.transform(test)).values
results_list.append(metrics(y_test, ols_pred, "OLS"))
print(f"  R² (train): {ols_res.rsquared:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Ridge
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 2. Ridge ---")
lambdas    = 10 ** np.linspace(6, -3, 100)
ridgeCV    = skl.ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=tscv, max_iter=10000)
ridge_pipe = Pipeline([('scaler', StandardScaler()), ('ridge', ridgeCV)])
ridge_pipe.fit(X_train, y_train)
print(f"  Best λ: {ridge_pipe.named_steps['ridge'].alpha_:.4f}")
results_list.append(metrics(y_test, ridge_pipe.predict(X_test), "Ridge"))
joblib.dump(ridge_pipe, MODELS / "ridge_lag.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Lasso
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 3. Lasso ---")
lassoCV    = skl.ElasticNetCV(n_alphas=100, l1_ratio=1, cv=tscv, max_iter=10000)
lasso_pipe = Pipeline([('scaler', StandardScaler()), ('lasso', lassoCV)])
lasso_pipe.fit(X_train, y_train)
print(f"  Best λ: {lasso_pipe.named_steps['lasso'].alpha_:.4f}")
n_zero = np.sum(lasso_pipe.named_steps['lasso'].coef_ == 0)
print(f"  Zeroed: {n_zero}/{len(FEATURE_COLS_LAG)}")
results_list.append(metrics(y_test, lasso_pipe.predict(X_test), "Lasso"))
joblib.dump(lasso_pipe, MODELS / "lasso_lag.pkl")


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
results_list.append(metrics(y_test, grid_dtr.predict(X_test), "Decision Tree"))
joblib.dump(grid_dtr.best_estimator_, MODELS / "decision_tree_lag.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Random Forest
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 5. Random Forest ---")
rf = RF(n_estimators=300, max_features='sqrt', min_samples_leaf=5,
        random_state=0, n_jobs=-1)
rf.fit(X_train, y_train)
results_list.append(metrics(y_test, rf.predict(X_test), "Random Forest"))
print("\n  Top 5 feature importances:")
imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS_LAG).sort_values(ascending=False)
print(imp.head(5).to_string())
joblib.dump(rf, MODELS / "random_forest_lag.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Gradient Boosting
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 6. Gradient Boosting ---")
gbr = GBR(n_estimators=500, learning_rate=0.05, max_depth=5,
          min_samples_leaf=10, subsample=0.8, random_state=0)
gbr.fit(X_train, y_train)
results_list.append(metrics(y_test, gbr.predict(X_test), "Gradient Boosting"))
joblib.dump(gbr, MODELS / "gradient_boosting_lag.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SVR
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 7. SVR ---")
svr_pipe = Pipeline([('scaler', StandardScaler()),
                     ('svr', SVR(kernel='rbf', C=1e4, epsilon=500, gamma='scale'))])
svr_pipe.fit(X_train, y_train)
results_list.append(metrics(y_test, svr_pipe.predict(X_test), "SVR"))
joblib.dump(svr_pipe, MODELS / "svr_lag.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MLP (PyTorch) — same architecture, 29-feature input
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 8. MLP (PyTorch) ---")

mlp_scaler  = StandardScaler()
X_train_np  = mlp_scaler.fit_transform(X_train)
X_test_np   = mlp_scaler.transform(X_test)
X_train_t   = torch.tensor(X_train_np.astype(np.float32))
y_train_t   = torch.tensor(y_train.astype(np.float32))
X_test_t    = torch.tensor(X_test_np.astype(np.float32))
y_test_t    = torch.tensor(y_test.astype(np.float32))

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

mlp_dm = SimpleDataModule(train_ds, test_ds,
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
    log_every_n_steps=20, logger=CSVLogger('logs', name='ERCOT_MLP_lag'),
    callbacks=[ErrorTracker()], enable_progress_bar=True)
mlp_trainer.fit(mlp_module, datamodule=mlp_dm)
mlp_trainer.test(mlp_module, datamodule=mlp_dm)

mlp_model.eval()
with torch.no_grad():
    mlp_pred = mlp_module(X_test_t).numpy()
results_list.append(metrics(y_test, mlp_pred, "MLP (PyTorch)"))

torch.save(mlp_model.state_dict(), MODELS / "mlp_state_lag.pt")
joblib.dump(mlp_scaler, MODELS / "mlp_scaler_lag.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Lag Model Results (Test Set: 2025) ===")
res_df = pd.DataFrame(results_list).set_index('model').sort_values('RMSE')
print(res_df.round(2).to_string())
res_df.to_csv(PROC / "regression_lag_results.csv")
print("\nPhase 4b complete.")
