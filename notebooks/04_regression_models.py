"""
Phase 4: Regression Models
Predicts hourly ERCOT load (MW) using all course-covered methods.

Train/test split strategy: temporal — train 2021–2024, test on 2025.
This is the correct approach for time series: no future information leaks
into training. Random KFold would be inappropriate here.

Cross-validation: TimeSeriesSplit (5 folds) — preserves temporal order
within the training set.

Models (ISLP lab style):
  1. OLS Linear Regression       (statsmodels + ISLP MS)
  2. Ridge Regression            (skl.ElasticNetCV, l1_ratio=0)
  3. Lasso Regression            (skl.ElasticNetCV, l1_ratio=1)
  4. Decision Tree Regressor     (sklearn DTR + GridSearchCV)
  5. Random Forest Regressor     (sklearn RF)
  6. Gradient Boosting Regressor (sklearn GBR)
  7. Support Vector Regression   (sklearn SVR + Pipeline)
  8. MLP Neural Network          (PyTorch nn.Module + ISLP SimpleDataModule, Ch10 style)
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
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
from ISLP.models import ModelSpec as MS, summarize
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset
from torchmetrics import MeanAbsoluteError, R2Score
from ISLP.torch import SimpleDataModule, SimpleModule, ErrorTracker, rec_num_workers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

torch.set_float32_matmul_precision('high')  # tensor core utilization on RTX 5070 Ti
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

ROOT    = Path(__file__).resolve().parent.parent
PROC    = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MODELS  = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# ── Feature columns (no lags — compatible with live prediction) ───────────────
FEATURE_COLS = [
    # Per-city temperatures (captures regional variation)
    'temperature_2m_Austin', 'temperature_2m_Dallas',
    'temperature_2m_Houston', 'temperature_2m_SanAntonio',
    # System-wide weather averages
    'temperature_2m_avg', 'apparent_temperature_avg',
    'relative_humidity_2m_avg', 'dew_point_2m_avg',
    'wind_speed_10m_avg', 'shortwave_radiation_avg',
    # Engineered weather
    'CDH', 'HDH', 'temp_sq', 'apparent_temp_delta_avg',
    # Cyclic time encodings
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    # Calendar flags
    'is_weekend', 'is_holiday', 'is_workday', 'season',
    # Interactions
    'temp_x_hour_sin', 'CDH_x_hour_sin', 'temp_x_is_weekend',
]
TARGET = 'Load_MW'

# ── Helpers ───────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    if label:
        print(f"  {label:30s}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return dict(model=label, RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)

def savefig(name):
    plt.savefig(FIGURES / name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {name}")

def residual_plot(ax, y_true, y_pred, title):
    resid = y_true - y_pred
    ax.scatter(y_pred / 1000, resid / 1000, alpha=0.15, s=4, rasterized=True)
    ax.axhline(0, c='k', ls='--', lw=1)
    ax.set_xlabel('Fitted value (GW)')
    ax.set_ylabel('Residual (GW)')
    ax.set_title(title)


# ══════════════════════════════════════════════════════════════════════════════
# Load data & temporal train/test split
# ══════════════════════════════════════════════════════════════════════════════
print("=== Phase 4: Regression Models ===\n")

df = pd.read_csv(PROC / "features_nolag.csv", parse_dates=["HourEnding"])
df = df.sort_values("HourEnding").reset_index(drop=True)

train = df[df["HourEnding"].dt.year < 2025]
test  = df[df["HourEnding"].dt.year == 2025]

X_train = train[FEATURE_COLS].values
y_train = train[TARGET].values
X_test  = test[FEATURE_COLS].values
y_test  = test[TARGET].values

print(f"Training: {len(train):,} rows ({train.HourEnding.min().date()} → {train.HourEnding.max().date()})")
print(f"Test:     {len(test):,} rows  ({test.HourEnding.min().date()} → {test.HourEnding.max().date()})")
print(f"Features: {len(FEATURE_COLS)}\n")

# TimeSeriesSplit: preserves temporal order within training fold
tscv = skm.TimeSeriesSplit(n_splits=5)

results_list = []


# ══════════════════════════════════════════════════════════════════════════════
# 1. OLS Linear Regression  (statsmodels + ISLP ModelSpec)
# ══════════════════════════════════════════════════════════════════════════════
print("--- 1. OLS Linear Regression ---")

design = MS(FEATURE_COLS).fit(train)
X_sm   = design.transform(train)           # adds intercept column
y_sm   = train[TARGET]

ols_model   = sm.OLS(y_sm, X_sm)
ols_results = ols_model.fit()

print(f"  R² (train): {ols_results.rsquared:.4f}")
print(f"  Adj R²:     {ols_results.rsquared_adj:.4f}")

# Evaluate on test set
X_sm_test = design.transform(test)
ols_pred  = ols_results.predict(X_sm_test)
results_list.append(metrics(y_test, ols_pred, "OLS Linear"))

# Coefficient table (top 10 by |coef|)
coef_df = summarize(ols_results)
print("\n  Top 10 coefficients by magnitude:")
print(coef_df.reindex(coef_df['coef'].abs().sort_values(ascending=False).index).head(10).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 2. Ridge Regression  (ElasticNetCV, l1_ratio=0, Pipeline)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 2. Ridge Regression ---")

lambdas    = 10 ** np.linspace(6, -3, 100)
scaler     = StandardScaler(with_mean=True, with_std=True)
ridgeCV    = skl.ElasticNetCV(alphas=lambdas, l1_ratio=0, cv=tscv, max_iter=10000)
ridge_pipe = Pipeline([('scaler', scaler), ('ridge', ridgeCV)])
ridge_pipe.fit(X_train, y_train)

tuned_ridge = ridge_pipe.named_steps['ridge']
print(f"  Best λ (alpha): {tuned_ridge.alpha_:.4f}")

ridge_pred = ridge_pipe.predict(X_test)
results_list.append(metrics(y_test, ridge_pred, "Ridge"))

# Ridge coefficient path plot
lambdas_path, coef_path = skl.ElasticNet.path(
    scaler.transform(X_train), y_train, l1_ratio=0, alphas=lambdas)[:2]
fig, ax = subplots(figsize=(9, 5))
ax.plot(-np.log(lambdas_path), coef_path.T, lw=0.8, alpha=0.7)
ax.axvline(-np.log(tuned_ridge.alpha_), c='k', ls='--', label=f'CV best λ')
ax.set_xlabel(r'$-\log(\lambda)$', fontsize=13)
ax.set_ylabel('Standardized coefficients', fontsize=13)
ax.set_title('Ridge Regression — Coefficient Path')
ax.legend()
savefig("10_ridge_path.png")

joblib.dump(ridge_pipe, MODELS / "ridge.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Lasso Regression  (ElasticNetCV, l1_ratio=1, Pipeline)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 3. Lasso Regression ---")

lassoCV    = skl.ElasticNetCV(n_alphas=100, l1_ratio=1, cv=tscv, max_iter=10000)
lasso_pipe = Pipeline([('scaler', scaler), ('lasso', lassoCV)])
lasso_pipe.fit(X_train, y_train)

tuned_lasso = lasso_pipe.named_steps['lasso']
print(f"  Best λ (alpha): {tuned_lasso.alpha_:.4f}")

lasso_coefs = tuned_lasso.coef_
n_zero = np.sum(lasso_coefs == 0)
print(f"  Coefficients zeroed out: {n_zero} of {len(lasso_coefs)}")
print("  Non-zero features:")
for feat, coef in zip(FEATURE_COLS, lasso_coefs):
    if coef != 0:
        print(f"    {feat:40s} {coef:+.2f}")

lasso_pred = lasso_pipe.predict(X_test)
results_list.append(metrics(y_test, lasso_pred, "Lasso"))

# Lasso CV MSE plot
fig, ax = subplots(figsize=(9, 5))
ax.errorbar(-np.log(tuned_lasso.alphas_),
            tuned_lasso.mse_path_.mean(1),
            yerr=tuned_lasso.mse_path_.std(1) / np.sqrt(5),
            lw=1.2, elinewidth=0.8)
ax.axvline(-np.log(tuned_lasso.alpha_), c='k', ls='--', label='CV best λ')
ax.set_xlabel(r'$-\log(\lambda)$', fontsize=13)
ax.set_ylabel('Cross-validated MSE', fontsize=13)
ax.set_title('Lasso — Cross-Validated MSE vs Regularization')
ax.legend()
savefig("11_lasso_cv.png")

joblib.dump(lasso_pipe, MODELS / "lasso.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Decision Tree Regressor
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 4. Decision Tree Regressor ---")

dtr = DTR(random_state=0)
ccp_path = dtr.cost_complexity_pruning_path(X_train, y_train)
grid_dtr = skm.GridSearchCV(
    DTR(random_state=0),
    {'max_depth': [3, 5, 8, 12, None],
     'min_samples_leaf': [10, 50, 100]},
    cv=tscv,
    scoring='neg_mean_squared_error',
    refit=True
)
grid_dtr.fit(X_train, y_train)
best_dtr = grid_dtr.best_estimator_
print(f"  Best params: {grid_dtr.best_params_}")

dtr_pred = best_dtr.predict(X_test)
results_list.append(metrics(y_test, dtr_pred, "Decision Tree"))

joblib.dump(best_dtr, MODELS / "decision_tree.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Random Forest
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 5. Random Forest ---")

rf = RF(n_estimators=300,
        max_features='sqrt',
        min_samples_leaf=5,
        random_state=0,
        n_jobs=-1)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
results_list.append(metrics(y_test, rf_pred, "Random Forest"))

# Feature importance
feat_imp = pd.DataFrame({'importance': rf.feature_importances_},
                        index=FEATURE_COLS).sort_values('importance', ascending=False)
print("\n  Top 10 feature importances:")
print(feat_imp.head(10).to_string())

fig, ax = subplots(figsize=(8, 7))
feat_imp.head(15).sort_values('importance').plot.barh(ax=ax, legend=False, color='steelblue')
ax.set_xlabel('Feature Importance (mean decrease in impurity)')
ax.set_title('Random Forest — Top 15 Feature Importances')
plt.tight_layout()
savefig("12_rf_feature_importance.png")

joblib.dump(rf, MODELS / "random_forest.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Gradient Boosting
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 6. Gradient Boosting ---")

gbr = GBR(n_estimators=500,
          learning_rate=0.05,
          max_depth=5,
          min_samples_leaf=10,
          subsample=0.8,
          random_state=0)
gbr.fit(X_train, y_train)

# Track test error vs n_estimators
test_errors = np.array([mean_squared_error(y_test, y_pred)
                        for y_pred in gbr.staged_predict(X_test)])
best_n = np.argmin(test_errors) + 1
print(f"  Best n_estimators (by test MSE): {best_n}")

gbr_pred = gbr.predict(X_test)
results_list.append(metrics(y_test, gbr_pred, "Gradient Boosting"))

# Training vs test error plot
fig, ax = subplots(figsize=(9, 5))
ax.plot(gbr.train_score_, color='steelblue', lw=1, label='Train (neg MSE)')
ax.plot(-test_errors, color='tomato', lw=1, label='Test (neg MSE)')
ax.axvline(best_n - 1, c='k', ls='--', lw=1, label=f'Best @ {best_n}')
ax.set_xlabel('Number of Trees')
ax.set_ylabel('Neg MSE')
ax.set_title('Gradient Boosting — Train vs Test Error')
ax.legend()
savefig("13_gbr_train_test.png")

# GBR feature importance
gbr_imp = pd.DataFrame({'importance': gbr.feature_importances_},
                       index=FEATURE_COLS).sort_values('importance', ascending=False)
joblib.dump(gbr, MODELS / "gradient_boosting.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Support Vector Regression
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 7. Support Vector Regression ---")

svr      = SVR(kernel='rbf', C=1e4, epsilon=500, gamma='scale')
svr_pipe = Pipeline([('scaler', scaler), ('svr', svr)])
svr_pipe.fit(X_train, y_train)

svr_pred = svr_pipe.predict(X_test)
results_list.append(metrics(y_test, svr_pred, "SVR (RBF kernel)"))

joblib.dump(svr_pipe, MODELS / "svr.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MLP Neural Network  (PyTorch / ISLP Ch10 style)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 8. MLP Neural Network (PyTorch) ---")

# Scale features using the same scaler as other models
X_train_np = scaler.fit_transform(X_train)
X_test_np  = scaler.transform(X_test)

# Convert to float32 tensors (torch requirement)
X_train_t = torch.tensor(X_train_np.astype(np.float32))
y_train_t = torch.tensor(np.array(y_train).astype(np.float32))
X_test_t  = torch.tensor(X_test_np.astype(np.float32))
y_test_t  = torch.tensor(np.array(y_test).astype(np.float32))

hit_train_ds = TensorDataset(X_train_t, y_train_t)
hit_test_ds  = TensorDataset(X_test_t,  y_test_t)

max_num_workers = rec_num_workers()

mlp_dm = SimpleDataModule(hit_train_ds,
                          hit_test_ds,
                          batch_size=1024,          # large batch → full GPU utilization
                          num_workers=min(4, max_num_workers),
                          validation=hit_test_ds)


class ERCOTModel(nn.Module):
    def __init__(self, input_size):
        super(ERCOTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1))

    def forward(self, x):
        x = self.flatten(x)
        return torch.flatten(self.sequential(x))


mlp_model     = ERCOTModel(X_train_t.shape[1])
mlp_optimizer = Adam(mlp_model.parameters(), lr=1e-3, weight_decay=1e-5)
mlp_module    = SimpleModule.regression(mlp_model,
                                        optimizer=mlp_optimizer,
                                        metrics={'mae': MeanAbsoluteError(),
                                                 'r2':  R2Score()})
mlp_logger  = CSVLogger('logs', name='ERCOT_MLP')
mlp_trainer = Trainer(max_epochs=200,
                      accelerator='gpu',
                      devices=1,
                      precision='bf16-mixed',        # bf16 keeps fp32 range → no overflow on MW-scale targets
                      gradient_clip_val=1.0,
                      log_every_n_steps=20,
                      logger=mlp_logger,
                      callbacks=[ErrorTracker()],
                      enable_progress_bar=True)
mlp_trainer.fit(mlp_module, datamodule=mlp_dm)
mlp_trainer.test(mlp_module, datamodule=mlp_dm)

# Extract predictions — follow Ch10 style: model.eval() then module(X_test_t)
mlp_model.eval()
with torch.no_grad():
    mlp_pred = mlp_module(X_test_t).numpy()

results_list.append(metrics(y_test, mlp_pred, "MLP Neural Network"))

torch.save(mlp_model.state_dict(), MODELS / "mlp_state.pt")


# ══════════════════════════════════════════════════════════════════════════════
# Results comparison table
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Model Comparison (Test Set: 2025) ===")
results_df = pd.DataFrame(results_list).set_index('model').sort_values('RMSE')
print(results_df.round(2).to_string())
results_df.to_csv(PROC / "regression_results.csv")

# Bar chart comparison
fig, axes = subplots(1, 2, figsize=(13, 5))
results_df['RMSE'].sort_values().plot.barh(ax=axes[0], color='steelblue', alpha=0.85)
axes[0].set_xlabel('RMSE (MW)')
axes[0].set_title('Model Comparison — RMSE (lower is better)')

results_df['R2'].sort_values().plot.barh(ax=axes[1], color='darkorange', alpha=0.85)
axes[1].set_xlabel('R²')
axes[1].set_title('Model Comparison — R² (higher is better)')
plt.tight_layout()
savefig("14_model_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# Residual plots for best 4 models
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = subplots(2, 2, figsize=(13, 10))
pairs = [
    ("OLS Linear",       ols_pred),
    ("Random Forest",    rf_pred),
    ("Gradient Boosting",gbr_pred),
    ("MLP Neural Network",mlp_pred),
]
for ax, (label, pred) in zip(axes.flat, pairs):
    residual_plot(ax, y_test, pred, label)
plt.suptitle('Residual Plots — Test Set (2025)', fontsize=14, y=1.01)
plt.tight_layout()
savefig("15_residual_plots.png")


# ══════════════════════════════════════════════════════════════════════════════
# Actual vs predicted: best model on a sample week
# ══════════════════════════════════════════════════════════════════════════════
best_model_name = results_df['RMSE'].idxmin()
best_pred_map   = {
    "OLS Linear":        ols_pred,
    "Ridge":             ridge_pred,
    "Lasso":             lasso_pred,
    "Decision Tree":     dtr_pred,
    "Random Forest":     rf_pred,
    "Gradient Boosting": gbr_pred,
    "SVR (RBF kernel)":  svr_pred,
    "MLP Neural Network":mlp_pred,
}
best_pred = best_pred_map[best_model_name]

# Pick a representative summer week (July 2025)
week_mask = (test["HourEnding"] >= "2025-07-14") & (test["HourEnding"] < "2025-07-21")
week_ts   = test.loc[week_mask, "HourEnding"]
week_true = y_test[week_mask.values]
week_pred = best_pred[week_mask.values]

fig, ax = subplots(figsize=(12, 4))
ax.plot(week_ts, week_true / 1000, label='Actual', color='#003087', lw=2)
ax.plot(week_ts, week_pred / 1000, label=f'{best_model_name} (predicted)',
        color='tomato', lw=1.8, ls='--')
ax.set_ylabel('Load (GW)')
ax.set_title(f'Actual vs Predicted — Week of Jul 14–20, 2025  [{best_model_name}]')
ax.legend()
savefig("16_best_model_week.png")

print(f"\nBest model: {best_model_name}")
print("Phase 4 complete. Run 05_classification_models.py next.")
