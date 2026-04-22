"""
Phase 4c: Bias Offset Computation  (Approach 2B)

Computes a per-model bias correction offset from the 2024 training data —
the most recent complete year in the training set, closest to present conditions.

Methodology:
  offset[model] = mean( actual_2024 - predicted_2024 )

At inference time: corrected_prediction = raw_prediction + offset[model]

This is a post-hoc calibration step that requires no model retraining.
The 2024 calibration year was chosen because:
  - It is in the training set (no data leakage)
  - It is the most recent complete year, best reflecting current load growth

Output: models/bias_offsets.json
  Keys match model labels used in 07_approach_comparison.py.

Approaches 2A (year trend feature) and 2C (retrain with 2025 data) slot into
the same 07_approach_comparison.py framework without touching this file.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib
import json
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from ISLP.models import ModelSpec as MS
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

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
TARGET = 'Load_MW'

print("=== Phase 4c: Bias Offset Computation ===\n")

df = pd.read_csv(PROC / "features_nolag.csv", parse_dates=["HourEnding"])
df = df.sort_values("HourEnding").reset_index(drop=True)

train   = df[df["HourEnding"].dt.year < 2025]
calib   = df[df["HourEnding"].dt.year == 2024]   # calibration: most recent training year

X_calib = calib[FEATURE_COLS].values
y_calib = calib[TARGET].values

print(f"Calibration set (2024): {len(calib):,} rows")
print(f"Mean actual load:       {y_calib.mean():,.0f} MW\n")


class ERCOTModel(nn.Module):
    """Must match architecture in 04_regression_models.py exactly."""
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


offsets = {}

# 1. OLS — refit from training data (statsmodels, fast)
print("Computing OLS offset ...")
design   = MS(FEATURE_COLS).fit(train)
ols_res  = sm.OLS(train[TARGET], design.transform(train)).fit()
ols_pred = ols_res.predict(design.transform(calib)).values
offsets["OLS"] = float(np.mean(y_calib - ols_pred))

# 2–7. sklearn pipelines
sklearn_models = {
    "Ridge":          MODELS / "ridge.pkl",
    "Lasso":          MODELS / "lasso.pkl",
    "Decision Tree":  MODELS / "decision_tree.pkl",
    "Random Forest":  MODELS / "random_forest.pkl",
    "Gradient Boost": MODELS / "gradient_boosting.pkl",
    "SVR":            MODELS / "svr.pkl",
}
for label, path in sklearn_models.items():
    model = joblib.load(path)
    pred  = model.predict(X_calib)
    offsets[label] = float(np.mean(y_calib - pred))
    print(f"  {label:20s}  offset = {offsets[label]:+,.0f} MW")

# 8. MLP (PyTorch)
print("Computing MLP offset ...")
mlp_scaler = StandardScaler().fit(train[FEATURE_COLS].values)
X_calib_t  = torch.tensor(mlp_scaler.transform(X_calib).astype(np.float32))
mlp_model  = ERCOTModel(len(FEATURE_COLS))
mlp_model.load_state_dict(torch.load(MODELS / "mlp_state.pt", map_location="cpu"))
mlp_model.eval()
with torch.no_grad():
    mlp_pred = mlp_model(X_calib_t).numpy()
offsets["MLP (PyTorch)"] = float(np.mean(y_calib - mlp_pred))

print(f"\n{'Model':<22}  {'2024 Offset (MW)':>18}")
print(f"  {'-'*42}")
for k, v in offsets.items():
    print(f"  {k:<20}  {v:>+16,.0f}")

out = MODELS / "bias_offsets.json"
with open(out, "w") as f:
    json.dump(offsets, f, indent=2)
print(f"\nSaved → {out}")
print("Phase 4c complete.")
