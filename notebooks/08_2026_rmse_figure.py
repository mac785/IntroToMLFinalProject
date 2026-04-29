"""
Phase 8: 2026 RMSE Comparison Figure

Evaluates ALL approach variants on the Jan–Apr 2026 test set and produces
a bar chart analogous to Figure 24 (live RMSE) but using the full offline
2026 test period (2,663 hours instead of 16).

Approaches:
  1. Current (no lag)            — original models, test on features_nolag_2026
  2. Bias-corrected              — original models + bias offsets
  3. Lag features                — _lag models, test on features_lag_2026
  4. Days-elapsed trend (2A)     — _trend models, test on features_trend_2026
  5. 2025 retrain - no lag (2C)  — _2025train models, test on features_nolag_2026
  6. 2025 retrain + lag (2C)     — _2025train_lag models, test on features_lag_2026
  7. 2025 retrain + trend (2C)   — _2025train_trend models, test on features_trend_2026

Figure saved: figures/26_2026_test_rmse_comparison.png
"""

import numpy as np
import pandas as pd
import joblib
import json
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from ISLP.models import ModelSpec as MS
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

ROOT    = Path(__file__).resolve().parent.parent
PROC    = ROOT / "data" / "processed"
MODELS  = ROOT / "models"
FIGURES = ROOT / "figures"

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

MODEL_NAMES = ["OLS", "Ridge", "Lasso", "Decision Tree",
               "Random Forest", "Gradient Boost", "SVR", "MLP"]

MODEL_FILES = {
    "Ridge":          "ridge{s}.pkl",
    "Lasso":          "lasso{s}.pkl",
    "Decision Tree":  "decision_tree{s}.pkl",
    "Random Forest":  "random_forest{s}.pkl",
    "Gradient Boost": "gradient_boosting{s}.pkl",
    "SVR":            "svr{s}.pkl",
}


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


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_variant(suffix, feat_cols, train_df, test_df, bias_offsets=None):
    """Predict with all 8 models for one variant; return {model: RMSE}."""
    y_test  = test_df[TARGET].values
    X_test  = test_df[feat_cols].values
    results = {}

    # OLS (refit on train_df each time — no pkl saved for OLS)
    try:
        design  = MS(feat_cols).fit(train_df)
        ols_res = sm.OLS(train_df[TARGET], design.transform(train_df)).fit()
        ols_df  = test_df[feat_cols].copy()
        ols_pred = ols_res.predict(design.transform(ols_df)).values
        if bias_offsets:
            ols_pred += bias_offsets.get("OLS", 0.0)
        results["OLS"] = rmse(y_test, ols_pred)
    except Exception as e:
        print(f"    OLS failed: {e}")

    # sklearn models
    for name in ["Ridge", "Lasso", "Decision Tree",
                 "Random Forest", "Gradient Boost", "SVR"]:
        fpath = MODELS / MODEL_FILES[name].format(s=suffix)
        if not fpath.exists():
            continue
        pred = joblib.load(fpath).predict(X_test)
        if bias_offsets:
            pred += bias_offsets.get(name, 0.0)
        results[name] = rmse(y_test, pred)

    # MLP
    scaler_path = MODELS / f"mlp_scaler{suffix}.pkl"
    state_path  = MODELS / f"mlp_state{suffix}.pt"
    if scaler_path.exists() and state_path.exists():
        sc    = joblib.load(scaler_path)
        X_sc  = torch.tensor(sc.transform(X_test).astype(np.float32))
        model = ERCOTModel(X_test.shape[1])
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            pred = model(X_sc).numpy()
        if bias_offsets:
            pred += bias_offsets.get("MLP", 0.0)
        results["MLP"] = rmse(y_test, pred)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data ...")

test_nolag     = pd.read_csv(PROC / "features_nolag_2026.csv",     parse_dates=["HourEnding"])
test_lag       = pd.read_csv(PROC / "features_lag_2026.csv",       parse_dates=["HourEnding"])
test_trend     = pd.read_csv(PROC / "features_trend_2026.csv",     parse_dates=["HourEnding"])
test_lag_trend = pd.read_csv(PROC / "features_lag_trend_2026.csv", parse_dates=["HourEnding"])

# Training data: 2021-2024 (original split)
train_nolag_orig  = pd.read_csv(PROC / "features_nolag.csv",  parse_dates=["HourEnding"])
train_nolag_orig  = train_nolag_orig[train_nolag_orig["HourEnding"].dt.year < 2025]

train_lag_orig    = pd.read_csv(PROC / "features_lag.csv",    parse_dates=["HourEnding"])
train_lag_orig    = train_lag_orig[train_lag_orig["HourEnding"].dt.year < 2025]

train_trend_orig  = pd.read_csv(PROC / "features_trend.csv",  parse_dates=["HourEnding"])
train_trend_orig  = train_trend_orig[train_trend_orig["HourEnding"].dt.year < 2025]

# Training data: 2021-2025 (2C full)
train_nolag_full     = pd.read_csv(PROC / "features_nolag.csv",     parse_dates=["HourEnding"])
train_lag_full       = pd.read_csv(PROC / "features_lag.csv",       parse_dates=["HourEnding"])
train_trend_full     = pd.read_csv(PROC / "features_trend.csv",     parse_dates=["HourEnding"])
train_lag_trend_full = pd.read_csv(PROC / "features_lag_trend.csv", parse_dates=["HourEnding"])

# Bias offsets
bias_offsets = {}
bp = MODELS / "bias_offsets.json"
if bp.exists():
    with open(bp) as f:
        bias_offsets = json.load(f)

print(f"  Test set: {len(test_nolag)} hours  "
      f"({test_nolag.HourEnding.min().date()} → {test_nolag.HourEnding.max().date()})")

# ══════════════════════════════════════════════════════════════════════════════
# Evaluate each approach
# ══════════════════════════════════════════════════════════════════════════════
APPROACHES = [
    {
        "label":   "Current (weather-only)",
        "suffix":  "",
        "feat_cols": FEATURE_COLS,
        "train_df":  train_nolag_orig,
        "test_df":   test_nolag,
        "bias":      None,
        "color":     "#4C72B0",
    },
    {
        "label":   "Bias-corrected",
        "suffix":  "",
        "feat_cols": FEATURE_COLS,
        "train_df":  train_nolag_orig,
        "test_df":   test_nolag,
        "bias":      bias_offsets,
        "color":     "#DD8452",
    },
    {
        "label":   "Lag features",
        "suffix":  "_lag",
        "feat_cols": FEATURE_COLS + LAG_COLS,
        "train_df":  train_lag_orig,
        "test_df":   test_lag,
        "bias":      None,
        "color":     "#55A868",
    },
    {
        "label":   "Days-elapsed trend",
        "suffix":  "_trend",
        "feat_cols": FEATURE_COLS + TREND_COLS,
        "train_df":  train_trend_orig,
        "test_df":   test_trend,
        "bias":      None,
        "color":     "#C44E52",
    },
    {
        "label":   "2025 retrain",
        "suffix":  "_2025train",
        "feat_cols": FEATURE_COLS,
        "train_df":  train_nolag_full,
        "test_df":   test_nolag,
        "bias":      None,
        "color":     "#8172B2",
    },
    {
        "label":   "2025 retrain\n+ lag",
        "suffix":  "_2025train_lag",
        "feat_cols": FEATURE_COLS + LAG_COLS,
        "train_df":  train_lag_full,
        "test_df":   test_lag,
        "bias":      None,
        "color":     "#937860",
    },
    {
        "label":   "2025 retrain\n+ trend",
        "suffix":  "_2025train_trend",
        "feat_cols": FEATURE_COLS + TREND_COLS,
        "train_df":  train_trend_full,
        "test_df":   test_trend,
        "bias":      None,
        "color":     "#E377C2",
    },
    {
        "label":   "2025 retrain\n+ lag + trend",
        "suffix":  "_2025train_lag_trend",
        "feat_cols": FEATURE_COLS + LAG_COLS + TREND_COLS,
        "train_df":  train_lag_trend_full,
        "test_df":   test_lag_trend,
        "bias":      None,
        "color":     "#2CA02C",
    },
]

all_rmse = {}   # approach_label → {model: RMSE in GW}
for ap in APPROACHES:
    print(f"  Evaluating: {ap['label'].replace(chr(10), ' ')} ...")
    res = evaluate_variant(
        suffix=ap["suffix"],
        feat_cols=ap["feat_cols"],
        train_df=ap["train_df"],
        test_df=ap["test_df"],
        bias_offsets=ap["bias"],
    )
    all_rmse[ap["label"]] = {m: v / 1000 for m, v in res.items()}  # → GW
    for m, v in res.items():
        print(f"    {m:20s}  {v/1000:.3f} GW")


# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting ...")

fig, ax = plt.subplots(figsize=(14, 5))

approach_labels = [ap["label"] for ap in APPROACHES]
colors          = [ap["color"] for ap in APPROACHES]
n_approaches    = len(APPROACHES)
width           = 0.8 / n_approaches
x               = np.arange(len(MODEL_NAMES))

# Collect all bar positions and values for min-finding
bar_data = []   # list of (model_idx, approach_idx, x_pos, rmse_val, bar_obj)

for i, (ap_label, color) in enumerate(zip(approach_labels, colors)):
    rmse_vals = [all_rmse[ap_label].get(m, np.nan) for m in MODEL_NAMES]
    offset    = (i - n_approaches / 2 + 0.5) * width
    bars      = ax.bar(x + offset, rmse_vals, width=width * 0.9,
                       label=ap_label.replace("\n", " "), color=color, alpha=0.85)
    for j, (bar, val) in enumerate(zip(bars, rmse_vals)):
        if not np.isnan(val):
            bar_data.append((j, i, bar.get_x() + bar.get_width() / 2, val, bar))

# For each model group, find and annotate the minimum bar
for model_idx in range(len(MODEL_NAMES)):
    group = [(ap_i, xpos, val, bar) for (m_i, ap_i, xpos, val, bar) in bar_data if m_i == model_idx]
    if not group:
        continue
    best = min(group, key=lambda t: t[2])
    _, xpos, val, bar = best

    # Highlight with a bold black border
    bar.set_edgecolor("black")
    bar.set_linewidth(2.0)

    # Annotate with value above the bar
    ax.text(xpos, val + 0.15, f"{val:.2f}",
            ha="center", va="bottom", fontsize=7, fontweight="bold", color="black",
            rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES, rotation=20, ha="right")
ax.set_ylabel("RMSE on Jan–Apr 2026 test set (GW)")
ax.set_title("RMSE by Model and Approach — Jan–Apr 2026 Holdout (2,663 hours)")
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.set_ylim(bottom=0)
plt.tight_layout()

FIGURES.mkdir(parents=True, exist_ok=True)
out = FIGURES / "26_2026_test_rmse_comparison.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")
