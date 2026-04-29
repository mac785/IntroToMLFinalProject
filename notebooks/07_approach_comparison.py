"""
Phase 7: Live Approach Comparison — Presentation Day

Runs all trained model variants against today's ERCOT actual load and
produces a side-by-side comparison of every approach.

Current variants (add new entries to VARIANTS to extend):
  - Current          : weather-only features, no correction  (04_regression_models.py)
  - Bias-corrected   : same models + 2024 calibration offset (04c_bias_offsets.py)
  - Lag features     : weather + load_lag_24h + load_lag_168h (04b_regression_lag.py)
  - Days-elapsed (2A): adds days_elapsed trend feature       (04d_regression_trend.py)
  - 2025 retrain (2C): above variants retrained on 2021-2025 (04e_regression_2025train.py)

Figures saved:
  figures/23_approach_comparison_live.png   — best model per variant vs ERCOT actual
  figures/24_approach_rmse_comparison.png   — bar chart of live RMSE by variant
  figures/25_all_models_by_approach.png     — all models per variant (subplots)
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import joblib
import json
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from ISLP.models import ModelSpec as MS
from pathlib import Path
from datetime import date, timedelta
import warnings

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

ROOT    = Path(__file__).resolve().parent.parent
PROC    = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MODELS  = ROOT / "models"

BALANCE_F      = 65.0
PEAK_MW        = 68_029
ERCOT_BLUE     = "#003087"
OMETO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OMETO_HIST     = "https://historical-forecast-api.open-meteo.com/v1/forecast"

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

CITIES = {
    'Austin':     (30.19, -97.67),
    'Dallas':     (32.90, -97.04),
    'Houston':    (29.65, -95.28),
    'SanAntonio': (29.53, -98.47),
}
WEATHER_VARS = [
    "temperature_2m", "apparent_temperature",
    "relative_humidity_2m", "dew_point_2m",
    "wind_speed_10m", "shortwave_radiation",
]

# ── Variant registry ──────────────────────────────────────────────────────────
# Each entry defines one approach. To add Approach 2A or 2C, append here.
#   feature_set : "nolag" or "lag" (or "year", "2025", etc. once scripts run)
#   suffix      : appended to model filenames (_lag, _year, _2025, ...)
#   bias_key    : None = no correction; "file" = load from bias_offsets.json
#   lag_hours   : list of lag hours needed at live prediction time
VARIANTS = [
    {
        "label":       "Current (weather-only)",
        "feature_set": "nolag",
        "suffix":      "",
        "bias_key":    None,
        "lag_hours":   [],
        "color":       "#4C72B0",
    },
    {
        "label":       "Bias-corrected",
        "feature_set": "nolag",
        "suffix":      "",
        "bias_key":    "file",
        "lag_hours":   [],
        "color":       "#DD8452",
    },
    {
        "label":       "Lag features (24h+168h)",
        "feature_set": "lag",
        "suffix":      "_lag",
        "bias_key":    None,
        "lag_hours":   [24, 168],
        "color":       "#55A868",
    },
    {
        "label":       "Days-elapsed trend",
        "feature_set": "trend",
        "suffix":      "_trend",
        "bias_key":    None,
        "lag_hours":   [],
        "color":       "#C44E52",
    },
    # ── 2025 retrain variants ─────────────────────────────────────────────────
    # Trained on full 2021-2025 data; run 04e_regression_2025train.py first
    {
        "label":       "2025 retrain",
        "feature_set": "nolag_full",
        "suffix":      "_2025train",
        "bias_key":    None,
        "lag_hours":   [],
        "color":       "#8172B2",
    },
    {
        "label":       "2025 retrain + lag",
        "feature_set": "lag_full",
        "suffix":      "_2025train_lag",
        "bias_key":    None,
        "lag_hours":   [24, 168],
        "color":       "#937860",
    },
    {
        "label":       "2025 retrain + trend",
        "feature_set": "trend_full",
        "suffix":      "_2025train_trend",
        "bias_key":    None,
        "lag_hours":   [],
        "color":       "#E377C2",
    },
    {
        "label":       "2025 retrain + lag + trend",
        "feature_set": "lag_trend_full",
        "suffix":      "_2025train_lag_trend",
        "bias_key":    None,
        "lag_hours":   [24, 168],
        "color":       "#2CA02C",
    },
]

# ─────────────────────────────────────────────────────────────────────────────

def savefig(name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES / name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {name}")

def _cyclic(s, period):
    a = 2 * np.pi * s / period
    return np.sin(a), np.cos(a)

def _load_env(path):
    env = {}
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return env

def _ercot_token(username, password, sub_key):
    url = ("https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com"
           "/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token")
    client_id = "fec253ea-0d06-4272-a5e6-b478baeecd70"
    r = requests.post(url, timeout=20,
                      headers={"Ocp-Apim-Subscription-Key": sub_key},
                      data={"grant_type": "password", "client_id": client_id,
                            "scope": f"openid {client_id} offline_access",
                            "response_type": "id_token",
                            "username": username, "password": password})
    return r.json().get("id_token") if r.status_code == 200 else None

def _fetch_ercot_hourly(hdrs, target_date):
    """Return dict {hour 0-23: MW} for a given date using NP6-345-CD."""
    date_str = target_date.strftime("%Y-%m-%d")
    r = requests.get(
        "https://api.ercot.com/api/public-reports/np6-345-cd/act_sys_load_by_wzn",
        headers=hdrs, timeout=20,
        params={"operatingDayFrom": date_str, "operatingDayTo": date_str})
    if r.status_code != 200 or not r.json().get("data"):
        return {}
    data        = r.json()
    field_names = [f["name"] for f in data["fields"]]
    result      = {}
    for row in data["data"]:
        r2 = dict(zip(field_names, row))
        try:
            h  = int(r2["hourEnding"].split(":")[0]) - 1
            mw = float(r2["total"])
            if 0 <= h <= 23 and mw > 0:
                result[h] = mw
        except (ValueError, KeyError):
            continue
    return result

def _parse_15min(data):
    """Parse NP6-235-CD 15-min response → {hour: avg MW}."""
    field_names = [f["name"] for f in data.get("fields", [])]
    buckets = {}
    for row in data.get("data", []):
        r2 = dict(zip(field_names, row))
        try:
            h  = int(r2["timeEnding"].split(":")[0])
            mw = float(r2["demand"])
            if 0 <= h <= 23 and mw > 0:
                buckets.setdefault(h, []).append(mw)
        except (ValueError, KeyError):
            continue
    return {h: sum(v) / len(v) for h, v in buckets.items()}


class ERCOTModel(nn.Module):
    """Matches architecture in 04_regression_models.py / 04b. input_size varies."""
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


today = date.today()
print(f"=== Approach Comparison  [{today}] ===\n")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Weather forecast
# ══════════════════════════════════════════════════════════════════════════════
print("Step 1: Weather forecast ...")
city_frames = {}
for city, (lat, lon) in CITIES.items():
    params = {"latitude": lat, "longitude": lon,
              "hourly": ",".join(WEATHER_VARS),
              "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
              "timezone": "America/Chicago",
              "start_date": str(today), "end_date": str(today)}
    for wx_url in [OMETO_FORECAST, OMETO_HIST]:
        r = requests.get(wx_url, params=params, timeout=30)
        if r.status_code == 200:
            break
    r.raise_for_status()
    h = r.json()["hourly"]
    city_frames[city] = pd.DataFrame({
        "HourEnding": pd.to_datetime(h["time"]),
        **{f"{v}_{city}": h[v] for v in WEATHER_VARS}})

wx = city_frames["Austin"].copy()
for city in ["Dallas", "Houston", "SanAntonio"]:
    wx = wx.merge(city_frames[city], on="HourEnding")
for v in WEATHER_VARS:
    wx[f"{v}_avg"] = wx[[f"{v}_{c}" for c in CITIES]].mean(axis=1)
wx = wx.reset_index(drop=True)
print(f"  {len(wx)} hourly forecasts  "
      f"({wx['temperature_2m_avg'].min():.1f}–{wx['temperature_2m_avg'].max():.1f}°F)")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Feature engineering
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2: Feature engineering ...")
ts = wx["HourEnding"]
wx["hour"]        = ts.dt.hour
wx["day_of_week"] = ts.dt.dayofweek
wx["month"]       = ts.dt.month
wx["is_weekend"]  = (wx["day_of_week"] >= 5).astype(int)
wx["season"]      = pd.cut(wx["month"], bins=[0, 2, 5, 8, 11, 12],
                            labels=[0, 1, 2, 3, 0], ordered=False).astype(int)
wx["hour_sin"],  wx["hour_cos"]  = _cyclic(wx["hour"],        24)
wx["month_sin"], wx["month_cos"] = _cyclic(wx["month"],       12)
wx["dow_sin"],   wx["dow_cos"]   = _cyclic(wx["day_of_week"],  7)
try:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    holidays = USFederalHolidayCalendar().holidays(start=ts.min(), end=ts.max())
    wx["is_holiday"] = ts.dt.normalize().isin(holidays).astype(int)
except Exception:
    wx["is_holiday"] = 0
wx["is_workday"]             = ((wx["is_weekend"] == 0) & (wx["is_holiday"] == 0)).astype(int)
temp = wx["temperature_2m_avg"]
wx["CDH"]                     = (temp - BALANCE_F).clip(lower=0)
wx["HDH"]                     = (BALANCE_F - temp).clip(lower=0)
wx["temp_sq"]                 = temp ** 2
wx["apparent_temp_delta_avg"] = wx["apparent_temperature_avg"] - temp
wx["temp_x_hour_sin"]         = temp * wx["hour_sin"]
wx["CDH_x_hour_sin"]          = wx["CDH"] * wx["hour_sin"]
wx["temp_x_is_weekend"]       = temp * wx["is_weekend"]

X_base = wx[FEATURE_COLS].values


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — ERCOT actual load + lag values
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 3: Fetching ERCOT data ...")
env      = _load_env(ROOT / ".env")
sub_key  = env.get("ERCOT_PRIMARY_KEY", "")
username = env.get("ERCOT_USERNAME", "")
password = env.get("ERCOT_PASSWORD", "")

actual_by_hour = {}
lag_by_hour    = {24: {}, 168: {}}   # hour → MW for each lag offset

if sub_key and username and password:
    token = _ercot_token(username, password, sub_key)
    if token:
        hdrs = {"Ocp-Apim-Subscription-Key": sub_key,
                "Authorization": f"Bearer {token}"}

        # Today's actual (NP6-235-CD real-time 15-min)
        r = requests.get(
            "https://api.ercot.com/api/public-reports/np6-235-cd/system_wide_demand",
            headers=hdrs, timeout=20,
            params={"deliveryDateFrom": today.strftime("%Y-%m-%d"),
                    "deliveryDateTo":   today.strftime("%Y-%m-%d")})
        if r.status_code == 200 and r.json().get("_meta", {}).get("totalRecords", 0) > 0:
            actual_by_hour = _parse_15min(r.json())
            print(f"  Today actual (NP6-235-CD): {len(actual_by_hour)} hours")
        else:
            # Fall back to yesterday's NP6-345-CD
            actual_by_hour = _fetch_ercot_hourly(hdrs, today - timedelta(days=1))
            print(f"  Fallback to yesterday actual: {len(actual_by_hour)} hours")

        # Lag data: yesterday (24h) and last week (168h)
        for lag_h in [24, 168]:
            lag_date  = today - timedelta(hours=lag_h)
            lag_by_hour[lag_h] = _fetch_ercot_hourly(hdrs, lag_date)
            n = len(lag_by_hour[lag_h])
            print(f"  Lag {lag_h:3d}h ({lag_date}): {n} hours")
    else:
        print("  OAuth2 failed — running without ERCOT actual data.")
else:
    print("  ERCOT credentials not in .env — running without actual data.")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Load training data for OLS refits + MLP scalers
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 4: Loading training data ...")
# 2021-2024 (original train split) — used by baseline/bias/lag/trend variants
train_nolag = pd.read_csv(PROC / "features_nolag.csv", parse_dates=["HourEnding"])
train_nolag = train_nolag[train_nolag["HourEnding"].dt.year < 2025]

train_lag = pd.read_csv(PROC / "features_lag.csv", parse_dates=["HourEnding"])
train_lag = train_lag[train_lag["HourEnding"].dt.year < 2025]

train_trend = pd.read_csv(PROC / "features_trend.csv", parse_dates=["HourEnding"])
train_trend = train_trend[train_trend["HourEnding"].dt.year < 2025]

# 2021-2025 (full) — used by 2C retrain variants (04e_regression_2025train.py)
train_nolag_full     = pd.read_csv(PROC / "features_nolag.csv",     parse_dates=["HourEnding"])
train_lag_full       = pd.read_csv(PROC / "features_lag.csv",       parse_dates=["HourEnding"])
train_trend_full     = pd.read_csv(PROC / "features_trend.csv",     parse_dates=["HourEnding"])
train_lag_trend_full = pd.read_csv(PROC / "features_lag_trend.csv", parse_dates=["HourEnding"])
print(f"  train_nolag_full: {len(train_nolag_full):,} rows "
      f"({train_nolag_full['HourEnding'].min().year}–{train_nolag_full['HourEnding'].max().year})")

bias_offsets = {}
bias_path    = MODELS / "bias_offsets.json"
if bias_path.exists():
    with open(bias_path) as f:
        bias_offsets = json.load(f)
    print(f"  Bias offsets loaded ({len(bias_offsets)} models)")
else:
    print("  bias_offsets.json not found — bias-corrected variant will show raw predictions.")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Inference for each variant
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 5: Running inference ...")

# Build lag feature rows for today (one value per hour)
LAG_COLS = ['load_lag_24h', 'load_lag_168h']
lag_matrix = np.zeros((24, 2))  # rows=hours, cols=[lag24, lag168]
for i, h in enumerate(range(24)):
    lag_matrix[i, 0] = lag_by_hour[24].get(h, np.nan)
    lag_matrix[i, 1] = lag_by_hour[168].get(h, np.nan)
X_lag_full = np.hstack([X_base, lag_matrix])

# days_elapsed for today: fractional days since 2021-01-01 for each hour
ORIGIN = pd.Timestamp("2021-01-01")
days_elapsed_live = np.array(
    [(ts - ORIGIN).total_seconds() / 86400 for ts in wx["HourEnding"]]
).reshape(-1, 1)
X_trend_full        = np.hstack([X_base, days_elapsed_live])
X_lag_trend_full    = np.hstack([X_base, lag_matrix, days_elapsed_live])

FEATURE_SETS = {
    "nolag":          (X_base,            FEATURE_COLS,                              train_nolag),
    "lag":            (X_lag_full,        FEATURE_COLS + LAG_COLS,                   train_lag),
    "trend":          (X_trend_full,      FEATURE_COLS + ['days_elapsed'],            train_trend),
    # 2C variants: same live features, but OLS fitted on full 2021-2025 training data
    "nolag_full":     (X_base,            FEATURE_COLS,                              train_nolag_full),
    "lag_full":       (X_lag_full,        FEATURE_COLS + LAG_COLS,                   train_lag_full),
    "trend_full":     (X_trend_full,      FEATURE_COLS + ['days_elapsed'],            train_trend_full),
    "lag_trend_full": (X_lag_trend_full,  FEATURE_COLS + LAG_COLS + ['days_elapsed'], train_lag_trend_full),
}

SKLEARN_MODELS = ["Ridge", "Lasso", "Decision Tree",
                  "Random Forest", "Gradient Boost", "SVR"]
MODEL_FILES    = {
    "Ridge":          "ridge{s}.pkl",
    "Lasso":          "lasso{s}.pkl",
    "Decision Tree":  "decision_tree{s}.pkl",
    "Random Forest":  "random_forest{s}.pkl",
    "Gradient Boost": "gradient_boosting{s}.pkl",
    "SVR":            "svr{s}.pkl",
}

variant_preds = {}   # variant_label → {model_label: array of 24 predictions}

for v in VARIANTS:
    label      = v["label"]
    suffix     = v["suffix"]
    feat_set   = v["feature_set"]
    bias_key   = v["bias_key"]

    if feat_set not in FEATURE_SETS:
        print(f"  [{label}] feature set '{feat_set}' not built yet — skipping.")
        continue

    X_live, feat_cols, train_df = FEATURE_SETS[feat_set]

    # Check required lag data is available
    if v["lag_hours"] and any(np.isnan(lag_matrix[:, i]).all()
                               for i, lh in enumerate(v["lag_hours"])
                               if lh in [24, 168]):
        print(f"  [{label}] lag data unavailable — skipping.")
        continue

    preds = {}

    # OLS
    design   = MS(feat_cols).fit(train_df)
    ols_res  = sm.OLS(train_df["Load_MW"], design.transform(train_df)).fit()
    ols_live = pd.DataFrame(X_live, columns=feat_cols)
    preds["OLS"] = ols_res.predict(design.transform(ols_live)).values

    # sklearn models
    for model_label in SKLEARN_MODELS:
        fpath = MODELS / MODEL_FILES[model_label].format(s=suffix)
        if not fpath.exists():
            print(f"  [{label}] {fpath.name} not found — skipping {model_label}.")
            continue
        preds[model_label] = joblib.load(fpath).predict(X_live)

    # MLP
    mlp_scaler_path = MODELS / f"mlp_scaler{suffix}.pkl"
    mlp_state_path  = MODELS / f"mlp_state{suffix}.pt"
    if mlp_scaler_path.exists() and mlp_state_path.exists():
        mlp_sc = joblib.load(mlp_scaler_path)
        X_mlp  = torch.tensor(mlp_sc.transform(X_live).astype(np.float32))
        mlp_m  = ERCOTModel(X_live.shape[1])
        mlp_m.load_state_dict(torch.load(mlp_state_path, map_location="cpu"))
        mlp_m.eval()
        with torch.no_grad():
            preds["MLP"] = mlp_m(X_mlp).numpy()
    else:
        # Fall back to baseline MLP only when feature dimensions match (27 features)
        state_fallback = MODELS / "mlp_state.pt"
        if state_fallback.exists() and X_live.shape[1] == len(FEATURE_COLS):
            mlp_sc = StandardScaler().fit(train_df[feat_cols].values)
            X_mlp  = torch.tensor(mlp_sc.transform(X_live).astype(np.float32))
            mlp_m  = ERCOTModel(X_live.shape[1])
            mlp_m.load_state_dict(torch.load(state_fallback, map_location="cpu"))
            mlp_m.eval()
            with torch.no_grad():
                preds["MLP"] = mlp_m(X_mlp).numpy()
        else:
            print(f"  [{label}] MLP weights not found — skipping MLP.")

    # Apply bias correction if requested
    if bias_key == "file" and bias_offsets:
        for model_label, arr in preds.items():
            bias_model_key = model_label.replace("Gradient Boost", "Gradient Boost")
            offset = bias_offsets.get(bias_model_key, 0.0)
            preds[model_label] = arr + offset

    variant_preds[label] = preds
    n_models = len(preds)
    print(f"  [{label}]  {n_models} models ready.")


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Live accuracy per variant (best model = SVR or RF)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"  APPROACH COMPARISON — {today.strftime('%B %d, %Y')}")
print(f"{'='*90}")

COMPARE_MODEL = "Random Forest"   # use same model across variants for fair comparison

hours_list = list(range(24))
if actual_by_hour:
    print(f"\n  Live accuracy ({len(actual_by_hour)} hours)  —  {COMPARE_MODEL}\n")
    header = f"  {'Hour':>4}  {'Actual GW':>9}  " + \
             "".join(f"  {v['label'][:18]:>18}" for v in VARIANTS
                     if v["label"] in variant_preds)
    print(header)
    print(f"  {'-'*80}")

    rmse_by_variant = {v["label"]: [] for v in VARIANTS}

    for h in hours_list:
        act = actual_by_hour.get(h)
        row = f"  {h:02d}:00  " + (f"{act/1000:>8.2f} GW" if act else "      n/a")
        for v in VARIANTS:
            vlabel = v["label"]
            if vlabel not in variant_preds:
                continue
            p = variant_preds[vlabel].get(COMPARE_MODEL)
            if p is None:
                row += "                    n/a"
            else:
                row += f"  {p[h]/1000:>16.2f} GW"
                if act:
                    rmse_by_variant[vlabel].append((p[h] - act)**2)
        print(row)

    print(f"\n  RMSE ({COMPARE_MODEL}):")
    for v in VARIANTS:
        vlabel = v["label"]
        errs   = rmse_by_variant.get(vlabel, [])
        if errs:
            rmse = np.sqrt(np.mean(errs))
            print(f"    {vlabel:<35}  {rmse:>8,.0f} MW")
else:
    print("\n  No ERCOT actual data — showing predictions only.\n")
    for v in VARIANTS:
        vlabel = v["label"]
        if vlabel not in variant_preds:
            continue
        p = variant_preds[vlabel].get(COMPARE_MODEL, np.zeros(24))
        print(f"  {vlabel}")
        for h in range(24):
            print(f"    {h:02d}:00  {p[h]/1000:.2f} GW")


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Figures
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 7: Generating figures ...")
hours     = wx["HourEnding"]
MODEL_COLORS = {
    "OLS":            "#4C72B0",
    "Ridge":          "#DD8452",
    "Lasso":          "#55A868",
    "Decision Tree":  "#C44E52",
    "Random Forest":  "#8172B2",
    "Gradient Boost": "#937860",
    "SVR":            "#DA8BC3",
    "MLP":  "#8C8C8C",
}

# Pre-compute per-model RMSE for every variant (used by figs 23 and 24)
all_rmse = {}   # variant_label → {model_label: RMSE float}
for v in VARIANTS:
    vlabel = v["label"]
    if vlabel not in variant_preds:
        continue
    model_rmses = {}
    for model_label, p in variant_preds[vlabel].items():
        if actual_by_hour:
            errs = [(p[h] - actual_by_hour[h])**2
                    for h in range(24) if h in actual_by_hour]
            if errs:
                model_rmses[model_label] = np.sqrt(np.mean(errs))
    all_rmse[vlabel] = model_rmses

def _best_model(vlabel):
    """Return (model_label, predictions) with lowest live RMSE for this variant."""
    rmses = all_rmse.get(vlabel, {})
    preds = variant_preds.get(vlabel, {})
    if rmses:
        best = min(rmses, key=rmses.get)
    elif preds:
        best = next(iter(preds))   # no actuals — just pick first
    else:
        return None, None
    return best, preds.get(best)


# ── Fig 23: Best model per variant vs ERCOT actual ──────────────────────────
fig, ax = subplots(figsize=(14, 5))

for v in VARIANTS:
    vlabel = v["label"]
    best_label, p = _best_model(vlabel)
    if p is None:
        continue
    rmse_str = (f"  RMSE={all_rmse[vlabel][best_label]/1000:.2f} GW"
                if best_label in all_rmse.get(vlabel, {}) else "")
    legend_label = f"{vlabel}  [{best_label}{rmse_str}]"
    ax.plot(hours, p / 1000, lw=2, color=v["color"], label=legend_label, alpha=0.9)

if actual_by_hour:
    act_h  = sorted(actual_by_hour)
    act_mw = [actual_by_hour[h] for h in act_h]
    ax.plot(hours.iloc[act_h], np.array(act_mw) / 1000,
            color="black", lw=3, label="ERCOT Actual", zorder=10)

ax.axhline(PEAK_MW / 1000, color="red", lw=1, ls="--", alpha=0.5,
           label=f"Peak threshold ({PEAK_MW/1000:.0f} GW)")
ax.set_ylabel("Load (GW)")
ax.set_title(f"Best Model per Approach vs ERCOT Actual — {today.strftime('%B %d, %Y')}")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I%p"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
savefig("23_approach_comparison_live.png")


# ── Fig 24: RMSE bar chart (all models, all variants) ───────────────────────
if actual_by_hour and all_rmse:
    all_model_names = ["OLS", "Ridge", "Lasso", "Decision Tree",
                       "Random Forest", "Gradient Boost", "SVR", "MLP"]
    rmse_df = pd.DataFrame(all_rmse).reindex(
        [m for m in all_model_names
         if any(m in all_rmse[k] for k in all_rmse)])

    fig, ax = subplots(figsize=(12, 5))
    x      = np.arange(len(rmse_df))
    n_vars = len(rmse_df.columns)
    width  = 0.8 / max(n_vars, 1)

    for i, vlabel in enumerate(rmse_df.columns):
        vcolor = next(v["color"] for v in VARIANTS if v["label"] == vlabel)
        offset = (i - n_vars / 2 + 0.5) * width
        ax.bar(x + offset, rmse_df[vlabel].values / 1000,
               width=width * 0.9, label=vlabel, color=vcolor, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(rmse_df.index, rotation=30, ha="right")
    ax.set_ylabel("RMSE (GW)")
    ax.set_title(f"Live RMSE by Model and Approach — {today.strftime('%B %d, %Y')}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("24_approach_rmse_comparison.png")


# ── Fig 25: All models for each variant in one figure (4×2 grid) ─────────────
active_variants = [v for v in VARIANTS if v["label"] in variant_preds]
n_panels        = len(active_variants)
n_cols          = 4
n_rows          = (n_panels + n_cols - 1) // n_cols
fig, axes       = plt.subplots(n_rows, n_cols,
                               figsize=(5 * n_cols, 4 * n_rows),
                               sharex=True, sharey=True)
axes_flat = axes.flatten() if n_panels > 1 else [axes]

act_h  = sorted(actual_by_hour) if actual_by_hour else []
act_mw = [actual_by_hour[h] for h in act_h]

for ax, v in zip(axes_flat, active_variants):
    vlabel = v["label"]
    preds  = variant_preds[vlabel]

    for model_label, p in preds.items():
        rmse_str = (f"  {all_rmse[vlabel][model_label]/1000:.2f} GW"
                    if model_label in all_rmse.get(vlabel, {}) else "")
        ax.plot(hours, p / 1000, lw=1.3, alpha=0.85,
                color=MODEL_COLORS.get(model_label, "gray"),
                label=f"{model_label}{rmse_str}")

    if act_h:
        ax.plot(hours.iloc[act_h], np.array(act_mw) / 1000,
                color="black", lw=2.2, label="ERCOT Actual", zorder=10)

    ax.axhline(PEAK_MW / 1000, color="red", lw=1, ls="--", alpha=0.4,
               label="Peak threshold (90th pct)" if ax is axes_flat[0] else "_nolegend_")
    ax.set_ylabel("Load (GW)")
    ax.set_title(vlabel, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I%p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    pass  # legend handled below

# Hide any unused axes (in case n_panels doesn't fill the grid)
for ax in axes_flat[n_panels:]:
    ax.set_visible(False)

# Single shared legend below the grid (strip per-panel RMSE from labels)
handles, labels = axes_flat[0].get_legend_handles_labels()
clean_labels = [lbl.split('  ')[0] for lbl in labels]
fig.legend(handles, clean_labels,
           loc="lower center", bbox_to_anchor=(0.5, 0),
           ncol=5, fontsize=8, framealpha=0.9)

plt.suptitle(f"All Models by Approach — {today.strftime('%B %d, %Y')}",
             fontsize=14, y=1.02)
plt.tight_layout(rect=[0, 0.09, 1, 1])
savefig("25_all_models_by_approach.png")

print(f"\nFigures saved to {FIGURES}")
print("Phase 7 complete.")
