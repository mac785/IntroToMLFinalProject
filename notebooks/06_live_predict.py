"""
Phase 6: Live Prediction — Presentation Day (April 22, 2026)

Usage:
    conda run -n islp python notebooks/06_live_predict.py

What this does:
  1. Fetches today's hourly weather forecast from Open-Meteo (same API as training)
  2. Engineers the same no-lag feature set used in training
  3. Loads all 7 regression models + 4 classification models
  4. Tries to fetch ERCOT actual system load from their public dashboard API
  5. Produces two presentation figures + a live comparison table in the terminal

Figures saved:
  figures/21_live_predictions.png   — all regression models vs ERCOT actual
  figures/22_live_peak_prob.png     — SVR forecast + GBC peak probability bars
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import joblib
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from ISLP.models import ModelSpec as MS
from pathlib import Path
from datetime import date
import warnings

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

ROOT    = Path(__file__).resolve().parent.parent
PROC    = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MODELS  = ROOT / "models"

BALANCE_F       = 65.0
PEAK_MW         = 68_029        # 90th-pct threshold from training data
GBC_OPT_THRESH  = 0.01          # optimal recall threshold from Phase 5 analysis
ERCOT_BLUE      = "#003087"
PEAK_RED        = "#D62728"

# Feature columns — must match training exactly (same order)
FEATURE_COLS = [
    'temperature_2m_Austin',    'temperature_2m_Dallas',
    'temperature_2m_Houston',   'temperature_2m_SanAntonio',
    'temperature_2m_avg',       'apparent_temperature_avg',
    'relative_humidity_2m_avg', 'dew_point_2m_avg',
    'wind_speed_10m_avg',       'shortwave_radiation_avg',
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

OMETO_FORECAST      = "https://api.open-meteo.com/v1/forecast"
OMETO_HIST_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"

def savefig(name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES / name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {name}")

def _cyclic(s, period):
    a = 2 * np.pi * s / period
    return np.sin(a), np.cos(a)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Weather forecast
# ══════════════════════════════════════════════════════════════════════════════
today = date.today()
print(f"=== ERCOT Live Prediction  [{today}] ===\n")
print("Step 1: Fetching hourly weather forecast from Open-Meteo ...")

city_frames = {}
for city, (lat, lon) in CITIES.items():
    params = {
        "latitude":         lat,
        "longitude":        lon,
        "hourly":           ",".join(WEATHER_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit":  "mph",
        "timezone":         "America/Chicago",
        "start_date":       str(today),
        "end_date":         str(today),
    }
    # Try live forecast first; fall back to historical-forecast API
    for wx_url in [OMETO_FORECAST, OMETO_HIST_FORECAST]:
        r = requests.get(wx_url, params=params, timeout=30)
        if r.status_code == 200:
            break
    r.raise_for_status()
    h = r.json()["hourly"]
    df_c = pd.DataFrame({
        "HourEnding": pd.to_datetime(h["time"]),
        **{f"{v}_{city}": h[v] for v in WEATHER_VARS},
    })
    city_frames[city] = df_c

wx = city_frames["Austin"].copy()
for city in ["Dallas", "Houston", "SanAntonio"]:
    wx = wx.merge(city_frames[city], on="HourEnding")

for v in WEATHER_VARS:
    wx[f"{v}_avg"] = wx[[f"{v}_{c}" for c in CITIES]].mean(axis=1)

wx = wx.reset_index(drop=True)
print(f"  {len(wx)} hourly forecasts  |  "
      f"temp range: {wx['temperature_2m_avg'].min():.1f}–{wx['temperature_2m_avg'].max():.1f}°F")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Feature engineering (mirrors 02_feature_engineering.py)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2: Engineering features ...")

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

wx["is_workday"] = ((wx["is_weekend"] == 0) & (wx["is_holiday"] == 0)).astype(int)

temp = wx["temperature_2m_avg"]
wx["CDH"]                    = (temp - BALANCE_F).clip(lower=0)
wx["HDH"]                    = (BALANCE_F - temp).clip(lower=0)
wx["temp_sq"]                = temp ** 2
wx["apparent_temp_delta_avg"] = wx["apparent_temperature_avg"] - temp
wx["temp_x_hour_sin"]         = temp * wx["hour_sin"]
wx["CDH_x_hour_sin"]          = wx["CDH"] * wx["hour_sin"]
wx["temp_x_is_weekend"]       = temp * wx["is_weekend"]

X_live = wx[FEATURE_COLS].values


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Load training data for OLS design matrix + MLP scaler
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 3: Loading training data (for OLS re-fit + MLP scaler) ...")
train_df = pd.read_csv(PROC / "features_nolag.csv", parse_dates=["HourEnding"])
train_df = train_df[train_df["HourEnding"].dt.year < 2025]

X_train = train_df[FEATURE_COLS].values
y_train = train_df["Load_MW"].values

# Scaler for MLP (only model that requires external scaling)
mlp_scaler = StandardScaler().fit(X_train)

# OLS via statsmodels — fast to re-fit (~0.5s)
design      = MS(FEATURE_COLS).fit(train_df)
ols_result  = sm.OLS(y_train, design.transform(train_df)).fit()
X_sm_live   = design.transform(wx)
print(f"  OLS R² (train): {ols_result.rsquared:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Load sklearn regression models
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 4: Loading regression models ...")
ridge_pipe = joblib.load(MODELS / "ridge.pkl")
lasso_pipe = joblib.load(MODELS / "lasso.pkl")
dtr_pipe   = joblib.load(MODELS / "decision_tree.pkl")
rf_model   = joblib.load(MODELS / "random_forest.pkl")
gbr_model  = joblib.load(MODELS / "gradient_boosting.pkl")
svr_pipe   = joblib.load(MODELS / "svr.pkl")
print("  Ridge, Lasso, DT, RF, GBM, SVR  loaded.")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Load PyTorch MLP
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 5: Loading PyTorch MLP ...")

class ERCOTModel(nn.Module):
    """Must match architecture in 04_regression_models.py exactly."""
    def __init__(self, input_size):
        super(ERCOTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1))
    def forward(self, x):
        return torch.flatten(self.sequential(self.flatten(x)))

mlp_model = ERCOTModel(len(FEATURE_COLS))
mlp_model.load_state_dict(torch.load(MODELS / "mlp_state.pt", map_location="cpu"))
mlp_model.eval()
print("  PyTorch MLP loaded.")


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Load classification models
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 6: Loading classification models ...")
scaler_clf_r, ridge_logit = joblib.load(MODELS / "logit_ridge.pkl")
scaler_clf_l, lasso_logit = joblib.load(MODELS / "logit_lasso.pkl")
rfc_clf = joblib.load(MODELS / "rf_classifier.pkl")
gbc_clf = joblib.load(MODELS / "gbc_classifier.pkl")
print("  Logistic+Ridge, Logistic+Lasso, RFC, GBC  loaded.")


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Generate predictions
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 7: Running inference ...")

X_live_mlp = torch.tensor(mlp_scaler.transform(X_live).astype(np.float32))

reg_preds = {
    "OLS":            ols_result.predict(X_sm_live).values,
    "Ridge":          ridge_pipe.predict(X_live),
    "Lasso":          lasso_pipe.predict(X_live),
    "Decision Tree":  dtr_pipe.predict(X_live),
    "Random Forest":  rf_model.predict(X_live),
    "Gradient Boost": gbr_model.predict(X_live),
    "SVR":            svr_pipe.predict(X_live),
    "MLP (PyTorch)":  mlp_model(X_live_mlp).detach().numpy(),
}

clf_probs = {
    "Logistic+Ridge": ridge_logit.predict_proba(scaler_clf_r.transform(X_live))[:, 1],
    "Logistic+Lasso": lasso_logit.predict_proba(scaler_clf_l.transform(X_live))[:, 1],
    "Random Forest":  rfc_clf.predict_proba(X_live)[:, 1],
    "Gradient Boost": gbc_clf.predict_proba(X_live)[:, 1],
}
gbc_peak = (clf_probs["Gradient Boost"] >= GBC_OPT_THRESH).astype(int)
print(f"  Inference complete.  GBC peak hours flagged: {gbc_peak.sum()}/24")


# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Fetch ERCOT actual load via authenticated public API
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 8: Fetching ERCOT actual load ...")
actual_by_hour = {}   # hour (0-23) → MW

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
    """ERCOT OAuth2 ROPC — returns id_token string or None."""
    url = ("https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com"
           "/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token")
    client_id = "fec253ea-0d06-4272-a5e6-b478baeecd70"
    r = requests.post(url, timeout=20,
                      headers={"Ocp-Apim-Subscription-Key": sub_key},
                      data={
                          "grant_type":    "password",
                          "client_id":     client_id,
                          "scope":         f"openid {client_id} offline_access",
                          "response_type": "id_token",
                          "username":      username,
                          "password":      password,
                      })
    if r.status_code == 200:
        return r.json().get("id_token")
    print(f"  OAuth2 [{r.status_code}]: {r.text[:300]}")
    return None

env      = _load_env(ROOT / ".env")
sub_key  = env.get("ERCOT_PRIMARY_KEY", "")
username = env.get("ERCOT_USERNAME", "")
password = env.get("ERCOT_PASSWORD", "")

def _parse_ercot_rows(data, hour_fn, mw_fn):
    """
    Parse ERCOT list-of-lists response into actual_by_hour dict.
    hour_fn(row) → int 0-23, mw_fn(row) → float MW
    """
    result = {}
    field_names = [f["name"] for f in data.get("fields", [])]
    for row in data.get("data", []):
        r = dict(zip(field_names, row))
        try:
            h  = hour_fn(r)
            mw = mw_fn(r)
            if 0 <= h <= 23 and mw > 0:
                if h not in result:
                    result[h] = []
                result[h].append(mw)
        except (ValueError, TypeError, KeyError):
            continue
    return {h: sum(v) / len(v) for h, v in result.items()}   # average if multiple

if sub_key and username and password:
    token = _ercot_token(username, password, sub_key)
    if token:
        print("  OAuth2 token obtained.")
        hdrs      = {"Ocp-Apim-Subscription-Key": sub_key,
                     "Authorization":              f"Bearer {token}"}
        today_str = today.strftime("%Y-%m-%d")

        # Primary: NP6-235-CD — System-Wide Demand (15-min, real-time)
        try:
            r = requests.get(
                "https://api.ercot.com/api/public-reports/np6-235-cd/system_wide_demand",
                headers=hdrs, timeout=20,
                params={"deliveryDateFrom": today_str, "deliveryDateTo": today_str})
            r.raise_for_status()
            data = r.json()
            if data.get("_meta", {}).get("totalRecords", 0) > 0:
                # timeEnding "HH:MM" → hour = int(HH); average all 15-min in that hour
                actual_by_hour = _parse_ercot_rows(
                    data,
                    hour_fn=lambda r: int(r["timeEnding"].split(":")[0]),
                    mw_fn  =lambda r: float(r["demand"]))
                print(f"  NP6-235-CD (real-time 15-min): {len(actual_by_hour)} hours")
        except Exception as e:
            print(f"  NP6-235-CD failed: {e}")

        # Fallback: NP6-345-CD — Actual System Load by Weather Zone (end-of-day)
        if not actual_by_hour:
            from datetime import timedelta
            fallback_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            try:
                r = requests.get(
                    "https://api.ercot.com/api/public-reports/np6-345-cd/act_sys_load_by_wzn",
                    headers=hdrs, timeout=20,
                    params={"operatingDayFrom": fallback_date,
                            "operatingDayTo":   fallback_date})
                r.raise_for_status()
                data = r.json()
                if data.get("_meta", {}).get("totalRecords", 0) > 0:
                    # hourEnding "HH:MM" 1-24 → 0-23
                    actual_by_hour = _parse_ercot_rows(
                        data,
                        hour_fn=lambda r: int(r["hourEnding"].split(":")[0]) - 1,
                        mw_fn  =lambda r: float(r["total"]))
                    print(f"  NP6-345-CD (yesterday {fallback_date}): "
                          f"{len(actual_by_hour)} hours  [no today data yet]")
            except Exception as e:
                print(f"  NP6-345-CD fallback failed: {e}")
    else:
        print("  Skipping ERCOT API (token exchange failed).")
else:
    missing = [k for k, v in [("ERCOT_PRIMARY_KEY", sub_key),
                               ("ERCOT_USERNAME", username),
                               ("ERCOT_PASSWORD", password)] if not v]
    print(f"  Missing .env keys: {missing} — skipping API call.")

if not actual_by_hour:
    print("  No ERCOT actual data available — showing predictions only.")


# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — Figures
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 9: Generating figures ...")

hours  = wx["HourEnding"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
          "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]

# ── Fig 21: All regression models + ERCOT actual ─────────────────────────────
fig, ax = subplots(figsize=(14, 5))
for (label, pred), color in zip(reg_preds.items(), colors):
    ax.plot(hours, pred / 1000, lw=1.5, alpha=0.8, color=color, label=label)

if actual_by_hour:
    act_h  = sorted(actual_by_hour)
    act_mw = [actual_by_hour[h] for h in act_h]
    ax.plot(hours.iloc[act_h], np.array(act_mw) / 1000,
            color="black", lw=3, label="ERCOT Actual", zorder=10)

ax.axhline(PEAK_MW / 1000, color=PEAK_RED, lw=1, ls="--", alpha=0.7,
           label=f"Peak threshold ({PEAK_MW/1000:.0f} GW)")
ax.set_ylabel("Load (GW)")
ax.set_title(f"Live ERCOT Load Predictions — {today.strftime('%B %d, %Y')}")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I%p"))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.legend(fontsize=8, ncol=3, loc="upper left")
plt.tight_layout()
savefig("21_live_predictions.png")


# ── Fig 22: SVR forecast (top) + GBC peak probability bars (bottom) ──────────
fig, (ax1, ax2) = subplots(2, 1, figsize=(14, 6))

ax1.plot(hours, reg_preds["SVR"] / 1000,
         color=ERCOT_BLUE, lw=2.5, label="SVR prediction")
if actual_by_hour:
    act_h  = sorted(actual_by_hour)
    act_mw = [actual_by_hour[h] for h in act_h]
    ax1.plot(hours.iloc[act_h], np.array(act_mw) / 1000,
             color="black", lw=2.5, ls="-", label="ERCOT Actual", zorder=10)
ax1.axhline(PEAK_MW / 1000, color=PEAK_RED, lw=1, ls="--", alpha=0.6)
ax1.set_ylabel("Load (GW)")
ax1.set_title(f"SVR Forecast vs ERCOT Actual  [{today.strftime('%B %d, %Y')}]")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%-I%p"))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax1.legend(fontsize=10)

prob = clf_probs["Gradient Boost"]
bar_colors = [PEAK_RED if p >= GBC_OPT_THRESH else ERCOT_BLUE for p in prob]
ax2.bar(hours, prob, width=pd.Timedelta(minutes=50), color=bar_colors, alpha=0.85)
ax2.axhline(GBC_OPT_THRESH, color="gray", lw=1.2, ls="--",
            label=f"Threshold = {GBC_OPT_THRESH:.2f} (optimal recall)")
ax2.set_ylabel("Peak Probability")
ax2.set_xlabel(f"Hour (Central Time)  |  {today.strftime('%B %d, %Y')}")
ax2.set_title("GBC — Peak Hour Classification Probability")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%-I%p"))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax2.legend(fontsize=9)

plt.tight_layout()
savefig("22_live_peak_probability.png")


# ══════════════════════════════════════════════════════════════════════════════
# Step 10 — Live comparison table
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"  LIVE ERCOT LOAD FORECAST — {today.strftime('%B %d, %Y')}")
print(f"{'='*110}")
hdr = (f"  {'Hour':>5}  {'Temp°F':>7}  {'CDH':>5}  "
       f"{'OLS':>7}  {'Ridge':>7}  {'RF':>7}  {'GBM':>7}  {'SVR':>7}  {'MLP':>7}  "
       f"{'Actual':>8}  {'Δ SVR':>7}  {'Peak%':>6}  {'⚡':>4}")
print(hdr)
print(f"  {'-'*105}")

rmse_accum = {k: [] for k in reg_preds}
mape_accum = {k: [] for k in reg_preds}

for i in range(len(wx)):
    h        = int(wx.loc[i, "hour"])
    t        = wx.loc[i, "temperature_2m_avg"]
    cdh      = wx.loc[i, "CDH"]
    act_str  = f"{actual_by_hour[h]/1000:7.2f}" if h in actual_by_hour else "    n/a"
    svr_pred = reg_preds["SVR"][i]
    delta    = f"{(svr_pred - actual_by_hour[h])/1000:+6.2f}" if h in actual_by_hour else "    n/a"
    peak_ico = "PEAK" if gbc_peak[i] else "    "
    prob_pct = clf_probs["Gradient Boost"][i] * 100

    print(
        f"  {h:02d}:00  {t:7.1f}  {cdh:5.1f}  "
        f"{reg_preds['OLS'][i]/1000:7.2f}  "
        f"{reg_preds['Ridge'][i]/1000:7.2f}  "
        f"{reg_preds['Random Forest'][i]/1000:7.2f}  "
        f"{reg_preds['Gradient Boost'][i]/1000:7.2f}  "
        f"{svr_pred/1000:7.2f}  "
        f"{reg_preds['MLP (PyTorch)'][i]/1000:7.2f}  "
        f"  {act_str}  {delta}  {prob_pct:5.1f}%  {peak_ico}"
    )

    if h in actual_by_hour:
        act = actual_by_hour[h]
        for k, p in reg_preds.items():
            rmse_accum[k].append((p[i] - act) ** 2)
            mape_accum[k].append(abs(p[i] - act) / act)

print(f"  {'-'*105}")
print(f"  All load values in GW  |  Δ SVR = SVR prediction minus ERCOT actual\n")

# Live accuracy summary (only if we have actual data)
if any(rmse_accum[k] for k in rmse_accum):
    n_hours = len(rmse_accum["SVR"])
    print(f"  Live accuracy ({n_hours} completed hours):")
    print(f"  {'Model':<22}  {'RMSE (MW)':>10}  {'MAPE':>7}")
    print(f"  {'-'*45}")
    ranked = sorted(reg_preds.keys(),
                    key=lambda k: np.mean(rmse_accum[k]) if rmse_accum[k] else 1e9)
    for k in ranked:
        if rmse_accum[k]:
            rmse = np.sqrt(np.mean(rmse_accum[k]))
            mape = np.mean(mape_accum[k]) * 100
            print(f"  {k:<22}  {rmse:>10,.0f}  {mape:>6.1f}%")

print(f"\nFigures saved to {FIGURES}")
print("Live prediction complete.")
