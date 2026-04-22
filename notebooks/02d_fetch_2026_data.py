"""
Phase 2d: Fetch 2026 ERCOT + Weather Data and Engineer Features (Approach 2C)

Collects Jan 1 – Apr 21, 2026 data from:
  - ERCOT NP6-345-CD (hourly actual system load, OAuth2 authenticated)
  - Open-Meteo historical-forecast-api (same 4 cities, same variables as training)

Outputs (data/processed/):
  features_nolag_2026.csv  — 27-feature schema identical to features_nolag.csv
  features_lag_2026.csv    — adds load_lag_24h + load_lag_168h (joined against 2025 tail)
  features_trend_2026.csv  — adds days_elapsed (fractional days since 2021-01-01)
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

try:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    HAS_HOLIDAY = True
except ImportError:
    HAS_HOLIDAY = False

ROOT  = Path(__file__).resolve().parent.parent
PROC  = ROOT / "data" / "processed"
RAW   = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BALANCE_TEMP_F = 65.0
ORIGIN         = pd.Timestamp("2021-01-01")

START_DATE = "2026-01-01"
END_DATE   = "2026-04-21"

STATIONS = {
    "Austin":     (30.1945, -97.6699),
    "Dallas":     (32.8998, -97.0403),
    "Houston":    (29.6454, -95.2789),
    "SanAntonio": (29.5337, -98.4698),
}

WEATHER_VARS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "dew_point_2m", "wind_speed_10m", "shortwave_radiation",
]

ERCOT_CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
ERCOT_B2C_URL   = ("https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com"
                   "/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token")
ERCOT_API_BASE  = "https://api.ercot.com/api/public-reports"

# ══════════════════════════════════════════════════════════════════════════════
# ERCOT Auth + Fetch
# ══════════════════════════════════════════════════════════════════════════════

def _load_creds():
    load_dotenv(ROOT / ".env")
    return (os.environ["ERCOT_USERNAME"],
            os.environ["ERCOT_PASSWORD"],
            os.environ["ERCOT_PRIMARY_KEY"])


def _ercot_token(username, password, sub_key):
    r = requests.post(ERCOT_B2C_URL, timeout=20,
                      headers={"Ocp-Apim-Subscription-Key": sub_key},
                      data={"grant_type": "password",
                            "client_id": ERCOT_CLIENT_ID,
                            "scope": f"openid {ERCOT_CLIENT_ID} offline_access",
                            "response_type": "id_token",
                            "username": username, "password": password})
    if r.status_code == 200:
        return r.json().get("id_token")
    raise RuntimeError(f"ERCOT token failed {r.status_code}: {r.text[:200]}")


def _parse_hourending(op_day: str, hour_end: str) -> pd.Timestamp:
    """Convert ERCOT operatingDay + hourEnding (01:00–24:00) to start-of-hour timestamp."""
    # ERCOT uses 24:00 for the last hour; replace with next day 00:00 before subtracting
    if hour_end.startswith("24"):
        ts = pd.Timestamp(op_day) + pd.Timedelta(days=1)
    else:
        ts = pd.Timestamp(f"{op_day} {hour_end}")
    return ts - pd.Timedelta(hours=1)


def fetch_ercot_2026(username, password, sub_key) -> pd.DataFrame:
    """Fetch NP6-345-CD for START_DATE–END_DATE with pagination."""
    cache = RAW / "ercot_np6345_2026.csv"
    if cache.exists():
        print(f"  [cached] {cache}")
        return pd.read_csv(cache, parse_dates=["HourEnding"])

    token = _ercot_token(username, password, sub_key)
    hdrs  = {"Authorization": f"Bearer {token}",
             "Ocp-Apim-Subscription-Key": sub_key}

    url    = f"{ERCOT_API_BASE}/np6-345-cd/act_sys_load_by_wzn"
    rows   = []
    page   = 1
    total  = None

    PAGE_SIZE = 1000
    while True:
        r = requests.get(url, headers=hdrs, params={
            "operatingDayFrom": START_DATE,
            "operatingDayTo":   END_DATE,
            "page": page, "size": PAGE_SIZE,
        }, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"NP6-345-CD page {page} failed: {r.status_code} {r.text[:200]}")

        payload = r.json()
        meta    = payload.get("_meta", {})
        if total is None:
            total = meta.get("totalRecords", "?")
            print(f"  NP6-345-CD totalRecords: {total}")

        # data is list-of-lists; fields order: operatingDay, hourEnding, coast, east,
        # farWest, north, northC, southern, southC, west, total, DSTFlag
        batch = payload.get("data", [])
        rows.extend(batch)
        print(f"  Page {page}: {len(batch)} rows  (cumulative: {len(rows)})")

        # Stop when last page returns fewer rows than page size
        if len(batch) < PAGE_SIZE:
            break
        page += 1
        time.sleep(0.3)

    df = pd.DataFrame(rows, columns=[
        "operatingDay", "hourEnding", "coast", "east", "farWest", "north",
        "northC", "southern", "southC", "west", "total", "DSTFlag"
    ])
    df["HourEnding"] = df.apply(
        lambda r: _parse_hourending(r["operatingDay"], r["hourEnding"]), axis=1)
    df = (df[["HourEnding", "total"]]
          .rename(columns={"total": "Load_MW"})
          .sort_values("HourEnding")
          .drop_duplicates("HourEnding")
          .reset_index(drop=True))

    print(f"  ERCOT 2026: {len(df)} rows  "
          f"({df.HourEnding.min()} → {df.HourEnding.max()})")
    df.to_csv(cache, index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Weather Fetch
# ══════════════════════════════════════════════════════════════════════════════

def fetch_weather_2026() -> pd.DataFrame:
    """Fetch Open-Meteo historical for all 4 cities, Jan 1–Apr 21, 2026."""
    cache = RAW / "weather_2026.csv"
    if cache.exists():
        print(f"  [cached] {cache}")
        return pd.read_csv(cache, parse_dates=["HourEnding"])

    url    = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    frames = []
    for city, (lat, lon) in STATIONS.items():
        r = requests.get(url, params={
            "latitude": lat, "longitude": lon,
            "start_date": START_DATE, "end_date": END_DATE,
            "hourly": ",".join(WEATHER_VARS),
            "wind_speed_unit": "mph",
            "temperature_unit": "fahrenheit",
            "timezone": "America/Chicago",
        }, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Open-Meteo {city} failed: {r.status_code} {r.text[:200]}")
        h = r.json()["hourly"]
        df = pd.DataFrame({"HourEnding": pd.to_datetime(h["time"])})
        for v in WEATHER_VARS:
            df[f"{v}_{city}"] = h[v]
        frames.append(df)
        print(f"  Weather {city}: {len(df)} hours")
        time.sleep(0.2)

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="HourEnding", how="inner")

    merged.to_csv(cache, index=False)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering (mirrors 02_feature_engineering.py)
# ══════════════════════════════════════════════════════════════════════════════

def _season(month):
    return pd.cut(month, bins=[0,2,5,8,11,12],
                  labels=[0,1,2,3,0], ordered=False).astype(int)

def _cyclic(series, period):
    a = 2 * np.pi * series / period
    return np.sin(a), np.cos(a)

def engineer_features_2026(load_df, weather_df) -> pd.DataFrame:
    df = load_df.merge(weather_df, on="HourEnding", how="inner")
    df = df.sort_values("HourEnding").reset_index(drop=True)

    # Per-city averages
    for v in WEATHER_VARS:
        cols = [f"{v}_{city}" for city in STATIONS]
        df[f"{v}_avg"] = df[cols].mean(axis=1)

    # Rename per-city columns to match training schema
    df = df.rename(columns={f"{v}_{city}": f"{v}_{city}"
                             for v in WEATHER_VARS for city in STATIONS})

    ts = df["HourEnding"]
    df["hour"]        = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"]       = ts.dt.month
    df["year"]        = ts.dt.year
    df["day_of_year"] = ts.dt.dayofyear
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["season"]      = _season(df["month"])

    df["hour_sin"], df["hour_cos"]   = _cyclic(df["hour"],        24)
    df["month_sin"], df["month_cos"] = _cyclic(df["month"],       12)
    df["dow_sin"],  df["dow_cos"]    = _cyclic(df["day_of_week"],  7)

    if HAS_HOLIDAY:
        cal      = USFederalHolidayCalendar()
        holidays = cal.holidays(start=ts.min(), end=ts.max())
        df["is_holiday"] = ts.dt.normalize().isin(holidays).astype(int)
    else:
        df["is_holiday"] = 0

    df["is_workday"] = ((df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)

    temp = df["temperature_2m_avg"]
    df["CDH"]  = (temp - BALANCE_TEMP_F).clip(lower=0)
    df["HDH"]  = (BALANCE_TEMP_F - temp).clip(lower=0)
    df["temp_sq"] = temp ** 2
    df["apparent_temp_delta_avg"] = df["apparent_temperature_avg"] - temp
    df["temp_x_hour_sin"]   = temp * df["hour_sin"]
    df["temp_x_is_weekend"] = temp * df["is_weekend"]
    df["CDH_x_hour_sin"]    = df["CDH"] * df["hour_sin"]

    peak_threshold = df["Load_MW"].quantile(0.90)
    df["is_peak"]  = (df["Load_MW"] >= peak_threshold).astype(int)
    print(f"  Peak threshold (90th pct): {peak_threshold:,.0f} MW")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Lag feature computation (joined, not shifted, to handle time series gaps)
# ══════════════════════════════════════════════════════════════════════════════

def add_lag_features(df_2026: pd.DataFrame) -> pd.DataFrame:
    """
    Compute load_lag_24h and load_lag_168h by datetime-join against a combined
    2025+2026 load series. Using join (not shift) avoids DST-gap misalignment.
    """
    df_hist = pd.read_csv(PROC / "features_nolag.csv",
                          parse_dates=["HourEnding"],
                          usecols=["HourEnding", "Load_MW"])

    combined = (pd.concat([df_hist[["HourEnding", "Load_MW"]],
                           df_2026[["HourEnding", "Load_MW"]]], ignore_index=True)
                .sort_values("HourEnding")
                .drop_duplicates("HourEnding")
                .reset_index(drop=True))

    df = df_2026.copy()
    lag_df = combined.set_index("HourEnding")["Load_MW"]

    for lag_h, col in [(24, "load_lag_24h"), (168, "load_lag_168h")]:
        target_times = df["HourEnding"] - pd.Timedelta(hours=lag_h)
        df[col] = lag_df.reindex(target_times).values

    before = len(df)
    df = df.dropna(subset=["load_lag_24h", "load_lag_168h"]).reset_index(drop=True)
    print(f"  Lag features: dropped {before - len(df)} NaN rows → {len(df)} rows")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

print("=== Phase 2d: Fetch & Engineer 2026 Data (Approach 2C) ===\n")

# Credentials
username, password, sub_key = _load_creds()

# Fetch raw data
print("--- Fetching ERCOT 2026 load ---")
load_df = fetch_ercot_2026(username, password, sub_key)

print("\n--- Fetching Open-Meteo 2026 weather ---")
weather_df = fetch_weather_2026()

# Engineer base features
print("\n--- Engineering features ---")
df = engineer_features_2026(load_df, weather_df)
print(f"  Merged rows: {len(df)}")
print(f"  Date range: {df.HourEnding.min()} → {df.HourEnding.max()}")

# ── Save no-lag variant ────────────────────────────────────────────────────────
out_nolag = PROC / "features_nolag_2026.csv"
df.to_csv(out_nolag, index=False)
print(f"\nSaved → {out_nolag}  ({len(df)} rows, {len(df.columns)} cols)")

# ── Save lag variant ───────────────────────────────────────────────────────────
print("\n--- Adding lag features ---")
df_lag = add_lag_features(df)
out_lag = PROC / "features_lag_2026.csv"
df_lag.to_csv(out_lag, index=False)
print(f"Saved → {out_lag}  ({len(df_lag)} rows)")

# ── Save trend variant ─────────────────────────────────────────────────────────
df_trend = df.copy()
df_trend["days_elapsed"] = (df_trend["HourEnding"] - ORIGIN).dt.total_seconds() / 86400
print(f"\ndays_elapsed range: {df_trend['days_elapsed'].min():.1f} – "
      f"{df_trend['days_elapsed'].max():.1f} days")
out_trend = PROC / "features_trend_2026.csv"
df_trend.to_csv(out_trend, index=False)
print(f"Saved → {out_trend}  ({len(df_trend)} rows)")

print("\nPhase 2d complete.")
