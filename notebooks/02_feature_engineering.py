"""
Phase 2: Feature Engineering
Transforms the merged ERCOT + weather dataset into a model-ready feature matrix.

Features created:
  Time:        hour, day_of_week, month, season, is_weekend, is_holiday
  Weather:     CDD, HDD, temp^2 (nonlinearity), per-city + avg columns (already present)
  Interactions: temp * hour_sin, temp * is_weekend
  Lag:         load_lag_1h, load_lag_24h, load_lag_168h (1 week), rolling_24h_mean
  Target (regression):      Load_MW  (continuous)
  Target (classification):  is_peak  (top 10% of Load_MW = 1, else 0)
"""

import pandas as pd
import numpy as np
from pathlib import Path

try:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    HAS_HOLIDAY = True
except ImportError:
    HAS_HOLIDAY = False

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

BALANCE_TEMP_F = 65.0   # industry standard base for HDD/CDD calculation


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _season(month: pd.Series) -> pd.Series:
    """Map month → season integer (0=Winter, 1=Spring, 2=Summer, 3=Fall)."""
    return pd.cut(
        month,
        bins=[0, 2, 5, 8, 11, 12],
        labels=[0, 1, 2, 3, 0],   # Dec wraps to Winter
        ordered=False,
    ).astype(int)


def _cyclic_encode(series: pd.Series, period: int):
    """Sin/cos encode a cyclic feature (e.g. hour 0-23, month 1-12)."""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


# ══════════════════════════════════════════════════════════════════════════════
# Main engineering function
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HourEnding"] = pd.to_datetime(df["HourEnding"])
    df = df.sort_values("HourEnding").reset_index(drop=True)

    ts = df["HourEnding"]

    # ── Time features ──────────────────────────────────────────────────────────
    df["hour"]        = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek          # 0=Mon … 6=Sun
    df["month"]       = ts.dt.month
    df["year"]        = ts.dt.year
    df["day_of_year"] = ts.dt.dayofyear
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["season"]      = _season(df["month"])

    # Cyclic encodings — avoids the false discontinuity at hour 23→0, Dec→Jan
    df["hour_sin"],    df["hour_cos"]    = _cyclic_encode(df["hour"],        24)
    df["month_sin"],   df["month_cos"]   = _cyclic_encode(df["month"],       12)
    df["dow_sin"],     df["dow_cos"]     = _cyclic_encode(df["day_of_week"],  7)

    # US Federal holidays
    if HAS_HOLIDAY:
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=ts.min(), end=ts.max())
        df["is_holiday"] = ts.dt.normalize().isin(holidays).astype(int)
    else:
        df["is_holiday"] = 0

    # is_workday: not weekend AND not holiday
    df["is_workday"] = ((df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)

    # ── Weather-derived features ────────────────────────────────────────────────
    temp = df["temperature_2m_avg"]

    # Cooling / Heating Degree (Hours) relative to 65°F balance point
    df["CDH"] = (temp - BALANCE_TEMP_F).clip(lower=0)   # cooling demand proxy
    df["HDH"] = (BALANCE_TEMP_F - temp).clip(lower=0)   # heating demand proxy

    # Temperature squared — captures the U-shaped load/temp curve directly
    df["temp_sq"] = temp ** 2

    # Apparent temperature delta: how much "feels like" diverges from actual
    df["apparent_temp_delta_avg"] = df["apparent_temperature_avg"] - temp

    # ── Interaction features ────────────────────────────────────────────────────
    # Temp × time-of-day: hot afternoons (high temp, high hour_sin) drive peak
    df["temp_x_hour_sin"]   = temp * df["hour_sin"]
    df["temp_x_is_weekend"] = temp * df["is_weekend"]
    df["CDH_x_hour_sin"]    = df["CDH"] * df["hour_sin"]

    # ── Lag / rolling features ─────────────────────────────────────────────────
    # NOTE: lags are derived from the target (Load_MW) — they must be excluded
    # from the live-prediction feature set and used only in retrospective eval.
    df["load_lag_1h"]   = df["Load_MW"].shift(1)
    df["load_lag_24h"]  = df["Load_MW"].shift(24)
    df["load_lag_168h"] = df["Load_MW"].shift(168)   # same hour, one week prior
    df["load_roll_24h"] = df["Load_MW"].shift(1).rolling(24).mean()

    # ── Classification target ──────────────────────────────────────────────────
    peak_threshold = df["Load_MW"].quantile(0.90)
    df["is_peak"]   = (df["Load_MW"] >= peak_threshold).astype(int)
    print(f"  Peak threshold (90th pct): {peak_threshold:,.0f} MW")
    print(f"  Peak hours: {df['is_peak'].sum():,} of {len(df):,} ({df['is_peak'].mean()*100:.1f}%)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Save two versions: with lags (train/eval) and without (live prediction)
# ══════════════════════════════════════════════════════════════════════════════

LAG_COLS = ["load_lag_1h", "load_lag_24h", "load_lag_168h", "load_roll_24h"]

def save_datasets(df: pd.DataFrame):
    # Full dataset (includes lags — NaN for first 168 rows)
    full_out = PROCESSED / "features_full.csv"
    df.to_csv(full_out, index=False)
    print(f"  [saved] Full feature set → {full_out.name}  shape={df.shape}")

    # No-lag version: used for live prediction and as lag-free model comparison
    nolag_cols = [c for c in df.columns if c not in LAG_COLS]
    nolag_out  = PROCESSED / "features_nolag.csv"
    df[nolag_cols].to_csv(nolag_out, index=False)
    print(f"  [saved] No-lag feature set → {nolag_out.name}  shape={df[nolag_cols].shape}")

    # Clean subset: drop the 168 NaN rows introduced by the longest lag
    clean = df.dropna(subset=LAG_COLS)
    clean_out = PROCESSED / "features_clean.csv"
    clean.to_csv(clean_out, index=False)
    print(f"  [saved] Clean (no NaN) → {clean_out.name}  shape={clean.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Phase 2: Feature Engineering ===\n")

    merged = pd.read_csv(PROCESSED / "ercot_weather_merged.csv", parse_dates=["HourEnding"])
    print(f"Loaded merged dataset: {merged.shape}")

    df = engineer_features(merged)

    print("\n--- Feature summary ---")
    feature_cols = [c for c in df.columns if c not in ("HourEnding", "Load_MW", "is_peak")]
    print(f"Total features: {len(feature_cols)}")

    groups = {
        "Time (raw)":       [c for c in feature_cols if c in ("hour","day_of_week","month","year","day_of_year","season","is_weekend","is_holiday","is_workday")],
        "Time (cyclic)":    [c for c in feature_cols if "_sin" in c or "_cos" in c],
        "Weather (raw)":    [c for c in feature_cols if any(c.startswith(v) for v in ("temperature","apparent","relative_humidity","dew_point","wind_speed","shortwave"))],
        "Weather (derived)":["CDH","HDH","temp_sq","apparent_temp_delta_avg"],
        "Interactions":     [c for c in feature_cols if "_x_" in c],
        "Lags":             LAG_COLS,
    }
    for grp, cols in groups.items():
        print(f"  {grp}: {cols}")

    print("\n--- Missing values (lag NaNs expected) ---")
    nans = df.isnull().sum()
    print(nans[nans > 0])

    save_datasets(df)

    print("\n--- Load_MW distribution ---")
    print(df["Load_MW"].describe())

    print("\n--- is_peak class balance ---")
    print(df["is_peak"].value_counts())

    print("\nPhase 2 complete. Run 03_eda.py next.")
