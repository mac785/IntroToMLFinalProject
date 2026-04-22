"""
Phase 2b: Feature Engineering — Lag Features

Adds two load-history features to the existing feature set:
  load_lag_24h   — actual ERCOT load from exactly 24 hours prior
  load_lag_168h  — actual ERCOT load from exactly 168 hours (1 week) prior

Uses datetime-join (not row shift) to handle DST gaps cleanly.
Rows with no lag match (first 168 hours of the dataset) are dropped.

Output: data/processed/features_lag.csv
  Same schema as features_nolag.csv, plus the two lag columns.

Kept separate from 02_feature_engineering.py so features_nolag.csv
remains untouched and the two feature sets can be compared directly.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

print("=== Phase 2b: Lag Feature Engineering ===\n")

df = pd.read_csv(PROC / "features_nolag.csv", parse_dates=["HourEnding"])
df = df.sort_values("HourEnding").reset_index(drop=True)
print(f"Loaded features_nolag.csv: {len(df):,} rows")

load_ts = df[["HourEnding", "Load_MW"]].copy()

def add_lag(df, hours):
    """Left-join actual load from `hours` hours earlier onto each row."""
    col = f"load_lag_{hours}h"
    shifted = load_ts.copy()
    shifted["HourEnding"] = shifted["HourEnding"] + pd.Timedelta(hours=hours)
    shifted = shifted.rename(columns={"Load_MW": col})
    return df.merge(shifted, on="HourEnding", how="left"), col

df, col_24  = add_lag(df, 24)
df, col_168 = add_lag(df, 168)

before = len(df)
df = df.dropna(subset=[col_24, col_168]).reset_index(drop=True)
print(f"Dropped {before - len(df):,} rows with missing lags  →  {len(df):,} rows remain")

train = df[df["HourEnding"].dt.year < 2025]
test  = df[df["HourEnding"].dt.year == 2025]
print(f"Train: {len(train):,}  |  Test: {len(test):,}")
print(f"lag_24h  range: {df[col_24].min():,.0f} – {df[col_24].max():,.0f} MW")
print(f"lag_168h range: {df[col_168].min():,.0f} – {df[col_168].max():,.0f} MW")

out = PROC / "features_lag.csv"
df.to_csv(out, index=False)
print(f"\nSaved → {out}")
