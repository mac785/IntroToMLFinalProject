"""
Phase 2c: Feature Engineering — Days-Elapsed Trend Feature (Approach 2A)

Adds one continuous time-trend feature to the existing weather feature set:
  days_elapsed — fractional days since 2021-01-01 00:00

Why days_elapsed over year_idx:
  - Continuous rather than discrete (5 possible values vs ~44,000)
  - Tree models can split at any point in time, finding discrete load-growth
    events (data center clusters, etc.) rather than just year boundaries
  - Linear models get a smoother trend (intra-year growth, not just step jumps)

Output: data/processed/features_trend.csv
  Same schema as features_nolag.csv, plus days_elapsed column.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

print("=== Phase 2c: Trend Feature Engineering ===\n")

df = pd.read_csv(PROC / "features_nolag.csv", parse_dates=["HourEnding"])
df = df.sort_values("HourEnding").reset_index(drop=True)

ORIGIN = pd.Timestamp("2021-01-01")
df["days_elapsed"] = (df["HourEnding"] - ORIGIN).dt.total_seconds() / 86400

print(f"days_elapsed range: {df['days_elapsed'].min():.1f} – {df['days_elapsed'].max():.1f} days")
print(f"Rows: {len(df):,}")

out = PROC / "features_trend.csv"
df.to_csv(out, index=False)
print(f"Saved → {out}")
