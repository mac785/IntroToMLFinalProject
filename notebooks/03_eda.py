"""
Phase 3: Exploratory Data Analysis
Produces publication-quality figures for the final presentation.

Figures saved to final_project/figures/:
  01_timeseries.png          - Full load time series 2021-2025
  02_uri_closeup.png         - Winter Storm Uri zoom (Feb 2021)
  03_hourly_profile.png      - Average load by hour (weekday vs weekend)
  04_monthly_profile.png     - Average load by month
  05_temp_vs_load.png        - Temperature vs load scatter (U-shape)
  06_correlation_heatmap.png - Correlation matrix of key features
  07_load_distribution.png   - Load distribution + peak threshold
  08_season_boxplots.png     - Load distribution by season
  09_feature_importance_preview.png - Correlation of each feature with Load_MW
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
PROC    = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE   = sns.color_palette("deep")
ERCOT_BLUE = "#003087"
PEAK_RED   = "#D62728"

def savefig(name: str):
    path = FIGURES / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {name}")


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data ...")
df = pd.read_csv(PROC / "features_full.csv", parse_dates=["HourEnding"])
df = df.sort_values("HourEnding").reset_index(drop=True)

PEAK_THRESHOLD = df["Load_MW"].quantile(0.90)
SEASON_LABELS  = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}
df["season_label"] = df["season"].map(SEASON_LABELS)

print(f"  Rows: {len(df):,}  |  Date range: {df.HourEnding.min().date()} → {df.HourEnding.max().date()}")
print(f"  Peak threshold (90th pct): {PEAK_THRESHOLD:,.0f} MW\n")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1: Full time series
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1: Full time series ...")
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df["HourEnding"], df["Load_MW"] / 1000, color=ERCOT_BLUE, lw=0.4, alpha=0.8)
ax.axhline(PEAK_THRESHOLD / 1000, color=PEAK_RED, lw=1.2, ls="--", label=f"90th pct peak ({PEAK_THRESHOLD/1000:.0f} GW)")
ax.set_ylabel("Load (GW)")
ax.set_title("ERCOT Hourly System Load  2021 – 2025")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(fontsize=10)

# Annotate Uri
uri_date = pd.Timestamp("2021-02-15")
ax.axvspan(pd.Timestamp("2021-02-10"), pd.Timestamp("2021-02-20"),
           alpha=0.15, color="steelblue", label="Winter Storm Uri")
ax.annotate("Winter Storm Uri\nFeb 2021", xy=(uri_date, 69), fontsize=8,
            ha="center", color="steelblue",
            arrowprops=dict(arrowstyle="->", color="steelblue"),
            xytext=(uri_date + pd.Timedelta(days=150), 73))

# Annotate all-time peak
peak_row = df.loc[df["Load_MW"].idxmax()]
ax.annotate(f"All-time peak\n{peak_row.Load_MW/1000:.1f} GW\n{peak_row.HourEnding.strftime('%b %Y')}",
            xy=(peak_row.HourEnding, peak_row.Load_MW / 1000),
            xytext=(peak_row.HourEnding - pd.Timedelta(days=300), 82),
            fontsize=8, color=PEAK_RED,
            arrowprops=dict(arrowstyle="->", color=PEAK_RED))
savefig("01_timeseries.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2: Winter Storm Uri close-up
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: Winter Storm Uri close-up ...")
uri = df[(df["HourEnding"] >= "2021-02-05") & (df["HourEnding"] <= "2021-02-25")].copy()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax1.plot(uri["HourEnding"], uri["Load_MW"] / 1000, color=ERCOT_BLUE, lw=1.5)
ax1.set_ylabel("Load (GW)")
ax1.set_title("Winter Storm Uri — ERCOT Load & Temperature  (Feb 2021)")
ax1.axvspan(pd.Timestamp("2021-02-10"), pd.Timestamp("2021-02-20"),
            alpha=0.12, color="steelblue")

ax2.plot(uri["HourEnding"], uri["temperature_2m_avg"], color="tomato", lw=1.5)
ax2.axhline(32, color="navy", lw=1, ls="--", alpha=0.6, label="Freezing (32°F)")
ax2.set_ylabel("Avg Temp (°F)")
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

fig.tight_layout()
savefig("02_uri_closeup.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3: Average load profile by hour (weekday vs weekend)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: Hourly load profile ...")
hourly = df.groupby(["hour", "is_weekend"])["Load_MW"].mean().reset_index()
hourly["is_weekend"] = hourly["is_weekend"].map({0: "Weekday", 1: "Weekend"})

fig, ax = plt.subplots(figsize=(10, 4))
for label, grp in hourly.groupby("is_weekend"):
    ax.plot(grp["hour"], grp["Load_MW"] / 1000, marker="o", ms=4,
            label=label, lw=2)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average Load (GW)")
ax.set_title("Average ERCOT Load by Hour of Day  (2021–2025)")
ax.set_xticks(range(0, 24, 2))
ax.legend()
savefig("03_hourly_profile.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4: Average load by month
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4: Monthly profile ...")
monthly = df.groupby("month")["Load_MW"].mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(month_names, monthly.values / 1000, color=ERCOT_BLUE, alpha=0.8, edgecolor="white")
ax.bar(month_names, monthly.values / 1000,
       color=[PEAK_RED if m in [7, 8] else ERCOT_BLUE for m in range(1, 13)],
       alpha=0.85, edgecolor="white")
ax.set_ylabel("Average Load (GW)")
ax.set_title("Average ERCOT Load by Month  (2021–2025)")
ax.set_ylim(0, monthly.max() / 1000 * 1.15)
for i, v in enumerate(monthly.values):
    ax.text(i, v / 1000 + 0.3, f"{v/1000:.1f}", ha="center", fontsize=9)
savefig("04_monthly_profile.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5: Temperature vs Load scatter (the U-shape)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 5: Temperature vs Load (U-shape) ...")
sample = df.sample(n=8000, random_state=42)

fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(sample["temperature_2m_avg"], sample["Load_MW"] / 1000,
                c=sample["hour"], cmap="RdYlBu_r", alpha=0.35, s=8, rasterized=True)
cbar = fig.colorbar(sc, ax=ax, label="Hour of Day")

# Fit and overlay a quadratic trend line
z = np.polyfit(df["temperature_2m_avg"], df["Load_MW"] / 1000, 2)
p = np.poly1d(z)
xline = np.linspace(df["temperature_2m_avg"].min(), df["temperature_2m_avg"].max(), 200)
ax.plot(xline, p(xline), color="black", lw=2.5, label="Quadratic trend")

ax.axvline(65, color="gray", lw=1, ls="--", alpha=0.7, label="65°F balance point")
ax.set_xlabel("Average Texas Temperature (°F)")
ax.set_ylabel("Load (GW)")
ax.set_title("Temperature vs. ERCOT Load  — the U-Shaped Relationship")
ax.legend(fontsize=9)
savefig("05_temp_vs_load.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6: Correlation heatmap (key features only)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 6: Correlation heatmap ...")
heat_cols = [
    "Load_MW", "temperature_2m_avg", "apparent_temperature_avg",
    "CDH", "HDH", "temp_sq", "relative_humidity_2m_avg",
    "shortwave_radiation_avg", "wind_speed_10m_avg",
    "hour", "hour_sin", "month", "is_weekend", "is_holiday", "is_workday",
]
corr = df[heat_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.4, annot_kws={"size": 8})
ax.set_title("Correlation Matrix — Key Features vs Load")
fig.tight_layout()
savefig("06_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7: Load distribution + peak threshold
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 7: Load distribution ...")
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(df["Load_MW"] / 1000, bins=80, color=ERCOT_BLUE, alpha=0.75, edgecolor="white")
ax.axvline(PEAK_THRESHOLD / 1000, color=PEAK_RED, lw=2, ls="--",
           label=f"Peak threshold: {PEAK_THRESHOLD/1000:.1f} GW  (90th pct)")
ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 3000],
                 PEAK_THRESHOLD / 1000, df["Load_MW"].max() / 1000,
                 alpha=0.12, color=PEAK_RED)
ax.set_xlabel("Load (GW)")
ax.set_ylabel("Count (hours)")
ax.set_title("Distribution of Hourly ERCOT Load  (2021–2025)")
ax.legend()
savefig("07_load_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8: Season box plots
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 8: Season boxplots ...")
season_order = ["Winter", "Spring", "Summer", "Fall"]
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="season_label", y="Load_MW", hue="season_label",
            order=season_order, legend=False,
            palette=["steelblue", "mediumseagreen", PEAK_RED, "darkorange"],
            ax=ax, width=0.5, flierprops=dict(marker=".", alpha=0.2, ms=3))
ax.set_xlabel("")
ax.set_ylabel("Load (MW)")
ax.set_title("ERCOT Load Distribution by Season  (2021–2025)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f} GW"))
savefig("08_season_boxplots.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9: Feature correlation with Load_MW (horizontal bar — presentation-ready)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 9: Feature correlations with Load_MW ...")
nolag_cols = [
    "temperature_2m_avg", "apparent_temperature_avg", "CDH", "HDH",
    "temp_sq", "relative_humidity_2m_avg", "dew_point_2m_avg",
    "shortwave_radiation_avg", "wind_speed_10m_avg",
    "hour_sin", "hour_cos", "hour", "month_sin", "month_cos", "month",
    "dow_sin", "is_weekend", "is_workday", "is_holiday", "season",
    "temp_x_hour_sin", "CDH_x_hour_sin", "temp_x_is_weekend",
    "apparent_temp_delta_avg",
]
corrs = df[nolag_cols + ["Load_MW"]].corr()["Load_MW"].drop("Load_MW").sort_values()

colors = [PEAK_RED if v > 0 else "steelblue" for v in corrs.values]
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(corrs.index, corrs.values, color=colors, alpha=0.8, edgecolor="white")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Pearson Correlation with Load_MW")
ax.set_title("Feature Correlations with ERCOT Load")
fig.tight_layout()
savefig("09_feature_correlations.png")


# ══════════════════════════════════════════════════════════════════════════════
# Print key EDA statistics
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Key EDA Statistics ===")
print(f"Date range:           {df.HourEnding.min().date()} → {df.HourEnding.max().date()}")
print(f"Total hours:          {len(df):,}")
print(f"Load mean:            {df.Load_MW.mean():,.0f} MW")
print(f"Load std:             {df.Load_MW.std():,.0f} MW")
print(f"Load min:             {df.Load_MW.min():,.0f} MW")
print(f"Load max:             {df.Load_MW.max():,.0f} MW  ({df.loc[df.Load_MW.idxmax(), 'HourEnding'].strftime('%b %Y')})")
print(f"Peak threshold (90%): {PEAK_THRESHOLD:,.0f} MW")
print(f"Temp range:           {df.temperature_2m_avg.min():.1f}°F → {df.temperature_2m_avg.max():.1f}°F")
print(f"\nTop correlates with Load_MW:")
print(corrs.abs().sort_values(ascending=False).head(8).to_string())

print(f"\nAll figures saved to: {FIGURES}")
print("Phase 3 complete. Run 04_models.py next.")
