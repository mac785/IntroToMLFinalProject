"""
Phase 1: Data Acquisition
Fetches ERCOT hourly native load and Open-Meteo weather data for Texas stations.

Data sources:
  - ERCOT Native Load: https://www.ercot.com/gridinfo/load/load_hist
  - Open-Meteo Historical Forecast API (free, no key): https://open-meteo.com/

Weather locations (lat/lon):
  Austin (KAUS):       30.1945, -97.6699
  Dallas/FW (KDFW):   32.8998, -97.0403
  Houston (KHOU):     29.6454, -95.2789
  San Antonio (KSAT): 29.5337, -98.4698
"""

import io
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── ERCOT: exact URLs scraped from ercot.com/gridinfo/load/load_hist ──────────
ERCOT_URLS = {
    2019: "https://www.ercot.com/files/docs/2020/01/09/Native_Load_2019.zip",
    2020: "https://www.ercot.com/files/docs/2021/01/12/Native_Load_2020.zip",
    2021: "https://www.ercot.com/files/docs/2021/11/12/Native_Load_2021.zip",
    2022: "https://www.ercot.com/files/docs/2022/02/08/Native_Load_2022.zip",
    2023: "https://www.ercot.com/files/docs/2023/02/09/Native_Load_2023.zip",
    2024: "https://www.ercot.com/files/docs/2024/02/06/Native_Load_2024.zip",
    2025: "https://www.ercot.com/files/docs/2025/02/11/Native_Load_2025.zip",
    2026: "https://www.ercot.com/files/docs/2026/02/10/Native_Load_2026.zip",
}
YEARS = [2021, 2022, 2023, 2024, 2025]   # target range

# ── Weather: four major Texas population/load centers ─────────────────────────
STATIONS = {
    "Austin":      (30.1945, -97.6699),
    "Dallas":      (32.8998, -97.0403),
    "Houston":     (29.6454, -95.2789),
    "SanAntonio":  (29.5337, -98.4698),
}

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "shortwave_radiation",    # solar proxy / cloud cover
    "apparent_temperature",   # "feels like" — humidity × temp interaction
]

# historical-forecast-api works where archive.open-meteo.com does not resolve
OMETO_HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OMETO_FCST_URL = "https://api.open-meteo.com/v1/forecast"


# ══════════════════════════════════════════════════════════════════════════════
# 1. ERCOT Native Load
# ══════════════════════════════════════════════════════════════════════════════

def download_ercot_year(year: int) -> pd.DataFrame | None:
    out_csv = RAW / f"ercot_load_{year}.csv"
    if out_csv.exists():
        print(f"  [cached] ERCOT {year}")
        return pd.read_csv(out_csv, parse_dates=["HourEnding"])

    url = ERCOT_URLS.get(year)
    if url is None:
        print(f"  [skip] No URL configured for ERCOT {year}")
        return None

    print(f"  Downloading ERCOT {year} ...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        fname = next(n for n in z.namelist() if n.endswith((".xls", ".xlsx", ".csv")))
        with z.open(fname) as f:
            df = pd.read_csv(f) if fname.endswith(".csv") else pd.read_excel(f)

    df = _clean_ercot(df, year)
    df.to_csv(out_csv, index=False)
    print(f"  [saved] ERCOT {year}: {len(df):,} rows")
    return df


def _clean_ercot(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df.columns = [c.strip().replace(" ", "_").upper() for c in df.columns]

    hour_col = next((c for c in df.columns if "HOUR" in c and "END" in c), None)
    if hour_col is None:
        raise ValueError(f"No HourEnding column found in ERCOT {year}. Columns: {df.columns.tolist()}")

    raw = df[hour_col].astype(str)
    df["HourEnding"] = pd.to_datetime(raw, errors="coerce")
    df["HourEnding"] = df["HourEnding"] - pd.Timedelta(hours=1)

    # Identify system-wide load column
    load_col = next(
        (c for c in df.columns if c in ("ERCOT", "SYSTEM_TOTAL", "TOTAL", "NATIVE_LOAD")),
        None,
    )
    if load_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        load_col = numeric_cols[-1]
        print(f"    [info] Using '{load_col}' as load column for {year}")

    out = df[["HourEnding", load_col]].copy()
    out.columns = ["HourEnding", "Load_MW"]
    out["Load_MW"] = pd.to_numeric(
        out["Load_MW"].astype(str).str.replace(",", ""), errors="coerce"
    )
    return out.dropna().sort_values("HourEnding").reset_index(drop=True)


def load_all_ercot() -> pd.DataFrame:
    print("\n=== Downloading ERCOT Native Load ===")
    frames = [df for yr in YEARS if (df := download_ercot_year(yr)) is not None]
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("HourEnding")
        .sort_values("HourEnding")
        .reset_index(drop=True)
    )
    print(f"ERCOT total: {len(combined):,} rows  ({combined.HourEnding.min()} → {combined.HourEnding.max()})")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 2. Weather via Open-Meteo
# ══════════════════════════════════════════════════════════════════════════════

def fetch_weather_station(name: str, lat: float, lon: float,
                           start: str, end: str) -> pd.DataFrame:
    tag = f"{start[:4]}_{end[:4]}"
    out_csv = RAW / f"weather_{name}_{tag}.csv"
    if out_csv.exists():
        print(f"  [cached] Weather {name}")
        return pd.read_csv(out_csv, parse_dates=["time"])

    print(f"  Fetching weather for {name} ({lat:.2f}, {lon:.2f}) {start} → {end} ...")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "America/Chicago",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
    }
    r = requests.get(OMETO_HIST_URL, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.insert(0, "station", name)
    df.to_csv(out_csv, index=False)
    print(f"  [saved] Weather {name}: {len(df):,} rows")
    return df


def load_all_weather(start: str, end: str) -> pd.DataFrame:
    print("\n=== Downloading Weather Data (Open-Meteo) ===")
    frames = [fetch_weather_station(n, lat, lon, start, end)
              for n, (lat, lon) in STATIONS.items()]
    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Merge & Save
# ══════════════════════════════════════════════════════════════════════════════

def merge_and_save(ercot: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Merging Datasets ===")

    pivot = weather.pivot_table(index="time", columns="station", values=WEATHER_VARS)
    pivot.columns = [f"{var}_{stn}" for var, stn in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"time": "HourEnding"})

    # Convenience averages across the four stations
    for var in WEATHER_VARS:
        cols = [c for c in pivot.columns if c.startswith(var + "_")]
        pivot[f"{var}_avg"] = pivot[cols].mean(axis=1)

    merged = ercot.merge(pivot, on="HourEnding", how="inner")

    before = len(merged)
    merged = merged[(merged["Load_MW"] > 20_000) & (merged["Load_MW"] < 110_000)]
    merged = merged.dropna(subset=["temperature_2m_avg"])
    print(f"  QA: dropped {before - len(merged)} rows; {len(merged):,} remain")

    out = PROCESSED / "ercot_weather_merged.csv"
    merged.to_csv(out, index=False)
    print(f"  [saved] → {out.name}  ({merged.columns.tolist()})")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# 4. Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start = f"{YEARS[0]}-01-01"
    end   = f"{YEARS[-1]}-12-31"

    ercot   = load_all_ercot()
    weather = load_all_weather(start, end)
    merged  = merge_and_save(ercot, weather)

    print("\n=== Dataset Summary ===")
    print(f"Shape: {merged.shape}")
    print(merged[["HourEnding", "Load_MW", "temperature_2m_avg"]].describe())
    print("\nPhase 1 complete. Run 02_feature_engineering.py next.")
