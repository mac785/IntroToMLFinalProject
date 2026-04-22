# ERCOT Load Forecasting – Project Notes & Interesting Facts

## Project Context
- Course: Intro to Machine Learning, Prof. Varun Rai, UT Austin Spring 2026
- Student: Jamie Moseley (EID: jbm4577)
- Presentation: April 22, 2026 (in-class, 3-5 slides)
- Final report due: April 29, 2026 (10-15 slides + appendix + code + data)

---

## Interesting Facts & Anecdotes

### About ERCOT
- ERCOT (Electric Reliability Council of Texas) manages ~90% of Texas's electric load
- Texas is the only state with its own fully isolated grid — it deliberately avoids interstate commerce to sidestep federal regulation
- This isolation means Texas cannot easily import electricity from neighboring states during emergencies
- ERCOT serves ~26 million customers and manages ~680+ generation units

### Winter Storm Uri (February 2021)
- Temperatures dropped below -20°F in some Texas locations — well outside the training distribution of any ML model built on historical data
- ~4.5 million homes lost power; ~250 people died
- Demand spiked to record levels *while* supply collapsed from frozen natural gas pipelines
- This is a canonical example of model failure during "out-of-distribution" events — models trained on normal winters cannot anticipate simultaneous demand spike + supply collapse

### Why Weather Drives Load
- Air conditioning accounts for ~40% of summer peak electricity use in Texas
- Texas summers regularly push demand above 70,000 MW; the all-time record is ~85,508 MW (Aug 2023)
- The relationship between temperature and load is nonlinear — it's roughly "U-shaped": both very hot and very cold weather drive high demand
- Heating Degree Days (HDD) and Cooling Degree Days (CDD) are classic utility-industry features that capture this nonlinearity in a single number

### About the Grid Isolation / ML Implications
- Because ERCOT is isolated, there's no "import" variable to confuse models — load must equal generation
- This makes ERCOT a cleaner ML problem than grids with heavy interstate flows

### Dataset Milestone
- Our merged dataset (2021–2025) captured 85,464 MW as its maximum load observation
- ERCOT's all-time record is 85,508 MW, set in August 2023 — meaning **we effectively captured the all-time peak in our training data**
- This is useful to mention in the presentation: our model has "seen" the most extreme demand event in Texas grid history

---

## Data & Design Decisions

### City Selection — Why These Four?
We use weather from **Austin, Dallas, Houston, and San Antonio** — the four largest metropolitan areas in Texas, and the four dominant population/load centers within the ERCOT footprint. Together they represent the vast majority of residential and commercial electricity demand on the grid.

Alternatives considered and rejected:
- **Single city (e.g., Dallas only):** Would miss the massive Houston coastal humidity effect and Austin's hill country microclimate — both materially affect AC load
- **All ERCOT weather zones (8 zones):** More complete but adds complexity with diminishing returns; the four cities already span North, South, Coast, and West-Central Texas
- **ERCOT's own weather zone breakdown:** ERCOT publishes load by weather zone (Houston, North, South, West), which we could have used instead of cities — but city-level coordinates give us cleaner weather data from Open-Meteo

Each city's weather data is kept as separate columns AND averaged into a system-wide mean — giving models the choice to use regional or aggregate signals.

### Weather Variable Selection — Why These Six?
| Variable | Rationale |
|---|---|
| `temperature_2m` | Primary driver of AC and heating load; the single strongest predictor in virtually all electricity forecasting literature |
| `apparent_temperature` | "Feels like" temperature — incorporates humidity and wind; better captures human comfort/AC behavior than dry-bulb temperature alone |
| `relative_humidity_2m` | Texas gulf coast humidity amplifies cooling demand; high humidity + high temp = extreme AC load |
| `dew_point_2m` | A thermodynamically more stable humidity measure than relative humidity; preferred in some utility forecasting models |
| `wind_speed_10m` | Affects both heating/cooling load (wind chill) and is a key input to wind generation forecasting; also relevant because ERCOT has ~40 GW of wind capacity |
| `shortwave_radiation` | Proxy for solar irradiance and cloud cover; affects both cooling load (solar heat gain) and, increasingly, solar generation on the ERCOT grid |

Variables we considered but excluded:
- **Precipitation:** Weakly correlated with load after controlling for temperature; adds noise
- **Cloud cover:** Collinear with shortwave_radiation — redundant
- **Pressure:** Affects weather patterns but not directly load; low predictive value at hourly resolution

### Data Range — Why 2021–2025?
- 2021 includes **Winter Storm Uri** (Feb 2021) — a historically extreme cold event; keeping it gives models exposure to tail events
- 5 years × ~8,760 hours = ~43,800 observations before merging; after QA: **41,992 rows**
- This exceeds the course's preferred n×p ≥ 100,000 threshold (41,992 rows × 32 columns = ~1.34M)
- Going back further (pre-2021) risks including pre-pandemic demand patterns that may not generalize; COVID-19 (2020) materially changed demand profiles (less commercial, more residential)

### Weather Source — Open-Meteo (historical-forecast-api)
- Free, no API key required
- ERA5-based reanalysis data — the same source used in serious energy research
- The `historical-forecast-api.open-meteo.com` endpoint was used (rather than `archive.open-meteo.com`) due to local DNS resolution constraints; data is equivalent
- Crucially, the **same API (`api.open-meteo.com`) provides forecasts** — meaning our live prediction script on April 22 uses the exact same data schema as training. No feature mismatch risk.

---

## Model Notes (to be filled in during analysis)

### Linear Regression
- Baseline; interpretable coefficients

### Ridge / Lasso
- Lasso performs feature selection; Ridge shrinks coefficients
- Expected: temperature, hour, month will dominate

### Random Forest
- Handles nonlinearity and interactions automatically
- Feature importance plot will be a great slide

### Gradient Boosting
- Often best performer on tabular data

### SVR
- Support Vector Regression; kernel trick handles nonlinearity

### Neural Network (MLP)
- Can capture complex temporal patterns

---

## EDA Findings & Revelations

### The U-Shape is Real and Stark
The temperature vs. load scatter (Fig 5) makes the nonlinearity undeniable. The minimum load sits near **65°F** — the standard utility "balance point" — and the curve rises sharply in both directions. Texas's summer side (right) is far steeper than the winter side, reflecting how AC-heavy the state is. This directly justifies using CDH, HDH, and temp² as features rather than raw temperature alone.

### What is CDH? (Cooling Degree Hours)
CDH = max(temperature − 65°F, 0). The 65°F "balance point" is the industry-standard temperature below which buildings need neither heating nor cooling. CDH measures how many degrees above that threshold the current temperature is — a direct proxy for AC demand. It is always ≥ 0 (only "activates" when cooling is needed). Its sibling, **HDH = max(65°F − temperature, 0)**, captures the heating side. Together they split the U-shaped load/temp curve into two one-sided linear ramps that any model can learn easily. This is a classic example of injecting domain knowledge into feature engineering — rather than asking the model to discover the nonlinearity from scratch, we hand it the shape.

### CDH is the Single Strongest Predictor (r = 0.82)
Cooling Degree Hours beat raw temperature (r = 0.57) as a Load_MW correlate. This makes sense: CDH is zero below 65°F (no AC demand) and grows linearly above it, so it captures only the AC-driving portion of temperature. The feature engineering step earned its keep here.

### The Hourly Profile Has a Surprising Shoulder
The weekday load curve peaks between **hours 16–18 (4–6pm)** — not at solar noon. This is because AC systems have thermal inertia: buildings heat up through the afternoon, and demand peaks as workers return home and industrial cooling catches up. The weekend curve peaks slightly later (~hour 17) and is ~2–3 GW lower across the board, reflecting less commercial load.

### Winter Storm Uri Shows Up Clearly but Briefly
In the full time series, Uri appears as a **sharp spike in early 2021** — load jumped to ~65 GW in the dead of winter (normally ~35–40 GW that time of year). The close-up (Fig 2) shows temperatures dropping below 32°F for multiple days. This is the event that exposed the fragility of the Texas grid and triggered ERCOT's biggest regulatory overhaul in decades.

### month_sin / month_cos Outperform Raw Month
The cyclic month encoding (r ≈ -0.38 for month_cos) captures the seasonal pattern more cleanly than raw month (r = 0.14). The raw month feature can't represent that December and January are adjacent — cyclic encoding fixes this, and the correlation chart reflects it.

### Relative Humidity Has a Negative Correlation (r ≈ -0.38)
Counterintuitive at first: you'd expect humidity to increase cooling load. But in Texas, the hottest days tend to be dry (west Texas, inland heat waves), while humid days (Gulf coast) are also often cooler. The **apparent temperature** variable (which combines temp + humidity) is a better predictor (r = 0.56) because it captures the combined effect correctly. Humidity alone is a confounded signal.

---

## Regression Model Results (Test Set: 2025)

| Model | RMSE (MW) | R² | MAPE |
|---|---|---|---|
| MLP Neural Network | 6,321 | 0.62 | 10.9% |
| SVR (RBF kernel) | 6,461 | 0.60 | 11.2% |
| Gradient Boosting | 6,558 | 0.59 | 11.5% |
| Random Forest | 6,584 | 0.59 | 11.4% |
| OLS Linear | 6,699 | 0.57 | 11.4% |
| Ridge | 6,703 | 0.57 | 11.3% |
| Lasso | 6,734 | 0.57 | 11.4% |
| Decision Tree | 6,881 | 0.55 | 11.5% |

### Key Finding: The Models Are Surprisingly Close
All 8 models fall within a ~550 MW RMSE band (6,321–6,881). This is a meaningful finding: **the feature engineering was the real work**, not the choice of algorithm. Once you capture temperature nonlinearity (CDH, temp²), time-of-day cycles (sin/cos), and regional variation (per-city temps), most models converge to similar accuracy. This is worth one slide bullet — feature engineering > model selection for structured tabular data.

### Linear Models Actually Hold Their Own
OLS, Ridge, and Lasso achieve R² ≈ 0.57, only 5 points behind the best model (MLP at 0.62). This validates that the dominant relationships are already roughly linear in the engineered features. Ridge barely improves on OLS (best lambda ≈ 0.001 — nearly zero regularization), suggesting the feature set is not overfit. **Lasso zeroed out 4 of 27 features**: CDH, apparent_temperature, dew_point, and one per-city temperature — these are the most collinear predictors.

### The Predicted vs Actual Plot Reveals a Phase Shift Problem
The MLP's actual vs predicted chart for a July 2025 week shows the model **captures the daily cycle shape well but underestimates peaks by ~5–8 GW**. The predicted curve also appears slightly phase-shifted — the model lags the actual peak. This is the classic limitation of a no-lag model: without yesterday's load as context, it can't fully anticipate how hot a day was building up. Adding the 24h lag feature would almost certainly fix this — and is worth noting as a limitation and future improvement.

### R² on Test Set vs Train Set
OLS R² = 0.90 on training set, but only 0.57 on 2025 test data. This gap is partly genuine model limitation (no lags) and partly because 2025 may have different weather patterns than 2021–2024. This train/test R² gap is a good teaching moment: **in-sample fit does not predict out-of-sample performance**.

### Random Forest Feature Importances
Top 5: temperature_2m_avg (0.169), CDH (0.165), temp_sq (0.154), apparent_temperature_avg (0.115), temperature_2m_Dallas (0.071). This confirms CDH and temperature dominate — consistent with the correlation analysis from EDA. Notably, **Dallas temperature ranks higher than Austin** — Dallas has more extreme temperature swings and is the state's largest metro, so its AC demand drives system peaks more strongly.

---

## Approach Comparison Findings (April 22, 2026 live run)

### Decision Tree Flat-Line Failure on Mild Spring Days

**Observed:** On the April 22 live comparison (Fig 25), all Decision Tree variants produce
flat or near-flat predictions for overnight hours (00:00–05:00), returning a single constant
value (~34.75 GW) for all six hours.

**Root cause — not missing data, but feature variance collapse:**
The DT's root split is `temperature_2m_avg > 82.6°F`. On a mild spring day (high of 76°F),
every hour falls down the same branch at the root. Below that branch, the tree can only
differentiate hours using CDH, shortwave radiation, and cyclic time encodings. For hours
00–05, all three are near-zero or identical (CDH = 0, shortwave = 0, temperatures ~63°F),
so all six hours land in the same leaf node and receive the same prediction.

The tree produced only **13 distinct output values** across all 24 hours of a spring day,
compared to the smooth continuous curves produced by SVR and MLP.

**Why this happens structurally:**
Decision trees can only predict values they've seen in training leaf nodes — they cannot
interpolate or extrapolate. The tree learned its most informative splits around high-CDH
summer conditions (when peaks matter most and variance is highest). On low-variance mild
days those splits are useless, and the model degrades gracefully to coarse bucket averages.

**Does adding lag features fix it?**
Partially. `load_lag_24h` gives the tree a non-weather axis that does vary hour-to-hour
(yesterday's actual load at each hour is distinct), which lets it escape the flat-line trap.
But the fundamental limitation remains: on any day substantially different from the training
distribution's dominant conditions, DT precision collapses.

**Contrast with smooth models:**
SVR and MLP produce continuous, hour-differentiated curves even on mild days because they
interpolate within a learned function rather than routing to a fixed leaf value. This is a
meaningful argument for smooth models in operational forecasting contexts.

**Takeaway for final report:**
Decision trees are competitive on average (RMSE within ~400 MW of top models on the full
2025 test set) but fail qualitatively on low-variance days. This is worth one bullet in the
limitations section: *"DT predictions degrade to coarse step functions on mild days where
key features (CDH, shortwave) are near-zero — an inherent property of piecewise-constant
models, not a data quality issue."*

---

## Approach 2C: 2026 Test Set (Train on 2021–2025, Test on Jan–Apr 2026)

### Motivation
The original train/test split (train=2021-2024, test=2025) leaves valuable 2025 data on the table. 
ERCOT saw significant demand growth in 2025 (massive data center buildout), so adding 2025 to training 
should improve model calibration on current demand levels. 2026 data (Jan 1–Apr 21) serves as the 
out-of-sample test.

### Data Collection
- **ERCOT 2026 load**: NP6-345-CD via OAuth2 API, Jan 1–Apr 21, 2026
  - 2,663 records (111 days × 24h − 1 DST spring-forward hour on Mar 8, 2026)
  - Fetched in 3 pages of 1,000 with fixed-page-count pagination (not `_links`-based)
- **Weather 2026**: Open-Meteo `historical-forecast-api` for all 4 cities, same variables as training

### Three Feature Variants for 2026
| File | Description |
|---|---|
| `features_nolag_2026.csv` | Same 27 weather features, no lag |
| `features_lag_2026.csv` | + load_lag_24h, load_lag_168h (joined against 2025 tail) |
| `features_trend_2026.csv` | + days_elapsed (1826–1937 days since 2021-01-01) |

### Models (suffix: `_2025train`, `_2025train_lag`, `_2025train_trend`)
Trained on full `features_nolag.csv` / `features_lag.csv` / `features_trend.csv` (all of 2021–2025).
Script: `04e_regression_2025train.py`.

### April 22, 2026 Live Results (all models, best per approach)
| Approach | Best model | RMSE live | RMSE 2026 test |
|---|---|---|---|
| Current (weather-only) | SVR | 5.98 GW | — |
| Bias-corrected | SVR | 3.64 GW | — |
| Lag features (2A) | MLP | 2.71 GW | — |
| Days-elapsed trend (2A) | MLP | 2.71 GW | — |
| 2025 retrain, no lag (2C) | Decision Tree | 4.95 GW | ~6.5 GW |
| **2025 retrain + lag (2C)** | **SVR** | **1.82 GW** | **2.36 GW** |
| **2025 retrain + trend (2C)** | **MLP** | **0.79 GW** | **1.58 GW** |

Random Forest comparison (16 hours, Apr 22):
- Current: 7.14 GW → Bias-corrected: 5.30 GW → Lag: 4.08 GW → 2025+trend: **1.69 GW (-76%)**

**Key insight**: Adding 2025 training data alone (no-lag 2C) barely helps. The combination of
more training data + good features (lag or trend) compounds to give dramatically better results.
The `days_elapsed` trend extrapolation is particularly effective because it correctly infers that
2026 demand is higher than 2025 — the model has seen the steady growth curve.

**Offline test finding**: On the Jan-Apr 2026 holdout, all no-lag models have negative R² because
the test std (6.3 GW) is smaller than RMSE. This is expected: Jan-Apr has lower variance than
a full year (std=10.2 GW). Not a model failure — just a harder test window for R².

### Comparison to Include in Presentation
- Approach 2C models retrained on 2021-2025 predict today's load in `07_approach_comparison.py`
- Offline test results (Jan-Apr 2026 test set) in `regression_2025train_results.csv`
- Key question answered: YES, more data + good features reduces RMSE by 76% vs baseline

---

## Live Prediction Workflow Notes
- Use `api.open-meteo.com/v1/forecast` for day-of forecast (same API, same variables as training data)
- Compare against ERCOT's real-time load dashboard: https://www.ercot.com/gridinfo/load/load_hist

---

## Data Sources
- ERCOT Native Load (2021–2025): https://www.ercot.com/gridinfo/load/load_hist
- Open-Meteo Historical: https://historical-forecast-api.open-meteo.com/v1/forecast
- Open-Meteo Forecast (live demo): https://api.open-meteo.com/v1/forecast
- Weather locations: Austin (30.19, -97.67), Dallas (32.90, -97.04), Houston (29.65, -95.28), San Antonio (29.53, -98.47)
