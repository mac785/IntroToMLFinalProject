# ERCOT Final Report — Review Checklist

This document is a self-contained worklist for finishing Jamie Moseley's
Intro-to-ML final-report slide deck. Items are ordered by importance.
Numeric/factual issues first, then structural improvements, then polish.

Final report is due **2026-04-29 11:59 PM**.

---

## 0. Project map (where things live)

| Path | Purpose |
|---|---|
| `final_project/report_outline.md` | The slide deck (Marp markdown). **This is the final deliverable.** |
| `final_project/notebooks/` | All Python scripts that generate data, train models, build figures |
| `final_project/data/processed/` | Train/test feature CSVs and result CSVs |
| `final_project/figures/` | Live/2026 figures (23–26) referenced by the deck |
| `final_project/figures_archive_prefix/` | Older figures (01–22) referenced by the deck |
| `final_project/models/` | Saved sklearn `.pkl` and PyTorch `.pt` model files |
| `final_project/reference_files/` | Syllabus, project guidance, proposal, 3 reference student decks |

**Key result CSVs to verify any number against:**

- `data/processed/regression_results.csv` — baseline 8 models trained 2021–2024, tested on **2025**
- `data/processed/regression_lag_results.csv` — baseline + lag features
- `data/processed/regression_trend_results.csv` — baseline + days_elapsed
- `data/processed/regression_2025train_results.csv` — 2025 retrain + variants (the most important one — covers 2025train_nolag / lag / trend / lag_trend)
- `data/processed/classification_results.csv` — peak-classification metrics

**Scripts that produce live/2026 numbers (re-run before submitting):**

- `notebooks/07_approach_comparison.py` — generates figures 23, 24, 25 (live April-29 evaluation across 8 approaches)
- `notebooks/08_2026_rmse_figure.py` — generates figure 26 (offline Jan–Apr 2026 holdout RMSE bar chart)

To rebuild figures: `cd final_project && python notebooks/07_approach_comparison.py`
and `python notebooks/08_2026_rmse_figure.py`. Both must complete without errors.

---

## 1. CRITICAL — fix factual errors in the deck

These are wrong as written and need correcting before submission.

### 1a. "2,663 hours" → real holdout size

**Where:** `report_outline.md` slide 17 (Best Approach), in the line:
```
**Offline test results (Jan–Apr 2026 holdout, 2,663 hours):**
```

**Why it's wrong:** The 2026 feature CSVs actually have:
- `features_nolag_2026.csv`: **2,831 rows** (Jan 1 → Apr 28)
- `features_lag_2026.csv`: 2,822 rows
- `features_trend_2026.csv`: 2,831 rows
- `features_lag_trend_2026.csv`: 2,654 rows (this file ends Apr 21 — was generated earlier and not refreshed)

The "2,663" doesn't match any actual file.

**How to fix:** Either
- (a) Replace with `2,831 hours (Jan 1–Apr 28, 2026)` if reporting the no-lag/trend variant test set, or
- (b) Replace with `~2,650–2,830 hours (118 days, varies slightly by feature variant)` if you want one number for all 8 approaches.
- Recommended: option (a). The headline result (SVR 1,378 MW) comes from the 2,654-row lag+trend file — but that's a holdout-window quirk not worth detailing on the main slide.

**Verify after fix:** open the slide in a Marp preview and confirm the line reads correctly.

### 1b. "7,960 MW" baseline → wrong number

**Where:** `report_outline.md` slide 17 and slide 22 (Conclusions), the lines:
```
**83% RMSE reduction** vs. weather-only baseline (7,960 MW → 1,378 MW)
```
and
```
Combining lag features and a continuous time counter reduced RMSE by **83%** on the 2026 holdout (7,960 → 1,378 MW)
```

**Why it's wrong:** Real baseline RMSEs on the 2026 holdout (from re-running `08_2026_rmse_figure.py`):

| Model | RMSE (MW) on Current/weather-only baseline, 2026 |
|---|---|
| OLS | 7,985 |
| Ridge | 8,067 |
| Lasso | 8,072 |
| Decision Tree | 7,803 |
| Random Forest | 7,783 (best) |
| Gradient Boost | 7,997 |
| SVR | 7,895 |
| MLP | 8,004 |

No model is at 7,960. The "7,960" appears to be a stale average.

**How to fix:** Pick one of:
- **SVR-to-SVR comparison (cleanest, recommended):** `(7,895 → 1,378 MW, 83% reduction)` — same algorithm, two feature/training setups
- **Best-to-best comparison:** `(7,783 → 1,378 MW, 82% reduction)` — but 82% sounds slightly less crisp
- Use `replace_all` in your editor to replace `7,960` with `7,895` in both locations.

**Verify after fix:** `grep -n "7,960\|7960" final_project/report_outline.md` should return nothing.

### 1c. "16 hours" → 17 hours of live actuals

**Where:** `report_outline.md` slide 16 (Live Evaluation):
```
8 approaches evaluated on April 29, 2026 actual ERCOT load (16 hours)
```

**Why it's wrong:** Today's run of `07_approach_comparison.py` reported "Today actual (NP6-235-CD): **17 hours**" (00:00 through 16:00). The 16 was from an earlier run.

**How to fix:** Change `(16 hours)` to `(17 hours)`.

**Note:** If you re-run `07_approach_comparison.py` before submission and it returns a different hour count (e.g., 18 because more data is available), update accordingly. The `print` statement to look for in script output is `Today actual (NP6-235-CD): N hours`.

### 1d. Live RMSE table (slide 16) — verify by re-running

The table on slide 16 lists best-model live RMSE per approach:

```
Current (weather-only)        Ridge           7.44 GW
Bias-corrected                Ridge           4.42 GW
2025 retrain                  Gradient Boost  5.23 GW
Lag features                  Lasso           1.36 GW
Days-elapsed trend            Ridge           1.11 GW
2025 retrain + lag            Gradient Boost  0.94 GW
2025 retrain + trend          SVR             0.71 GW
2025 retrain + lag + trend    SVR             0.51 GW
```

**These were captured from an earlier April-29 run.** If you re-run
`07_approach_comparison.py` close to submission, the live RMSE values *will*
change because more hours of actual ERCOT load become available throughout the
day. The figures (24 and 25) will auto-update; the slide 16 table will not.

**How to verify/fix:** After running `07_approach_comparison.py`, open
`figures/24_approach_rmse_comparison.png` and read the bar heights for each
approach's best (smallest) bar. Update the slide table to match. The "best
model" name for each approach is the model corresponding to the shortest bar
within that approach's group.

If the table is hard to extract from the figure, you can add a print statement
near the end of `07_approach_comparison.py` to log best-model-per-approach
RMSE values explicitly. Suggested addition just before "Step 7: Generating
figures …":

```python
print("\n  Best model per approach (live RMSE):")
for v in VARIANTS:
    if v['label'] not in all_rmse:
        continue
    best = min(all_rmse[v['label']].items(), key=lambda kv: kv[1])
    print(f"    {v['label']:30s}  {best[0]:18s}  {best[1]/1000:.2f} GW")
```

---

## 2. STRUCTURAL — additions for completeness

These aren't errors but the deck is missing things that all three reference
decks include.

### 2a. Add a Table of Contents slide

**Why:** All three reference decks (Wang, Choi, Goradia) open with one. It
helps a 5-minute reader orient themselves.

**Where:** Insert as a new slide between the title slide and "Problem & Motivation"
(currently slide 1 → slide 2 transition; insert between lines ~35 and ~37 of
`report_outline.md`).

**Suggested content:**
```markdown
## Agenda

1. Problem & motivation
2. Data & feature engineering
3. Model setup & baseline results
4. Classification (peak detection)
5. The underprediction problem & fix
6. Limitations and conclusions
```

### 2b. Add a References slide

**Why:** Project guidance asks for references (per the proposal section), and
Wang's reference deck cites academic sources at the end. You have none in the
final report.

**Where:** Insert before or after the AI Usage Disclosure slide (currently
~line 315–328 of `report_outline.md`).

**Suggested content (verify each link before submitting):**
```markdown
## References

- ERCOT, Hourly Native Load Data: <https://www.ercot.com/gridinfo/load/load_hist>
- ERCOT, NP6-345-CD Actual System Load by Weather Zone API
- Open-Meteo, Historical Forecast API (ERA5 reanalysis): <https://open-meteo.com/>
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, 2ed (2021)
- Hong, T. & Fan, S. (2016). "Probabilistic electric load forecasting: A tutorial review." *Int. Journal of Forecasting*, 32(3), 914–938.
- ERCOT, "Long-Term Load Forecast" reports (data center growth context)
```

The first three are unambiguously real (you used them). Hong & Fan is the
canonical academic citation for load-forecasting methodology — keep it only if
you've actually read it.

### 2c. Restructure the 5→8-approach jump (slides 14→16)

**Why this matters:** Slide 14 ("The Underprediction Problem") describes 5
approaches. Slide 16 ("Live Evaluation") evaluates 8 approaches. The 3 extras
(2025 retrain + lag; 2025 retrain + trend; 2025 retrain + lag + trend) appear
without explanation. The reader gets confused.

**Recommended fix:** rewrite slide 14's table as a 2×2 matrix to make it
explicit that the approaches are formed by orthogonal axes:

| | No extra feature | + Lag features | + Trend feature | + Both |
|---|---|---|---|---|
| **Train 2021–2024** | Current (baseline) | Lag features | Days-elapsed trend | (skipped) |
| **Train 2021–2025 (retrain)** | 2025 retrain | 2025 retrain + lag | 2025 retrain + trend | 2025 retrain + lag + trend |

That gives 7 cells (the 8th is bias-corrected, which is a post-hoc additive
adjustment, not a training/feature change — call it out separately).

**Alternative simpler fix:** Keep slide 14 as-is but add one sentence at the
bottom: *"Combinations of these mechanisms yield 8 distinct evaluated
approaches; see slide 16."*

### 2d. Add hyperparameter tuning visualization (optional, in appendix)

**Why:** Choi's deck shows CCP-α vs F1, n_estimators vs F1, LightGBM heatmap,
etc. Your A8 (Ridge regularization path) and A9 (Lasso CV) already cover
linear models, but tree models have no tuning visualization.

**This is optional — only do if time permits.** Could add an appendix slide
showing the GridSearchCV results for Decision Tree (max_depth × min_samples_leaf
heatmap). Source data is the script `notebooks/04_regression_models.py`
(or wherever the GridSearchCV is performed; search for `grid_dtr`).

### 2e. Recommendations vs. Conclusions split (optional)

**Why:** Goradia's deck has a "Strategic Roadmap" / Recommendations slide
separate from analytical conclusions. Your slide 22 mixes them.

**Optional fix:** Add one slide *between* current slides 21 (Conclusions) and
22 (AI Disclosure) titled "Recommendations" with 3 actionable bullets:

```markdown
## Recommendations for Production Use

1. **For day-ahead operational forecasting**: deploy SVR with 2025 retrain +
   lag + trend; refresh lag inputs hourly via ERCOT API
2. **For long-horizon planning (>1 week)**: use OLS trend-only model;
   coefficient is interpretable and auditable
3. **Re-evaluate quarterly**: distribution-shift recurs as data center
   buildout continues; budget ~30 min/month for retraining
```

---

## 3. INTERNAL CLEANUP

Smaller content/wording issues.

### 3a. Slide 12 — remove unsupported claim

**Where:** `report_outline.md` slide 12 (Classification Results), bullet:
```
- Top features match regression: CDH, hour, temperature
```

**Issue:** No feature-importance plot for the classifier appears anywhere in
the deck or appendix. Either:
- Remove the bullet, or
- Add a feature-importance figure for the classification task to the appendix
  (would need to be generated from `notebooks/05_classification.py` or wherever
  classification models are trained — load the saved model and call
  `.feature_importances_` for tree models or coefficients for logistic).

**Easiest fix:** delete the bullet.

### 3b. Slide 19 (Future Work) — clarify probabilistic forecasting

**Where:** `report_outline.md` slide 19, last bullet:
```
- Probabilistic forecasting: predict a confidence interval, not just a point estimate
```

**Issue:** This wasn't done; it's framed as future work but reads ambiguously.

**Fix:** Either keep as-is (it's already under "Future improvements" header,
so it's clear) — or strengthen with: *"... not just a point estimate
(e.g., quantile regression or conformal prediction)"* to show familiarity.

### 3c. Slides 7 & 8 (EDA + temperature) feel redundant

**Where:** Slides 7 ("EDA: Key Findings") and 8 ("Temperature vs. Load (U-shape)")

**Issue:** Both restate the U-shape finding. Slide 7's bullet 1 and Slide 8's
header bullets cover the same ground.

**Fix (optional):** Either merge into one slide, or differentiate Slide 8 by
making it a quantitative deep-dive (e.g., "Cooling slope: +X MW/°F; Heating
slope: −Y MW/°F") since the two slopes are visibly different in the figure.

### 3d. Verify trend coefficient claim

**Where:** `report_outline.md` slide 17 and slide 22:
```
OLS trend coefficient (trend-only model): **+5.78 MW/day (+2.1 GW/year)**
```

**Issue:** I couldn't find this number in any saved CSV. To confirm, you'd
need to re-fit the trend-only OLS model and read off the `days_elapsed`
coefficient.

**How to verify:** Run this snippet from `final_project/`:
```python
import pandas as pd, statsmodels.api as sm
from ISLP.models import ModelSpec as MS

df = pd.read_csv("data/processed/features_trend.csv", parse_dates=["HourEnding"])
df = df[df.HourEnding.dt.year < 2025]
FEATURES = ['temperature_2m_Austin', 'temperature_2m_Dallas',
            'temperature_2m_Houston', 'temperature_2m_SanAntonio',
            'temperature_2m_avg', 'apparent_temperature_avg',
            'relative_humidity_2m_avg', 'dew_point_2m_avg',
            'wind_speed_10m_avg', 'shortwave_radiation_avg',
            'CDH', 'HDH', 'temp_sq', 'apparent_temp_delta_avg',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
            'is_weekend', 'is_holiday', 'is_workday', 'season',
            'temp_x_hour_sin', 'CDH_x_hour_sin', 'temp_x_is_weekend',
            'days_elapsed']
design = MS(FEATURES).fit(df)
res = sm.OLS(df['Load_MW'], design.transform(df)).fit()
print("days_elapsed coefficient:", res.params.get('days_elapsed'))
print("× 365 = MW/year:", res.params.get('days_elapsed') * 365)
```

If the printed `days_elapsed` coefficient is ≈ 5.78 (and × 365 ≈ 2,100 MW), the
slide is correct. If different, update the slide accordingly.

---

## 4. ABSOLUTELY NOT (do not do these)

- **Do not** delete or modify any file under `figures_archive_prefix/`. Those
  are referenced by appendix slides A1–A15.
- **Do not** rename CSVs or change the `_2025train`/`_lag`/`_trend` naming
  convention. The 8 approach evaluation in `08_2026_rmse_figure.py` parses
  these by suffix.
- **Do not** retrain models from scratch unless asked. Training the full 8
  models (×4 variants = 32 models) takes substantial time. The saved `.pkl`
  and `.pt` files in `models/` are the source of truth.
- **Do not** push to any git remote — this project may not have one
  configured, and the user has not asked.
- **Do not** "improve" the writing style of unmodified slides. Only edit
  slides on this checklist.

---

## 5. VERIFICATION CHECKLIST (run before declaring done)

Tick each:

- [ ] `grep -n "7,960\|7960" final_project/report_outline.md` returns nothing
- [ ] `grep -n "2,663" final_project/report_outline.md` returns nothing
- [ ] `grep -n "16 hours" final_project/report_outline.md` returns nothing (or only in non-live contexts)
- [ ] `python notebooks/07_approach_comparison.py` runs without error
- [ ] `python notebooks/08_2026_rmse_figure.py` runs without error
- [ ] `figures/23`, `24`, `25`, `26` PNG files have today's mtime
- [ ] Slide 16 hour count matches whatever `07_approach_comparison.py` reported
- [ ] Slide 16 RMSE values match figure 24 bars (re-extract if re-ran)
- [ ] Slide 17 baseline RMSE (7,895 or chosen replacement) matches one of the 8 numbers in section 1b above
- [ ] Marp preview of `report_outline.md` renders without overflow on any slide
  (use `npx @marp-team/marp-cli@latest --preview report_outline.md` if Marp CLI installed)
- [ ] Total slide count is reasonable (~25–35 with appendix)

---

## 6. ABOUT THE 3 REFERENCE DECKS

For context — these are in `final_project/reference_files/`:

- **Wang (`wangtianchang_*finaldeck.pdf`)** — Fashion design convergence, 14 slides. Strengths: clean ToC, formal references, hypothesis-driven framing. Uses Fashion-CLIP embeddings + PCA + OLS regression.
- **Choi (`choiwoojin_*Final Project Presentaion*.pdf`)** — Water leakage detection, ~30 slides. Strengths: explicit 2-stage approach (general fitting → fine-tuning), visible hyperparameter sweeps, threshold optimization, feature-importance pies. Uses simulation data (EPANET).
- **Goradia (`goradiashwetamayur_*PredictiveMaintenance*.pdf`)** — Predictive maintenance, 36 slides. Strengths: very polished design, dual classification + regression for the same problem (failure-in-24h binary + RUL hours), explicit "Conclusions & Recommendations" split, feature engineering pipeline diagram.

The Jamie/ERCOT deck has substantially more analytical depth than any of
these (n×p ≈ 1.13M, 8 regression + 5 classification models, real-time
evaluation, distribution-shift narrative). The gaps to close are mostly
presentation polish, not content.
