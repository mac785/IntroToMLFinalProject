"""
Phase 5: Classification Models
Secondary task: predict whether a given hour is a PEAK demand hour.

is_peak = 1  if Load_MW >= 68,029 MW (90th percentile)
is_peak = 0  otherwise

Class balance: 90% non-peak, 10% peak — a classic imbalanced problem.
Accuracy alone is misleading (a "predict always 0" classifier gets 90%).
Primary metrics: Recall, Precision, F1, ROC-AUC, Confusion Matrix.

Energy interpretation:
  False Negative (miss a real peak) = fail to dispatch generation → blackout risk
  False Positive (false alarm)      = unnecessary peaker plant spinup → waste

Therefore recall (catching real peaks) > precision in this domain.

Models (ISLP lab style):
  1. Logistic Regression           (statsmodels GLM + Binomial family)
  2. Logistic Regression + Ridge   (sklearn LogisticRegression, tuned C)
  3. Logistic Regression + Lasso   (sklearn LogisticRegression, L1 penalty)
  4. Random Forest Classifier      (sklearn RF)
  5. Gradient Boosting Classifier  (sklearn GBC)
"""

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier as RFC,
                              GradientBoostingClassifier as GBC)
from sklearn.metrics import (roc_auc_score, roc_curve,
                             precision_recall_curve,
                             classification_report,
                             ConfusionMatrixDisplay,
                             confusion_matrix)
from ISLP import confusion_table
from ISLP.models import ModelSpec as MS, summarize
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

ROOT    = Path(__file__).resolve().parent.parent
PROC    = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MODELS  = ROOT / "models"

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
TARGET = 'is_peak'

def savefig(name):
    plt.savefig(FIGURES / name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {name}")


# ══════════════════════════════════════════════════════════════════════════════
# Load data & temporal split (same as regression: train 2021-2024, test 2025)
# ══════════════════════════════════════════════════════════════════════════════
print("=== Phase 5: Classification Models ===\n")

df = pd.read_csv(PROC / "features_nolag.csv", parse_dates=["HourEnding"])
df = df.sort_values("HourEnding").reset_index(drop=True)

train = df[df["HourEnding"].dt.year < 2025]
test  = df[df["HourEnding"].dt.year == 2025]

X_train_raw = train[FEATURE_COLS].values
y_train     = train[TARGET].values
X_test_raw  = test[FEATURE_COLS].values
y_test      = test[TARGET].values

print(f"Training: {len(train):,} rows  |  Peak hours: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"Test:     {len(test):,} rows   |  Peak hours: {y_test.sum():,}  ({y_test.mean()*100:.1f}%)")

tscv = skm.TimeSeriesSplit(n_splits=5)

results_list = []

def clf_metrics(y_true, y_pred, y_prob, label=""):
    from sklearn.metrics import f1_score, precision_score, recall_score
    auc  = roc_auc_score(y_true, y_prob)
    f1   = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    acc  = np.mean(y_true == y_pred)
    print(f"  {label:35s}  AUC={auc:.3f}  F1={f1:.3f}  Prec={prec:.3f}  Recall={rec:.3f}  Acc={acc:.3f}")
    return dict(model=label, AUC=auc, F1=f1, Precision=prec, Recall=rec, Accuracy=acc)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Logistic Regression — statsmodels GLM (ISLP style)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 1. Logistic Regression (statsmodels GLM) ---")

design  = MS(FEATURE_COLS).fit(train)
X_sm    = design.transform(train)
X_sm_te = design.transform(test)
y_sm    = train[TARGET]

glm        = sm.GLM(y_sm, X_sm, family=sm.families.Binomial())
glm_result = glm.fit()

print(f"\n  GLM summary (significant predictors, p < 0.05):")
coef_summary = summarize(glm_result)
sig = coef_summary[coef_summary["P>|z|"] < 0.05].sort_values("coef", key=abs, ascending=False)
print(sig.head(12).to_string())

glm_probs = glm_result.predict(exog=X_sm_te).values
glm_pred  = (glm_probs > 0.5).astype(int)

print("\n  Confusion table (threshold = 0.5):")
ct = confusion_table(glm_pred, y_test)
print(ct)

results_list.append(clf_metrics(y_test, glm_pred, glm_probs, "Logistic (statsmodels)"))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Logistic Regression + Ridge (sklearn, tuned C via cross-validation)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 2. Logistic Regression + Ridge (sklearn, tuned C) ---")

scaler_ridge = StandardScaler()
X_train_sc   = scaler_ridge.fit_transform(X_train_raw)
X_test_sc    = scaler_ridge.transform(X_test_raw)

# Fixed C=0.1 (moderate L2 regularization) — logistic models are baselines
# for comparison; perfect C tuning is not the presentation point here.
ridge_lr = skl.LogisticRegression(
    penalty='l2', C=0.1, solver='lbfgs',
    max_iter=500, random_state=0)
ridge_lr.fit(X_train_sc, y_train)
print(f"  C=0.1  (equiv. λ = 10.0)")

ridge_probs = ridge_lr.predict_proba(X_test_sc)[:, 1]
ridge_pred  = ridge_lr.predict(X_test_sc)

print("\n  Confusion table (threshold = 0.5):")
print(confusion_table(ridge_pred, y_test))
results_list.append(clf_metrics(y_test, ridge_pred, ridge_probs, "Logistic + Ridge"))

joblib.dump((scaler_ridge, ridge_lr), MODELS / "logit_ridge.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Logistic Regression + Lasso (sklearn, tuned C, L1 penalty)
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 3. Logistic Regression + Lasso (sklearn, tuned C) ---")

scaler_lasso = StandardScaler()
X_train_sc_l = scaler_lasso.fit_transform(X_train_raw)
X_test_sc_l  = scaler_lasso.transform(X_test_raw)

lasso_lr = skl.LogisticRegression(
    penalty='l1', C=0.1, solver='liblinear',
    max_iter=500, random_state=0)
lasso_lr.fit(X_train_sc_l, y_train)
print(f"  C=0.1")

lasso_coefs = lasso_lr.coef_[0]
n_zero = np.sum(lasso_coefs == 0)
print(f"  Features zeroed by Lasso: {n_zero} of {len(lasso_coefs)}")
nonzero = [(f, c) for f, c in zip(FEATURE_COLS, lasso_coefs) if c != 0]
print(f"  Retained features ({len(nonzero)}):")
for f, c in sorted(nonzero, key=lambda x: abs(x[1]), reverse=True):
    print(f"    {f:40s}  {c:+.3f}")

lasso_probs = lasso_lr.predict_proba(X_test_sc_l)[:, 1]
lasso_pred  = lasso_lr.predict(X_test_sc_l)

print("\n  Confusion table (threshold = 0.5):")
print(confusion_table(lasso_pred, y_test))
results_list.append(clf_metrics(y_test, lasso_pred, lasso_probs, "Logistic + Lasso"))

joblib.dump((scaler_lasso, lasso_lr), MODELS / "logit_lasso.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Random Forest Classifier
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 4. Random Forest Classifier ---")

rfc = RFC(n_estimators=300,
          max_features='sqrt',
          min_samples_leaf=5,
          class_weight='balanced',   # compensates for 90/10 imbalance
          random_state=0,
          n_jobs=-1)
rfc.fit(X_train_raw, y_train)

rfc_probs = rfc.predict_proba(X_test_raw)[:, 1]
rfc_pred  = rfc.predict(X_test_raw)

print("\n  Confusion table (threshold = 0.5):")
print(confusion_table(rfc_pred, y_test))
results_list.append(clf_metrics(y_test, rfc_pred, rfc_probs, "Random Forest"))

# Feature importance
rfc_imp = pd.DataFrame({'importance': rfc.feature_importances_},
                       index=FEATURE_COLS).sort_values('importance', ascending=False)
print("\n  Top 10 feature importances:")
print(rfc_imp.head(10).to_string())

joblib.dump(rfc, MODELS / "rf_classifier.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Gradient Boosting Classifier
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- 5. Gradient Boosting Classifier ---")

gbc = GBC(n_estimators=150,       # GBC is single-threaded; 150 is sufficient
          learning_rate=0.05,
          max_depth=4,
          min_samples_leaf=10,
          subsample=0.8,
          random_state=0)
gbc.fit(X_train_raw, y_train)
print("  GBC training complete.")

gbc_probs = gbc.predict_proba(X_test_raw)[:, 1]
gbc_pred  = gbc.predict(X_test_raw)

print("\n  Confusion table (threshold = 0.5):")
print(confusion_table(gbc_pred, y_test))
results_list.append(clf_metrics(y_test, gbc_pred, gbc_probs, "Gradient Boosting"))

joblib.dump(gbc, MODELS / "gbc_classifier.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# Results summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Classification Results (Test Set: 2025) ===")
res_df = pd.DataFrame(results_list).set_index('model').sort_values('AUC', ascending=False)
print(res_df.round(3).to_string())
res_df.to_csv(PROC / "classification_results.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

# --- ROC curves for all models ---
fig, ax = subplots(figsize=(8, 7))
model_probs = [
    ("Logistic (statsmodels)", glm_probs),
    ("Logistic + Ridge",       ridge_probs),
    ("Logistic + Lasso",       lasso_probs),
    ("Random Forest",          rfc_probs),
    ("Gradient Boosting",      gbc_probs),
]
colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']
for (label, probs), color in zip(model_probs, colors):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{label}  (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random chance')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=13)
ax.set_title('ROC Curves — Peak Demand Classification')
ax.legend(fontsize=9, loc='lower right')
savefig("17_roc_curves.png")


# --- Precision-Recall curves (best 2 models) ---
fig, ax = subplots(figsize=(8, 6))
best_models = [("Gradient Boosting", gbc_probs, 'red'),
               ("Random Forest",     rfc_probs, 'steelblue')]
for label, probs, color in best_models:
    prec, rec, thresh = precision_recall_curve(y_test, probs)
    ax.plot(rec, prec, lw=2, color=color, label=label)
ax.axhline(y_test.mean(), color='k', ls='--', lw=1,
           label=f'Baseline precision ({y_test.mean():.2f})')
ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision–Recall Curve  (Gradient Boosting vs Random Forest)')
ax.legend(fontsize=10)
savefig("18_precision_recall.png")


# --- Confusion matrices side by side (Logistic vs best ensemble) ---
fig, axes = subplots(1, 2, figsize=(11, 4))
for ax, (label, pred) in zip(axes, [
        ("Logistic Regression", glm_pred),
        ("Gradient Boosting",   gbc_pred)]):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-Peak", "Peak"])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(label)
plt.suptitle('Confusion Matrices — Test Set (2025)', y=1.02, fontsize=13)
plt.tight_layout()
savefig("19_confusion_matrices.png")


# --- Optimal threshold analysis for Gradient Boosting ---
# Show how precision and recall trade off as we vary the decision threshold
prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, gbc_probs)
f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)

fig, ax = subplots(figsize=(9, 5))
ax.plot(thresh_arr, prec_arr[:-1], label='Precision', color='darkorange', lw=2)
ax.plot(thresh_arr, rec_arr[:-1],  label='Recall',    color='steelblue',  lw=2)
ax.plot(thresh_arr, f1_arr[:-1],   label='F1',        color='green',      lw=2, ls='--')
best_f1_idx = np.argmax(f1_arr[:-1])
ax.axvline(thresh_arr[best_f1_idx], c='k', ls=':', lw=1.2,
           label=f'Best F1 threshold = {thresh_arr[best_f1_idx]:.2f}')
ax.set_xlabel('Decision Threshold', fontsize=13)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Gradient Boosting — Precision / Recall / F1 vs Threshold')
ax.legend(fontsize=10)
savefig("20_threshold_analysis.png")

best_thresh = thresh_arr[best_f1_idx]
gbc_pred_opt = (gbc_probs >= best_thresh).astype(int)
print(f"\n  Gradient Boosting at optimal threshold ({best_thresh:.2f}):")
print(confusion_table(gbc_pred_opt, y_test))
results_list.append(clf_metrics(y_test, gbc_pred_opt, gbc_probs,
                                f"GBR (thresh={best_thresh:.2f})"))

print(f"\nAll figures saved to {FIGURES}")
print("Phase 5 complete. Run 06_live_predict.py on presentation day.")
