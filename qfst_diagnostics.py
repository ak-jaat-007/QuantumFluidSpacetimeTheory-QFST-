#!/usr/bin/env python3
"""
QFST Diagnostics with CV & Publication Plots
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import qfst_vortex as qv  # Your vortex module (make sure it's importable)

OUTPUT_DIR = "qfst_diagnostics_plots"
PLOT_DIR = os.path.join(OUTPUT_DIR, "publication_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------------
# STEP 1 — Load hybrid summary
# -------------------------
summary_file = "qfst_vs_nfw_summary.csv"
if not os.path.exists(summary_file):
    raise FileNotFoundError(f"{summary_file} not found.")

df_summary = pd.read_csv(summary_file)
print(f"[INFO] Loaded hybrid summary: {len(df_summary)} galaxies")

# -------------------------
# STEP 1.5 — Prepare residuals detail and calculate residual if missing
# -------------------------
# If residual column missing, create a proxy residual (e.g. chi2 difference)
if "residual" not in df_summary.columns:
    if {"chi2_qfst", "chi2_nfw"}.issubset(df_summary.columns):
        df_summary["residual"] = df_summary["chi2_qfst"] - df_summary["chi2_nfw"]
        print("[INFO] 'residual' column created as chi2_qfst - chi2_nfw difference")
    else:
        print("[WARN] No 'residual' column and no chi2 columns to create residuals!")
        df_summary["residual"] = np.nan

residuals_detail = df_summary.copy()
residuals_detail.to_csv(os.path.join(OUTPUT_DIR, "qfst_residuals_detail.csv"), index=False)
print(f"[INFO] Residuals summary saved ({len(residuals_detail)} rows)")

# -------------------------
# STEP 2 — K-Fold Cross Validation
# -------------------------
print(f"[INFO] Prepared {len(df_summary)} galaxies for cross-validation")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

print("[INFO] Starting K-fold cross-validation...")
for fold, (train_idx, test_idx) in enumerate(kf.split(df_summary), 1):
    train = df_summary.iloc[train_idx]
    test = df_summary.iloc[test_idx]
    print(f"[CV] Fold {fold}: train {len(train)} galaxies, test {len(test)} galaxies")

    # Check required columns exist before fitting
    required_cols = ["M_b", "rho0_qfst", "r_core_qfst"]
    if not all(col in train.columns for col in required_cols):
        raise KeyError(f"Missing one of required columns for fitting: {required_cols}")

    # Fit scaling params on training set
    A, p = np.polyfit(np.log10(train["M_b"]), np.log10(train["rho0_qfst"]), 1)
    B, q = np.polyfit(np.log10(train["M_b"]), np.log10(train["r_core_qfst"]), 1)

    for _, row in test.iterrows():
        M_b = row["M_b"]
        rho0_pred = qv.rho0_from_scaling(A, p, M_b)
        rcore_pred = qv.rcore_from_scaling(B, q, M_b)

        # Safely extract comparison metrics, else set NaN
        chi2_qfst = row.get("chi2_qfst", np.nan)
        chi2_nfw = row.get("chi2_nfw", np.nan)
        AIC_qfst = row.get("AIC_qfst", np.nan)
        AIC_nfw = row.get("AIC_nfw", np.nan)
        BIC_qfst = row.get("BIC_qfst", np.nan)
        BIC_nfw = row.get("BIC_nfw", np.nan)

        cv_results.append({
            "galaxy": row["galaxy"],
            "fold": fold,
            "rho0_pred": rho0_pred,
            "r_core_pred": rcore_pred,
            "chi2_qfst": chi2_qfst,
            "chi2_nfw": chi2_nfw,
            "AIC_qfst": AIC_qfst,
            "AIC_nfw": AIC_nfw,
            "BIC_qfst": BIC_qfst,
            "BIC_nfw": BIC_nfw
        })

cv_df = pd.DataFrame(cv_results)
cv_file = os.path.join(OUTPUT_DIR, "qfst_cv_results.csv")
cv_df.to_csv(cv_file, index=False)
print(f"[INFO] Cross-validation results saved: {cv_file}")

# -------------------------
# STEP 3 — Compare QFST vs NFW
# -------------------------
print("[INFO] Computing comparative statistics...")

# Compute deltas safely, fill NaN if any missing columns
for col in ["chi2_qfst", "chi2_nfw", "AIC_qfst", "AIC_nfw", "BIC_qfst", "BIC_nfw"]:
    if col not in cv_df.columns:
        cv_df[col] = np.nan
        print(f"[WARN] Column '{col}' missing in CV results, filling with NaN.")

cv_df["delta_chi2"] = cv_df["chi2_qfst"] - cv_df["chi2_nfw"]
cv_df["delta_AIC"] = cv_df["AIC_qfst"] - cv_df["AIC_nfw"]
cv_df["delta_BIC"] = cv_df["BIC_qfst"] - cv_df["BIC_nfw"]

stats = cv_df[["delta_chi2", "delta_AIC", "delta_BIC"]].agg(["mean", "median", "std"])
stats.to_csv(os.path.join(PLOT_DIR, "qfst_vs_nfw_stats.csv"))
print("[OK] Comparative stats saved.")

# -------------------------
# STEP 4 — Outlier detection
# -------------------------
if "residual" in residuals_detail.columns and residuals_detail["residual"].notnull().any():
    residual_threshold = residuals_detail["residual"].mean() + 3 * residuals_detail["residual"].std()
    outliers = residuals_detail[residuals_detail["residual"] > residual_threshold]
    outliers.to_csv(os.path.join(PLOT_DIR, "outliers.csv"), index=False)
    print(f"[OK] Outliers saved ({len(outliers)} galaxies)")
else:
    print("[WARN] No valid 'residual' column for outlier detection. Skipping this step.")

# -------------------------
# STEP 5 — Plots for publication
# -------------------------
plt.figure(figsize=(8, 6))
plt.boxplot([cv_df["chi2_qfst"].dropna(), cv_df["chi2_nfw"].dropna()], labels=["QFST", "NFW"])
plt.ylabel("Chi-square")
plt.title("Chi-square comparison")
plt.savefig(os.path.join(PLOT_DIR, "chi2_boxplot.png"), dpi=300)
plt.savefig(os.path.join(PLOT_DIR, "chi2_boxplot.pdf"), dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(cv_df["delta_chi2"].dropna(), bins=30, alpha=0.7)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("ΔChi² (QFST − NFW)")
plt.ylabel("Count")
plt.title("Distribution of Chi² differences")
plt.savefig(os.path.join(PLOT_DIR, "delta_chi2_hist.png"), dpi=300)
plt.savefig(os.path.join(PLOT_DIR, "delta_chi2_hist.pdf"), dpi=300)
plt.close()

print(f"[DONE] Diagnostics complete. Outputs in: {PLOT_DIR}")
