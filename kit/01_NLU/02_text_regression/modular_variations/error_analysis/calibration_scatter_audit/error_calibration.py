"""
Error Analysis: Calibration & Quantile Distribution Audit.
Compares predicted percentiles against true percentiles to detect
prediction compression (variance collapse) or scale mismatch.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def audit_prediction_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Audits true vs predicted percentiles and spread."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    true_pcts = [np.percentile(y_true, p) for p in percentiles]
    pred_pcts = [np.percentile(y_pred, p) for p in percentiles]

    df_calib = pd.DataFrame({
        "Percentile": [f"P{p:02d}" for p in percentiles],
        "True_Target": np.round(true_pcts, 4),
        "Predicted": np.round(pred_pcts, 4),
        "Delta": np.round(np.array(pred_pcts) - np.array(true_pcts), 4)
    })

    print("\n" + "=" * 70)
    print("[ERROR ANALYSIS] CALIBRATION & DISTRIBUTION SPREAD AUDIT")
    print("=" * 70)
    print(df_calib.to_string(index=False))

    true_std = np.std(y_true)
    pred_std = np.std(y_pred)
    ratio = pred_std / max(1e-6, true_std)
    print(f"\nVariance Ratio (Pred_Std / True_Std): {ratio:.3f}")
    if ratio < 0.70:
        print("  [WARNING] Significant Prediction Compression (Variance Collapse) detected!")
        print("     Model predictions are clustering near the mean and failing to reach extremes.")
        print("     Consider post-processing quantile scaling or power transformation.")

    print("=" * 70 + "\n")
    return df_calib

