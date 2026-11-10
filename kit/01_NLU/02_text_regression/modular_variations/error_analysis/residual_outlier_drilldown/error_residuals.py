"""
Error Analysis: Residual & Outlier Drill-Down.
Identifies Top-K worst predicted samples, computes residual distributions,
and displays sample snippets to diagnose model blind spots.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def analyze_residuals_and_outliers(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
    target_col: str,
    pred_col: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """Audits prediction errors and extracts worst outlier cases."""
    df_eval = df.copy()
    y_true = df_eval[target_col].to_numpy()
    y_pred = df_eval[pred_col].to_numpy()

    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)

    df_eval["_residual"] = residuals
    df_eval["_abs_error"] = abs_errors

    mean_bias = float(np.mean(residuals))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(abs_errors))

    print("\n" + "=" * 70)
    print("[ERROR ANALYSIS] RESIDUAL & OUTLIER DRILL-DOWN")
    print("=" * 70)
    print(f"Overall Metrics - RMSE: {rmse:.4f} | MAE: {mae:.4f} | Mean Bias: {mean_bias:+.4f}")

    if abs(mean_bias) > 0.05:
        print(f"  [WARNING] Systematic Prediction Bias detected: Model consistently {'underpredicts' if mean_bias > 0 else 'overpredicts'} by {abs(mean_bias):.4f}.")

    # Top-K Outliers
    top_outliers = df_eval.sort_values(by="_abs_error", ascending=False).head(top_k)
    print(f"\n--- TOP {top_k} WORST PREDICTION OUTLIERS ---")
    for rank, (_, row) in enumerate(top_outliers.iterrows(), 1):
        snippet = str(row[text_col])[:120].replace("\n", " ")
        print(f"#{rank} [ID: {row[id_col]}] True: {row[target_col]:.3f} | Pred: {row[pred_col]:.3f} | Error: {row['_abs_error']:.3f}")
        print(f"   Snippet: \"{snippet}...\"\n")

    print("=" * 70 + "\n")
    return {
        "rmse": rmse,
        "mae": mae,
        "mean_bias": mean_bias,
        "top_outliers_df": top_outliers[[id_col, target_col, pred_col, "_residual", "_abs_error"]]
    }

