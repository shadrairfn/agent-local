"""
Error Analysis: Slice-Based Performance Auditing.
Evaluates model performance across document length bins and categorical/group slices
to isolate specific subsets where the model underperforms.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def evaluate_slices_by_length_and_group(
    df: pd.DataFrame,
    text_col: str,
    target_col: str,
    pred_col: str,
    group_col: str = None
) -> pd.DataFrame:
    """Evaluates RMSE and MAE partitioned by document length buckets and entity groups."""
    df_eval = df.copy()
    y_true = df_eval[target_col].to_numpy()
    y_pred = df_eval[pred_col].to_numpy()

    df_eval["_word_count"] = df_eval[text_col].astype(str).apply(lambda x: len(x.split()))
    df_eval["_error"] = y_true - y_pred
    df_eval["_sq_error"] = (y_true - y_pred) ** 2

    # 1. Length Bucket Evaluation
    try:
        df_eval["_length_slice"] = pd.qcut(df_eval["_word_count"].rank(method="first"), q=3, labels=["Short", "Medium", "Long"])
    except Exception:
        df_eval["_length_slice"] = pd.cut(df_eval["_word_count"], bins=3, labels=["Short", "Medium", "Long"])

    print("\n" + "=" * 70)
    print("[ERROR ANALYSIS] SLICE-BASED PERFORMANCE AUDITING")
    print("=" * 70)
    print("--- Performance by Document Length Slice ---")
    length_summary = df_eval.groupby("_length_slice", observed=False).agg(
        Count=("_sq_error", "count"),
        RMSE=("_sq_error", lambda x: np.sqrt(np.mean(x))),
        MAE=("_error", lambda x: np.mean(np.abs(x))),
        Mean_Bias=("_error", "mean")
    )
    print(length_summary.round(4).to_string())

    # 2. Group Slice Evaluation (if group_col provided)
    if group_col and group_col in df_eval.columns:
        print(f"\n--- Performance by Group Slice ('{group_col}') ---")
        group_summary = df_eval.groupby(group_col).agg(
            Count=("_sq_error", "count"),
            RMSE=("_sq_error", lambda x: np.sqrt(np.mean(x))),
            MAE=("_error", lambda x: np.mean(np.abs(x)))
        )
        print(group_summary.round(4).to_string())

    print("=" * 70 + "\n")
    return length_summary

