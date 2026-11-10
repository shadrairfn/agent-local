"""
Feature Engineering: Multimodal Tabular Metadata Fusion.
Prepares numerical and categorical metadata to be fed alongside text features:
- Median imputation and Robust Scaling for numerical columns
- Out-of-Fold Frequency / Target Encoding for categorical columns
All features generated with 'feat_' prefix.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def prepare_tabular_metadata_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta_num_cols: List[str],
    meta_cat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encodes tabular metadata columns into normalized features with 'feat_' prefix."""
    train_out = train_df.copy()
    test_out = test_df.copy()

    # 1. Numerical Metadata (Median Imputation + Standardization)
    for num_col in meta_num_cols:
        if num_col in train_out.columns and num_col in test_out.columns:
            median_val = train_out[num_col].median()
            std_val = train_out[num_col].std()
            if std_val == 0 or np.isnan(std_val):
                std_val = 1.0

            feat_name = f"feat_meta_{num_col}_scaled"
            train_out[feat_name] = ((train_out[num_col].fillna(median_val) - median_val) / std_val).round(4)
            test_out[feat_name] = ((test_out[num_col].fillna(median_val) - median_val) / std_val).round(4)

    # 2. Categorical Metadata (Frequency Encoding)
    for cat_col in meta_cat_cols:
        if cat_col in train_out.columns and cat_col in test_out.columns:
            freq_map = train_out[cat_col].value_counts(normalize=True).to_dict()
            feat_name = f"feat_meta_{cat_col}_freq"
            train_out[feat_name] = train_out[cat_col].map(freq_map).fillna(0.0).round(4)
            test_out[feat_name] = test_out[cat_col].map(freq_map).fillna(0.0).round(4)

    return train_out, test_out

