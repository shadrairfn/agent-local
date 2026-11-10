"""Validate a submission CSV against sample_submission.csv.

Usage: python tools/check_submission.py <submission.csv> <sample_submission.csv>
Prints PASS/FAIL lines for row count, id set, duplicates, missing and non-finite values.
"""
import sys

import numpy as np
import pandas as pd

submission_df = pd.read_csv(sys.argv[1])
sample_df = pd.read_csv(sys.argv[2])

print(f"Columns match sample: {list(submission_df.columns) == list(sample_df.columns)}")
print(f"Row count {len(submission_df)} vs sample {len(sample_df)}: {len(submission_df) == len(sample_df)}")
print(f"Ids unique: {submission_df.iloc[:, 0].is_unique}")
print(f"Ids equal sample ids: {set(submission_df.iloc[:, 0]) == set(sample_df.iloc[:, 0])}")
print(f"Missing predictions: {submission_df.iloc[:, 1].isna().sum()}")
numeric_values = pd.to_numeric(submission_df.iloc[:, 1], errors='coerce')
is_numeric_column = numeric_values.notna().mean() > 0.5
if is_numeric_column:
    print(f"Non-finite numeric predictions: {(~np.isfinite(numeric_values.dropna())).sum()}")
print(submission_df.iloc[:, 1].astype(str).value_counts().head(5).to_dict())
