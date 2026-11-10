# 🔀 Multi-Sample Dropout (MSD) Head

## Overview
Instead of a single dropout layer followed by a linear classification head, MSD duplicates the final hidden representation across multiple dropout masks with differing dropout rates, computes logits through shared weights, and averages the predictions.

## Benefits
- Accelerates training convergence (acts as internal mini-ensembling).
- Improves validation F1 stability on small-to-medium competition datasets.
