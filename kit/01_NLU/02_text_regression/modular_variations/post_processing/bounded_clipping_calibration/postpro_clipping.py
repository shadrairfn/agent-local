"""
Post-Processing: Bounded Clipping & Quantile Calibration.
Clamps raw predictions to domain-valid bounds (e.g. [0, 1] for toxicity, or [min, max] of train set)
and provides quantile-matching calibration.
"""

from typing import Optional
import numpy as np


def apply_boundary_clipping(
    predictions: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> np.ndarray:
    """Clips predictions to allowed mathematical range."""
    preds = np.asarray(predictions).copy()
    lower = min_val if min_val is not None else -np.inf
    upper = max_val if max_val is not None else np.inf
    return np.clip(preds, lower, upper)


def apply_quantile_scaling(
    oof_true: np.ndarray,
    predictions: np.ndarray
) -> np.ndarray:
    """Re-scales predictions to match the true target standard deviation (combats variance collapse)."""
    target_std = np.std(oof_true)
    pred_std = np.std(predictions)
    target_mean = np.mean(oof_true)
    pred_mean = np.mean(predictions)

    if pred_std < 1e-6:
        return predictions

    scale = target_std / pred_std
    adjusted = (predictions - pred_mean) * scale + target_mean
    return adjusted

