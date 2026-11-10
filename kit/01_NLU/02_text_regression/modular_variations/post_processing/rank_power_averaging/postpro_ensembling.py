"""
Post-Processing: Rank Averaging & Power Transformation for Ensembling.
Combines multiple diverse model predictions via rank scaling or generalized power means.
"""

from typing import List
import numpy as np
from scipy.stats import rankdata


def rank_average_predictions(prediction_list: List[np.ndarray]) -> np.ndarray:
    """Computes normalized rank average across multiple model predictions."""
    num_models = len(prediction_list)
    if num_models == 0:
        return np.array([])
    if num_models == 1:
        return prediction_list[0]

    n_samples = len(prediction_list[0])
    rank_sum = np.zeros(n_samples)

    for preds in prediction_list:
        ranks = rankdata(preds) / n_samples
        rank_sum += ranks

    return rank_sum / num_models


def power_average_predictions(prediction_list: List[np.ndarray], weights: List[float] = None, power: float = 2.0) -> np.ndarray:
    """Computes generalized power mean: (sum(w_i * y_i^p) / sum(w_i))^(1/p)."""
    num_models = len(prediction_list)
    if weights is None:
        weights = [1.0 / num_models] * num_models

    weights = np.array(weights) / sum(weights)
    weighted_pow_sum = np.zeros_like(prediction_list[0], dtype=float)

    for preds, w in zip(prediction_list, weights):
        weighted_pow_sum += w * (np.maximum(0, preds) ** power)

    return weighted_pow_sum ** (1.0 / power)

