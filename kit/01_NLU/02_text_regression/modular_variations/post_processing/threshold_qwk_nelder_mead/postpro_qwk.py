"""
Post-Processing: Nelder-Mead Cut-Point Optimizer for QWK (Quadratic Weighted Kappa).
Calibrates continuous predictions into discrete integer grades strictly on OOF predictions.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score


class QWKThresholdOptimizer:
    """Optimizes discrete boundaries on continuous predictions to maximize Cohen's Kappa."""

    def __init__(self, num_classes: int = 6, min_grade: int = 1):
        self.num_classes = num_classes
        self.min_grade = min_grade
        self.labels = [int(min_grade + i) for i in range(num_classes)]
        self.thresholds = [min_grade + i + 0.5 for i in range(num_classes - 1)]

    def _loss(self, thresholds: List[float], y_true: np.ndarray, y_pred: np.ndarray) -> float:
        preds_disc = pd.cut(
            y_pred,
            bins=[-np.inf] + list(thresholds) + [np.inf],
            labels=self.labels
        )
        score = cohen_kappa_score(y_true, np.asarray(preds_disc).astype(int), weights="quadratic")
        return -score

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true = np.asarray(y_true).ravel().astype(int)
        y_pred = np.asarray(y_pred).ravel()
        res = minimize(
            self._loss,
            x0=self.thresholds,
            args=(y_true, y_pred),
            method="Nelder-Mead",
            options={"maxiter": 600}
        )
        if res.success:
            self.thresholds = sorted(list(res.x))
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.asarray(y_pred).ravel()
        preds_disc = pd.cut(
            y_pred,
            bins=[-np.inf] + list(self.thresholds) + [np.inf],
            labels=self.labels
        )
        return np.asarray(preds_disc).astype(int)

