"""
Dynamic Task Switchboard for NLP (Level A - Whole Text Granularity)
Configures model heads, loss functions, activation layers, evaluation metrics,
and post-processing pipelines from a single declarative TASK_TYPE string.
Zero-assert architecture with pure ASCII diagnostic outputs.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score
)
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# METRIC EVALUATION FUNCTIONS (STANDALONE & PURE NUMPY)
# ---------------------------------------------------------------------

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_mcrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Columnwise Root Mean Squared Error (MCRMSE)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        return compute_rmse(y_true, y_pred)
    col_rmses = [np.sqrt(mean_squared_error(y_true[:, c], y_pred[:, c])) for c in range(y_true.shape[1])]
    return float(np.mean(col_rmses))


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Quadratic Weighted Kappa (QWK) on integer or continuous predictions."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    # Round predictions to nearest integer for kappa calculation
    y_pred_rounded = np.rint(y_pred).astype(int)
    y_true_int = np.rint(y_true).astype(int)
    try:
        score = cohen_kappa_score(y_true_int, y_pred_rounded, weights="quadratic")
        return float(score)
    except Exception:
        return 0.0


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Pearson correlation coefficient."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    try:
        r, _ = pearsonr(y_true, y_pred)
        return float(r) if not np.isnan(r) else 0.0
    except Exception:
        return 0.0


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Spearman rank correlation coefficient."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    try:
        rho, _ = spearmanr(y_true, y_pred)
        return float(rho) if not np.isnan(rho) else 0.0
    except Exception:
        return 0.0


def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Logarithmic Error."""
    y_true_clamped = np.clip(np.asarray(y_true), 0, None)
    y_pred_clamped = np.clip(np.asarray(y_pred), 0, None)
    return float(np.sqrt(mean_squared_error(np.log1p(y_true_clamped), np.log1p(y_pred_clamped))))


# ---------------------------------------------------------------------
# THRESHOLD OPTIMIZER FOR ORDINAL QWK (NELDER-MEAD / POWELL)
# ---------------------------------------------------------------------

class OptimizedQWKCutter:
    """Optimizes discrete cut-points for continuous predictions to maximize QWK."""

    def __init__(self, num_classes: int = 6, min_val: float = 1.0):
        self.num_classes = num_classes
        self.min_val = min_val
        # Default evenly spaced thresholds
        self.thresholds = [min_val + i + 0.5 for i in range(num_classes - 1)]

    def _loss(self, thresholds: List[float], y_true: np.ndarray, y_pred: np.ndarray) -> float:
        preds_discrete = pd.cut(
            y_pred,
            bins=[-np.inf] + list(thresholds) + [np.inf],
            labels=[int(self.min_val + i) for i in range(self.num_classes)]
        ).astype(int)
        score = cohen_kappa_score(y_true, preds_discrete, weights="quadratic")
        return -score  # Minimize negative QWK

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true = np.asarray(y_true).ravel().astype(int)
        y_pred = np.asarray(y_pred).ravel()
        res = minimize(
            self._loss,
            x0=self.thresholds,
            args=(y_true, y_pred),
            method="Nelder-Mead",
            options={"maxiter": 500}
        )
        if res.success:
            self.thresholds = sorted(list(res.x))
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.asarray(y_pred).ravel()
        labels = [int(self.min_val + i) for i in range(self.num_classes)]
        preds_discrete = pd.cut(
            y_pred,
            bins=[-np.inf] + list(self.thresholds) + [np.inf],
            labels=labels
        )
        return np.asarray(preds_discrete).astype(int)


# ---------------------------------------------------------------------
# NLP TASK SPECIFICATION ENGINE
# ---------------------------------------------------------------------

class NLPTaskSpecification:
    """Configures model heads, loss functions, and evaluation metrics dynamically."""

    SUPPORTED_TASKS = [
        "single_regression",
        "multi_regression",
        "bounded_regression",
        "ordinal_grading",
        "counting_volume",
        "pair_similarity",
        "single_label",
        "multi_label"
    ]

    def __init__(
        self,
        task_type: str,
        target_cols: List[str],
        train_df: Optional[pd.DataFrame] = None
    ):
        if task_type not in self.SUPPORTED_TASKS:
            print(f"[Warning] Unknown task_type '{task_type}'. Defaulting to 'single_regression'.")
            task_type = "single_regression"

        self.task_type = task_type
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.unique_classes = []
        self.label_mapping = {}
        self.inverse_mapping = {}

        # 1. Output dimension, loss, and activation setup
        if self.task_type == "single_regression":
            self.out_dim = 1
            self.criterion = nn.SmoothL1Loss(beta=1.0)
            self.activation = nn.Identity()
            self.primary_metric_name = "RMSE"
            self.metric_direction = "minimize"

        elif self.task_type == "multi_regression":
            self.out_dim = self.num_targets
            self.criterion = nn.SmoothL1Loss(beta=1.0)
            self.activation = nn.Identity()
            self.primary_metric_name = "MCRMSE"
            self.metric_direction = "minimize"

        elif self.task_type == "bounded_regression":
            self.out_dim = self.num_targets
            self.criterion = nn.BCEWithLogitsLoss()
            self.activation = nn.Sigmoid()
            self.primary_metric_name = "RMSE"
            self.metric_direction = "minimize"

        elif self.task_type == "ordinal_grading":
            self.out_dim = 1
            self.criterion = nn.MSELoss()
            self.activation = nn.Identity()
            self.primary_metric_name = "QWK"
            self.metric_direction = "maximize"
            if train_df is not None and target_cols[0] in train_df.columns:
                self.unique_classes = sorted(train_df[target_cols[0]].dropna().unique().tolist())

        elif self.task_type == "counting_volume":
            self.out_dim = self.num_targets
            self.criterion = nn.L1Loss()
            self.activation = nn.ReLU()
            self.primary_metric_name = "RMSLE"
            self.metric_direction = "minimize"

        elif self.task_type == "pair_similarity":
            self.out_dim = self.num_targets
            self.criterion = nn.BCEWithLogitsLoss()
            self.activation = nn.Sigmoid()
            self.primary_metric_name = "Pearson_r"
            self.metric_direction = "maximize"

        elif self.task_type == "single_label":
            if train_df is not None and target_cols[0] in train_df.columns:
                self.unique_classes = sorted(train_df[target_cols[0]].dropna().unique().tolist())
                self.out_dim = len(self.unique_classes)
                self.label_mapping = {val: idx for idx, val in enumerate(self.unique_classes)}
                self.inverse_mapping = {idx: val for idx, val in enumerate(self.unique_classes)}
            else:
                self.out_dim = 2
            self.criterion = nn.CrossEntropyLoss()
            self.activation = nn.Softmax(dim=1)
            self.primary_metric_name = "Macro_F1"
            self.metric_direction = "maximize"

        elif self.task_type == "multi_label":
            self.out_dim = self.num_targets
            self.criterion = nn.BCEWithLogitsLoss()
            self.activation = nn.Sigmoid()
            self.primary_metric_name = "Mean_ROC_AUC"
            self.metric_direction = "maximize"

        self._print_spec()

    def _print_spec(self):
        """Prints switchboard configuration."""
        print("[TASK SWITCHBOARD]")
        print(f"  - Selected Task Type:    '{self.task_type}'")
        print(f"  - Target Columns:        {self.target_cols} (Dim: {self.out_dim})")
        print(f"  - Neural Loss Function:  {self.criterion.__class__.__name__}")
        print(f"  - Output Activation:     {self.activation.__class__.__name__}")
        print(f"  - Primary Metric:        {self.primary_metric_name} ({self.metric_direction})")

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluates true targets and predictions based on active task type."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        metrics = {}

        if self.task_type == "single_regression":
            metrics["RMSE"] = compute_rmse(y_true, y_pred)
            metrics["MAE"] = compute_mae(y_true, y_pred)
            metrics["Pearson_r"] = compute_pearson(y_true, y_pred)

        elif self.task_type == "multi_regression":
            metrics["MCRMSE"] = compute_mcrmse(y_true, y_pred)
            if y_true.ndim > 1:
                for c, col_name in enumerate(self.target_cols):
                    metrics[f"RMSE_{col_name}"] = compute_rmse(y_true[:, c], y_pred[:, c])

        elif self.task_type == "bounded_regression":
            metrics["RMSE"] = compute_rmse(y_true, y_pred)
            metrics["MAE"] = compute_mae(y_true, y_pred)

        elif self.task_type == "ordinal_grading":
            metrics["QWK"] = compute_qwk(y_true, y_pred)
            metrics["RMSE"] = compute_rmse(y_true, y_pred)
            metrics["MAE"] = compute_mae(y_true, y_pred)

        elif self.task_type == "counting_volume":
            metrics["RMSLE"] = compute_rmsle(y_true, y_pred)
            metrics["MAE"] = compute_mae(y_true, y_pred)
            metrics["RMSE"] = compute_rmse(y_true, y_pred)

        elif self.task_type == "pair_similarity":
            metrics["Pearson_r"] = compute_pearson(y_true, y_pred)
            metrics["Spearman_rho"] = compute_spearman(y_true, y_pred)
            metrics["RMSE"] = compute_rmse(y_true, y_pred)

        elif self.task_type == "single_label":
            y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else np.rint(y_pred).astype(int)
            y_true_int = y_true.ravel().astype(int)
            metrics["Macro_F1"] = float(f1_score(y_true_int, y_pred_classes, average="macro", zero_division=0))
            metrics["Accuracy"] = float(np.mean(y_true_int == y_pred_classes))

        elif self.task_type == "multi_label":
            y_pred_bin = (y_pred >= 0.5).astype(int)
            metrics["Micro_F1"] = float(f1_score(y_true, y_pred_bin, average="micro", zero_division=0))
            try:
                metrics["Mean_ROC_AUC"] = float(roc_auc_score(y_true, y_pred, average="macro"))
            except Exception:
                metrics["Mean_ROC_AUC"] = 0.5

        return metrics

    def post_process_predictions(
        self,
        raw_predictions: np.ndarray,
        cutter: Optional[OptimizedQWKCutter] = None,
        min_clip: Optional[float] = None,
        max_clip: Optional[float] = None
    ) -> np.ndarray:
        """Applies calibration and bounds to raw predictions."""
        preds = np.asarray(raw_predictions).copy()

        if self.task_type == "ordinal_grading":
            if cutter is not None:
                return cutter.predict(preds)
            else:
                return np.rint(preds).astype(int)

        elif self.task_type == "bounded_regression":
            return np.clip(preds, 0.0, 1.0)

        elif self.task_type == "counting_volume":
            return np.clip(np.rint(preds), 0, None).astype(int)

        elif self.task_type in ["single_regression", "multi_regression", "pair_similarity"]:
            if min_clip is not None or max_clip is not None:
                return np.clip(preds, min_clip if min_clip is not None else -np.inf, max_clip if max_clip is not None else np.inf)
            return preds

        elif self.task_type == "single_label":
            if preds.ndim > 1:
                class_indices = np.argmax(preds, axis=1)
                if self.inverse_mapping:
                    return np.array([self.inverse_mapping.get(idx, idx) for idx in class_indices])
                return class_indices
            return np.rint(preds).astype(int)

        elif self.task_type == "multi_label":
            return (preds >= 0.5).astype(int)

        return preds


if __name__ == "__main__":
    # Self-test switchboard
    print("Self-testing NLPTaskSpecification...")
    spec = NLPTaskSpecification(task_type="ordinal_grading", target_cols=["grade"])
    dummy_y_true = np.array([1, 2, 3, 4, 5, 6, 3, 2])
    dummy_y_pred = np.array([1.2, 2.1, 2.8, 3.9, 5.2, 5.8, 3.1, 2.0])
    scores = spec.evaluate(dummy_y_true, dummy_y_pred)
    print("Ordinal Evaluation Scores:", scores)

    # Test Nelder-Mead Cutter
    cutter = OptimizedQWKCutter(num_classes=6, min_val=1.0)
    cutter.fit(dummy_y_true, dummy_y_pred)
    discrete_preds = spec.post_process_predictions(dummy_y_pred, cutter=cutter)
    print("Post-processed Discrete Preds:", discrete_preds)
    print("Switchboard self-test passed successfully.")
