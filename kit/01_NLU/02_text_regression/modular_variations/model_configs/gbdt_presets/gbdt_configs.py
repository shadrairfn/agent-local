"""
Gradient Boosted Decision Trees (GBDT) Preset Configurations.
Provides battle-tested hyperparameters for LightGBM, CatBoost, and XGBoost
for fast smoke test benchmarks and competition-grade ensembling.
"""

from typing import Dict, Any


def get_lightgbm_preset(task_type: str = "single_regression", mode: str = "thorough", random_state: int = 42) -> Dict[str, Any]:
    """Returns LightGBM hyperparameter dictionary."""
    is_fast = (mode == "fast")

    objective = "regression"
    metric = "rmse"
    if task_type in ["ordinal_grading", "single_regression", "multi_regression"]:
        objective = "regression"
        metric = "rmse"
    elif task_type == "bounded_regression":
        objective = "binary"
        metric = "binary_logloss"
    elif task_type == "counting_volume":
        objective = "poisson"
        metric = "poisson"
    elif task_type == "single_label":
        objective = "multiclass"
        metric = "multi_logloss"

    params = {
        "objective": objective,
        "metric": metric,
        "boosting_type": "gbdt",
        "n_estimators": 200 if is_fast else 800,
        "learning_rate": 0.08 if is_fast else 0.03,
        "num_leaves": 31,
        "max_depth": -1 if is_fast else 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1
    }
    return params


def get_catboost_preset(task_type: str = "single_regression", mode: str = "thorough", random_state: int = 42) -> Dict[str, Any]:
    """Returns CatBoost hyperparameter dictionary."""
    is_fast = (mode == "fast")
    loss_function = "RMSE"
    if task_type == "single_label":
        loss_function = "MultiClass"

    return {
        "loss_function": loss_function,
        "iterations": 250 if is_fast else 1000,
        "learning_rate": 0.08 if is_fast else 0.03,
        "depth": 5 if is_fast else 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False
    }

