"""
Probability Threshold Optimizer for Macro F1.
Finds optimal per-class probability scaling multipliers to maximize competition metric.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

class ProbabilityThresholdOptimizer:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.multipliers = np.ones(num_classes)

    def fit(self, y_true: np.ndarray, oof_probs: np.ndarray):
        def loss_func(weights):
            scaled_probs = oof_probs * weights
            preds = np.argmax(scaled_probs, axis=1)
            score = f1_score(y_true, preds, average='macro')
            return -score  # minimize negative F1

        init_weights = np.ones(self.num_classes)
        res = minimize(loss_func, init_weights, method='Nelder-Mead', options={'maxiter': 500})
        self.multipliers = res.x
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        scaled_probs = probs * self.multipliers
        return np.argmax(scaled_probs, axis=1)
