"""
Temperature Scaling for Logit Calibration.
Learns a single scalar T > 0 to calibrate model confidence before ensembling.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, y_true: np.ndarray, logits: np.ndarray):
        def nll_loss(t):
            temp = max(t[0], 1e-4)
            scaled_logits = logits / temp
            probs = softmax(scaled_logits, axis=1)
            # Negative log-likelihood
            nll = -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-12))
            return nll

        res = minimize(nll_loss, [1.0], bounds=[(0.05, 10.0)], method='L-BFGS-B')
        self.temperature = float(res.x[0])
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits / self.temperature, axis=1)
