"""
Classification Confusion Drilldown & High-Confidence Error Auditor.
Extracts false positives, false negatives, and ambiguous edge cases.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict

class ConfusionAuditor:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def audit(self, y_true: np.ndarray, y_pred_probs: np.ndarray, texts: pd.Series) -> Dict:
        y_pred = np.argmax(y_pred_probs, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)

        # High-confidence error drilldown
        confidence = np.max(y_pred_probs, axis=1)
        error_mask = (y_true != y_pred)
        
        error_df = pd.DataFrame({
            'text': texts[error_mask].values,
            'true_class': [self.class_names[t] for t in y_true[error_mask]],
            'pred_class': [self.class_names[p] for p in y_pred[error_mask]],
            'confidence': confidence[error_mask]
        }).sort_values(by='confidence', ascending=False)

        return {
            'confusion_matrix': cm,
            'report': report,
            'top_errors': error_df.head(20)
        }
