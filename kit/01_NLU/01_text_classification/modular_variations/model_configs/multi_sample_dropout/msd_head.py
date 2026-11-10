"""
Multi-Sample Dropout Classification Head.
Reference: Inoue, "Multi-Sample Dropout for Accelerated Training and Better Generalization" (2019).
"""
import torch
import torch.nn as nn
from typing import List

class MultiSampleDropoutClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropouts: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropouts])
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through multiple parallel dropouts and average the logits
        logits = torch.mean(
            torch.stack([self.fc(drop(x)) for drop in self.dropouts], dim=0),
            dim=0
        )
        return logits
