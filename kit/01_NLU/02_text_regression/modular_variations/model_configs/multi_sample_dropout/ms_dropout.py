"""
Multi-Sample Dropout (MSD) for Faster Convergence & Stronger Generalization.
Passes the pooled embedding through multiple parallel dropout layers (rates 0.1 to 0.5)
and averages their linear output predictions. Proven Kaggle winning technique.
"""

from typing import List
import torch
import torch.nn as nn


class MultiSampleDropoutHead(nn.Module):
    """Linear regression/classification head with multi-sample dropout regularization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_dropouts: int = 5,
        dropout_rates: List[float] = None
    ):
        super().__init__()
        if dropout_rates is None:
            dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5][:num_dropouts]

        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.linear = nn.Linear(in_features, out_features)

        # Standard Xavier/Kaiming initialization
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [self.linear(drop(x)) for drop in self.dropouts]
        return torch.mean(torch.stack(outputs, dim=0), dim=0)

