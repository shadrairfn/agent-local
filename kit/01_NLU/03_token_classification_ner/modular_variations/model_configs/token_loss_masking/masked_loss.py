"""
Masked Cross Entropy Loss for Token Classification.
Ignores padding tokens (PAD) and subword continuations marked with ignore_index=-100.
"""
import torch
import torch.nn as nn
from typing import Optional

class MaskedTokenLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, ignore_index: int = -100):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten batch and sequence dimensions: (N * L, C) vs (N * L)
        loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss
