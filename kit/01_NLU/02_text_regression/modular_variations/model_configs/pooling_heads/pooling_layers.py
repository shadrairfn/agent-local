"""
Advanced Pooling Heads for Pretrained Transformers.
Supports:
- MeanPooling (standard & robust)
- CLSPooling (first token representation)
- MaxPooling (dominant feature activation)
- AttentionPooling (learnable query weights)
- GeMPooling (Generalized Mean Pooling for sequence representations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPooling(nn.Module):
    """Computes attention-mask-weighted average across sequence tokens."""

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask


class CLSPooling(nn.Module):
    """Extracts first token [CLS] representation."""

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return last_hidden_state[:, 0]


class MaxPooling(nn.Module):
    """Computes max value across sequence tokens masked by attention."""

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = -1e9
        return torch.max(last_hidden_state, 1)[0]


class AttentionPooling(nn.Module):
    """Applies a learnable projection vector to compute weighted attention scores over tokens."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(last_hidden_state)  # (batch, seq, 1)
        scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        return torch.sum(last_hidden_state * weights, dim=1)


class GeMPooling(nn.Module):
    """Generalized Mean (GeM) Pooling for sequence embeddings."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        x = last_hidden_state.clamp(min=self.eps) * input_mask_expanded
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        x_pow = torch.sum(x.pow(self.p), dim=1) / sum_mask
        return x_pow.pow(1.0 / self.p)


def get_pooling_layer(pooling_type: str, hidden_size: int = 768) -> nn.Module:
    """Factory function for pooling layers."""
    pt = pooling_type.lower().strip()
    if pt == "mean":
        return MeanPooling()
    elif pt == "cls":
        return CLSPooling()
    elif pt == "max":
        return MaxPooling()
    elif pt == "attention":
        return AttentionPooling(hidden_size=hidden_size)
    elif pt == "gem":
        return GeMPooling()
    else:
        print(f"[Warning] Unknown pooling type '{pooling_type}'. Defaulting to MeanPooling.")
        return MeanPooling()

