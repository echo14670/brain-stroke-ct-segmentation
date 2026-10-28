from __future__ import annotations

import torch
from torch import nn


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        *,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        dimensions = tuple(range(1, probabilities.ndim))
        true_positive = torch.sum(probabilities * targets, dim=dimensions)
        false_positive = torch.sum(
            (1.0 - targets) * probabilities, dim=dimensions
        )
        false_negative = torch.sum(
            targets * (1.0 - probabilities), dim=dimensions
        )
        tversky = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        return torch.pow(1.0 - tversky, self.gamma).mean()
