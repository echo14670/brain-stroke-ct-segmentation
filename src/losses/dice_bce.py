from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probabilities = torch.sigmoid(logits)
        dimensions = tuple(range(1, probabilities.ndim))
        intersection = torch.sum(probabilities * targets, dim=dimensions)
        denominator = torch.sum(probabilities + targets, dim=dimensions)
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss
