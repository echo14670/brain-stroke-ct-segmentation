from __future__ import annotations

import math

import torch
from torch import nn


class FocalTverskyLoss(nn.Module):
    """Binary loss accepting logits and 0/1 floating targets [B, 1, H, W].

    TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth).
    Loss = mean((1 - TI) ** gamma), reduced per image like DiceBCELoss.
    With batch=True, aggregate TP/FP/FN over the entire batch before computing
    a single index. The original per-image behavior remains the default.
    alpha weights false positives; beta weights false negatives. gamma is
    the exponent itself, NOT its reciprocal. Defaults follow the weighting
    and exponent of the authors' implementation (whose alpha weights FN):
    https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py

    Half precision inputs accumulate in float32 for AMP stability. Clamping
    1-TI at eps prevents the singular derivative at zero when gamma < 1;
    consequently the minimum loss is eps ** gamma, rather than exact zero.
    Targets must already be on the same device and scaled to [0, 1].
    """

    def __init__(
        self,
        *,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1.0,
        eps: float = 1e-7,
        batch: bool = False,
    ) -> None:
        super().__init__()
        values = (alpha, beta, gamma, smooth, eps)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Loss parameters must be finite")
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError("alpha and beta must be nonnegative with positive sum")
        if gamma <= 0 or smooth <= 0 or not 0 < eps < 1:
            raise ValueError("Require gamma > 0, smooth > 0 and 0 < eps < 1")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps
        if not isinstance(batch, bool):
            raise TypeError("batch must be a boolean")
        self.batch = batch

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must have identical shapes")
        if logits.ndim != 4 or logits.shape[1] != 1 or logits.numel() == 0:
            raise ValueError("Expected nonempty binary tensors [B, 1, H, W]")
        if not logits.is_floating_point() or not targets.is_floating_point():
            raise TypeError("logits and targets must be floating point tensors")
        if logits.device != targets.device:
            raise ValueError("logits and targets must be on the same device")

        # Preserve float64 for numerical gradient checking; promote AMP inputs.
        working_logits = logits if logits.dtype == torch.float64 else logits.float()
        probabilities = torch.sigmoid(working_logits)
        targets = targets.to(dtype=probabilities.dtype)
        dimensions = (0, 1, 2, 3) if self.batch else (1, 2, 3)
        true_positive = (probabilities * targets).sum(dim=dimensions)
        false_positive = (probabilities * (1.0 - targets)).sum(dim=dimensions)
        false_negative = ((1.0 - probabilities) * targets).sum(dim=dimensions)
        index = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        error = (1.0 - index).clamp(min=self.eps, max=1.0)
        return error.pow(self.gamma).mean()
