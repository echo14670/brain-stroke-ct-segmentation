from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from src.losses.dice_bce import DiceBCELoss
from src.losses.focal_tversky import FocalTverskyLoss


def build_loss(config: Mapping[str, Any]) -> nn.Module:
    name = str(config.get("name", "dice_bce")).strip().lower()
    if name in {"dice_bce", "dice+bce"}:
        return DiceBCELoss(
            bce_weight=float(config.get("bce_weight", 1.0)),
            dice_weight=float(config.get("dice_weight", 1.0)),
        )
    if name in {"focal_tversky", "focal_tversky_loss"}:
        return FocalTverskyLoss(
            alpha=float(config.get("alpha", 0.7)),
            beta=float(config.get("beta", 0.3)),
            gamma=float(config.get("gamma", 0.75)),
            smooth=float(config.get("smooth", 1.0)),
        )
    raise ValueError(
        f"Unsupported loss.name '{name}'. "
        "Expected one of: dice_bce, focal_tversky."
    )
