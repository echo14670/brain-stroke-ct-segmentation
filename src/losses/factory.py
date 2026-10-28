from __future__ import annotations

import importlib
from typing import Any, Mapping

from torch import nn

from src.losses.dice_bce import DiceBCELoss


def _load_class(class_path: str) -> type[nn.Module]:
    if ":" not in class_path:
        raise ValueError(
            "loss.class_path must use the format 'module.path:ClassName'"
        )
    module_name, class_name = class_path.rsplit(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"Could not import loss module '{module_name}'. "
            "The experiment owner must add it before running this config."
        ) from error
    try:
        loss_class = getattr(module, class_name)
    except AttributeError as error:
        raise ImportError(
            f"Loss class '{class_name}' was not found in '{module_name}'."
        ) from error
    if not isinstance(loss_class, type):
        raise TypeError(f"Configured loss '{class_path}' is not a class")
    return loss_class


def _custom_params(config: Mapping[str, Any]) -> dict[str, Any]:
    params = config.get("params")
    if params is not None:
        if not isinstance(params, Mapping):
            raise ValueError("loss.params must be a mapping")
        return dict(params)
    return {
        key: value
        for key, value in config.items()
        if key not in {"name", "class_path"}
    }


def build_loss(config: Mapping[str, Any]) -> nn.Module:
    name = str(config.get("name", "dice_bce")).strip().lower()
    class_path = config.get("class_path")
    if class_path:
        loss_class = _load_class(str(class_path))
        loss = loss_class(**_custom_params(config))
        if not isinstance(loss, nn.Module):
            raise TypeError("Configured loss must inherit from torch.nn.Module")
        return loss

    if name in {"dice_bce", "dice+bce"}:
        return DiceBCELoss(
            bce_weight=float(config.get("bce_weight", 1.0)),
            dice_weight=float(config.get("dice_weight", 1.0)),
        )
    if name in {"focal_tversky", "focal_tversky_loss"}:
        raise ImportError(
            "loss.name 'focal_tversky' requires an explicit loss.class_path. "
            "The E3 owner must provide the implementation."
        )
    raise ValueError(
        f"Unsupported loss.name '{name}'. "
        "Expected dice_bce, focal_tversky with class_path, or a custom class_path."
    )
