from __future__ import annotations

import importlib
from typing import Any, Mapping

from torch import nn

from src.models.unet import UNet2D


def _load_class(class_path: str) -> type[nn.Module]:
    if ":" not in class_path:
        raise ValueError(
            "model.class_path must use the format 'module.path:ClassName'"
        )
    module_name, class_name = class_path.rsplit(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"Could not import model module '{module_name}'. "
            "The experiment owner must add it before running this config."
        ) from error
    try:
        model_class = getattr(module, class_name)
    except AttributeError as error:
        raise ImportError(
            f"Model class '{class_name}' was not found in '{module_name}'."
        ) from error
    if not isinstance(model_class, type):
        raise TypeError(f"Configured model '{class_path}' is not a class")
    return model_class


def _custom_params(config: Mapping[str, Any]) -> dict[str, Any]:
    params = config.get("params")
    if params is not None:
        if not isinstance(params, Mapping):
            raise ValueError("model.params must be a mapping")
        return dict(params)
    return {
        key: value
        for key, value in config.items()
        if key not in {"name", "class_path"}
    }


def build_model(
    config: Mapping[str, Any], *, pretrained: bool = True
) -> nn.Module:
    name = str(config.get("name", "unet")).strip().lower()
    class_path = config.get("class_path")
    if class_path:
        model_class = _load_class(str(class_path))
        model = model_class(**_custom_params(config))
        if not isinstance(model, nn.Module):
            raise TypeError("Configured model must inherit from torch.nn.Module")
        return model

    in_channels = int(config.get("in_channels", 1))
    out_channels = int(config.get("out_channels", 1))
    if name in {"unet", "unet2d"}:
        return UNet2D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=int(config.get("base_channels", 32)),
        )

    if name in {"smp_unet", "smp-unet"}:
        from src.models.smp_unet import SMPUnet

        encoder_weights = (
            str(config.get("encoder_weights", "imagenet"))
            if pretrained
            else None
        )
        return SMPUnet(
            encoder_name=str(config.get("encoder_name", "resnet34")),
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=int(config.get("classes", out_channels)),
            encoder_depth=int(config.get("encoder_depth", 5)),
        )

    raise ValueError(
        f"Unsupported model.name '{name}'. "
        "Expected unet, unet2d, smp_unet, or a custom model.class_path."
    )
