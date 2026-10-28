from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from src.models.unet import UNet2D


def build_model(
    config: Mapping[str, Any], *, pretrained: bool = True
) -> nn.Module:
    name = str(config.get("name", "unet")).strip().lower()
    in_channels = int(config.get("in_channels", 1))
    out_channels = int(config.get("out_channels", 1))

    if name in {"unet", "unet2d"}:
        return UNet2D(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=int(config.get("base_channels", 32)),
        )

    if name in {"smp_unet", "segmentation_models_pytorch"}:
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise ImportError(
                "Model 'smp_unet' requires segmentation-models-pytorch. "
                "Install it in the current environment before running E2."
            ) from error
        encoder_weights = config.get("encoder_weights")
        if not pretrained or str(encoder_weights).lower() in {"", "none", "null"}:
            encoder_weights = None
        return smp.Unet(
            encoder_name=str(config.get("encoder_name", "resnet34")),
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
            decoder_attention_type=config.get("decoder_attention_type"),
        )

    raise ValueError(
        f"Unsupported model.name '{name}'. "
        "Expected one of: unet, unet2d, smp_unet."
    )
