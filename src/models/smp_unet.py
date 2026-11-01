from __future__ import annotations

import torch
from torch import nn


class SMPUnet(nn.Module):
    """U-Net backed by segmentation-models-pytorch with a ResNet34 encoder.

    ``encoder_weights`` controls whether the encoder starts from ImageNet
    weights (training) or is left randomly initialized (evaluation/inference,
    where the checkpoint overwrites everything anyway).
    """

    def __init__(
        self,
        *,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 1,
        classes: int = 1,
        encoder_depth: int = 5,
    ) -> None:
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise ImportError(
                "SMPUnet requires 'segmentation-models-pytorch'. "
                "Install it with: pip install segmentation-models-pytorch"
            ) from error
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            encoder_depth=encoder_depth,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
