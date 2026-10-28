from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(
        self, inputs: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        inputs = self.up(inputs)
        return self.conv(torch.cat((inputs, skip), dim=1))


class UNet2D(nn.Module):
    """Small from-scratch U-Net used as the official baseline."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        bottleneck_channels = base_channels * 16
        self.encoder0 = DoubleConv(in_channels, channels[0])
        self.encoder1 = DownBlock(channels[0], channels[1])
        self.encoder2 = DownBlock(channels[1], channels[2])
        self.encoder3 = DownBlock(channels[2], channels[3])
        self.bottleneck = DownBlock(channels[3], bottleneck_channels)
        self.decoder3 = UpBlock(bottleneck_channels, channels[3], channels[3])
        self.decoder2 = UpBlock(channels[3], channels[2], channels[2])
        self.decoder1 = UpBlock(channels[2], channels[1], channels[1])
        self.decoder0 = UpBlock(channels[1], channels[0], channels[0])
        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip0 = self.encoder0(inputs)
        skip1 = self.encoder1(skip0)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        bottleneck = self.bottleneck(skip3)
        decoded = self.decoder3(bottleneck, skip3)
        decoded = self.decoder2(decoded, skip2)
        decoded = self.decoder1(decoded, skip1)
        decoded = self.decoder0(decoded, skip0)
        return self.output(decoded)
