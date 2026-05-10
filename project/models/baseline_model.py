"""
baseline_model.py
-----------------
Lightweight U-Net style baseline for OCT wetAMD binary segmentation.

Architecture summary
--------------------
Encoder  : 4 × (Conv-BN-ReLU) down-sampling blocks
Bottleneck: double conv block
Decoder  : 4 × bilinear upsample + skip-connection + (Conv-BN-ReLU)
Head     : 1×1 conv → sigmoid (binary mask)

This module is intentionally kept framework-clean so it can be
quantized, pruned, or exported to ONNX without modifications.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Building blocks                                                              #
# --------------------------------------------------------------------------- #

class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Max-pool followed by a DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Bilinear upsample + skip concatenation + DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd spatial dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# --------------------------------------------------------------------------- #
#  Baseline U-Net                                                               #
# --------------------------------------------------------------------------- #

class BaselineUNet(nn.Module):
    """
    Lightweight U-Net for OCT binary segmentation.

    Args:
        in_channels:  Number of input image channels (1 for greyscale OCT).
        out_channels: Number of output classes (1 for binary segmentation).
        features:     Channel widths for each encoder level.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256],
    ) -> None:
        super().__init__()

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Stem (first double-conv without pooling)
        self.stem = DoubleConv(in_channels, features[0])

        # Encoder
        for i in range(len(features) - 1):
            self.encoder_blocks.append(DownBlock(features[i], features[i + 1]))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (reverse order)
        dec_features = list(reversed(features))
        self.decoder_blocks.append(
            UpBlock(features[-1] * 2 + features[-1], dec_features[0])
        )
        for i in range(len(dec_features) - 1):
            self.decoder_blocks.append(
                UpBlock(dec_features[i] + dec_features[i + 1], dec_features[i + 1])
            )

        # Segmentation head
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -------- Encoder --------
        skips: List[torch.Tensor] = []
        out = self.stem(x)
        skips.append(out)
        for enc in self.encoder_blocks:
            out = enc(out)
            skips.append(out)

        # -------- Bottleneck --------
        out = self.bottleneck(out)

        # -------- Decoder --------
        for i, dec in enumerate(self.decoder_blocks):
            skip = skips[-(i + 1)]
            out = dec(out, skip)

        return self.head(out)   # logits; apply sigmoid externally for metrics


def build_baseline(in_channels: int = 1, out_channels: int = 1) -> BaselineUNet:
    """Convenience factory matching Config defaults."""
    return BaselineUNet(in_channels=in_channels, out_channels=out_channels)
