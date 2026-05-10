"""
baseline_model.py
-----------------
Baseline model: UNet++ with EfficientNet-B4 encoder.
Library: segmentation_models_pytorch (smp)

Task    : 6-class OCT segmentation
Classes : Background, Retinal Layer, PED, SRF, IRF, RPE
Input   : Grayscale OCT image  (B, 1, H, W)
Output  : Class logits          (B, 6, H, W)  — apply softmax externally
"""

import segmentation_models_pytorch as smp
import torch.nn as nn

NUM_CLASSES = 6
CLASS_NAMES = ["Background", "Retinal Layer", "PED", "SRF", "IRF", "RPE"]


class BaselineModel(nn.Module):
    """
    UNet++ with EfficientNet-B4 encoder for 6-class OCT segmentation.

    Args:
        in_channels:  Number of input channels (1 for grayscale OCT).
        out_channels: Number of output classes (6).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = NUM_CLASSES) -> None:
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name          = "efficientnet-b4",
            encoder_weights       = "imagenet",
            in_channels           = in_channels,
            classes               = out_channels,
            activation            = None,
            decoder_channels      = (256, 128, 64, 32, 16),
            decoder_use_batchnorm = True,
        )

    def forward(self, x):
        return self.model(x)   # (B, 6, H, W) logits


def build_baseline(in_channels: int = 1, out_channels: int = NUM_CLASSES) -> BaselineModel:
    """Instantiate the baseline model with default OCT settings."""
    return BaselineModel(in_channels=in_channels, out_channels=out_channels)