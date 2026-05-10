"""
baseline_model.py
-----------------
Baseline model: UNet++ with EfficientNet-B4 encoder.
Library: segmentation_models_pytorch (smp)

Task    : 6-class OCT segmentation
Classes : Background, Retinal Layer, PED, SRF, IRF, RPE
Input   : Grayscale OCT image  (B, 1, H, W)
Output  : Class logits          (B, 6, H, W)  — apply softmax externally

NOTE: The smp model is returned directly (NOT wrapped in nn.Module)
so that checkpoint keys match exactly what was saved during training:
    "encoder.*", "decoder.*", "segmentation_head.*"
"""

import segmentation_models_pytorch as smp
import torch.nn as nn

NUM_CLASSES = 6
CLASS_NAMES = ["Background", "Retinal Layer", "PED", "SRF", "IRF", "RPE"]


def build_baseline(in_channels: int = 1, out_channels: int = NUM_CLASSES) -> nn.Module:
    """
    Instantiate UNet++ with EfficientNet-B4 encoder.

    Returns the smp model directly so state_dict keys match
    the original checkpoint exactly.
    """
    return smp.UnetPlusPlus(
        encoder_name          = "efficientnet-b4",
        encoder_weights       = "imagenet",
        in_channels           = in_channels,
        classes               = out_channels,
        activation            = None,
        decoder_channels      = (256, 128, 64, 32, 16),
        decoder_use_batchnorm = True,
    )