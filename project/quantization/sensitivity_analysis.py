"""
sensitivity_analysis.py
-----------------------
Layer sensitivity analysis – Phase 3 placeholder.

Sensitivity analysis quantizes each layer independently and measures
the resulting drop in Dice score. Layers that cause large drops
are "sensitive" and should keep higher precision in mixed-precision
configurations.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.config import Config
from utils.logger import get_logger
from evaluation.metrics import compute_all_metrics

_log = get_logger(__name__)


def get_quantizable_layers(model: nn.Module) -> List[str]:
    """
    Return names of layers eligible for quantization (Conv2d, Linear).

    Args:
        model: The model to inspect.

    Returns:
        List of layer name strings.
    """
    eligible = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            eligible.append(name)
    return eligible


def layer_sensitivity_score(
    model: nn.Module,
    layer_name: str,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    bit_width: int = 8,
    baseline_dice: Optional[float] = None,
) -> float:
    """
    Quantize a single layer, evaluate, and return the Dice drop.

    A higher returned value means the layer is MORE sensitive.

    Args:
        model:          Float32 model.
        layer_name:     Name of the layer to temporarily quantize.
        val_loader:     Validation DataLoader.
        cfg:            Config.
        device:         Device.
        bit_width:      Quantization bit width to test.
        baseline_dice:  Pre-computed baseline Dice to avoid recomputing.

    Returns:
        Dice drop (baseline_dice - quantized_dice). Higher = more sensitive.

    Note:
        Placeholder – implement in Phase 3.
    """
    raise NotImplementedError(
        "Sensitivity analysis is a Phase 3 feature. "
        "get_quantizable_layers() is already implemented."
    )


def run_sensitivity_analysis(
    model: nn.Module,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    bit_widths: List[int] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Run full sensitivity analysis across all quantizable layers
    and bit widths.

    Returns:
        Nested dict: {layer_name: {bit_width: dice_drop}}.

    Note:
        Placeholder – implement in Phase 3.
    """
    raise NotImplementedError(
        "Full sensitivity analysis is a Phase 3 feature."
    )
