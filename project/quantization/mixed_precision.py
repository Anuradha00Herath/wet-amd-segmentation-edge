"""
mixed_precision.py
------------------
Mixed-precision and per-layer bit-width assignment – Phase 3 placeholder.

Enables assigning different precision (e.g. INT8 for early layers,
INT4 for later layers) based on sensitivity analysis results.
"""

from typing import Dict, List

import torch.nn as nn

from utils.config import Config
from utils.logger import get_logger

_log = get_logger(__name__)


def assign_bit_widths(
    model: nn.Module,
    sensitivity_scores: Dict[str, float],
    bit_choices: List[int] = None,
    cfg: Config = None,
) -> Dict[str, int]:
    """
    Assign per-layer bit widths based on sensitivity scores.

    Layers with low sensitivity (small drop in Dice) receive lower
    bit widths; sensitive layers keep higher precision.

    Args:
        model:               The model.
        sensitivity_scores:  Dict mapping layer_name → sensitivity_score.
                             Higher score = more sensitive = keep precision.
        bit_choices:         Allowed bit widths, e.g. [8, 4].
        cfg:                 Config (for bit_choices fallback).

    Returns:
        Dict mapping layer_name → assigned bit width.

    Note:
        Placeholder – implement in Phase 3.
    """
    raise NotImplementedError("Mixed-precision assignment is a Phase 3 feature.")


def apply_mixed_precision(
    model: nn.Module,
    bit_width_map: Dict[str, int],
) -> nn.Module:
    """
    Apply per-layer quantization configs derived from bit_width_map.

    Note:
        Placeholder – implement in Phase 3.
    """
    raise NotImplementedError("Mixed-precision application is a Phase 3 feature.")
