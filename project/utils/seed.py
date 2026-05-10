"""
seed.py
-------
Reproducibility utilities.

Every experiment should call `set_seed(cfg.seed)` before any
stochastic operations so results are reproducible across runs.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set all relevant random seeds for full reproducibility.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Make cuDNN deterministic (slight performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_generator(seed: int = 42) -> torch.Generator:
    """
    Return a seeded PyTorch Generator for DataLoader reproducibility.

    Args:
        seed: Integer seed value.

    Returns:
        Seeded torch.Generator instance.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
