"""
utils/reproducibility.py
------------------------
Global seed setting and experiment metadata tracking for reproducible research.

Call ``seed_everything(seed)`` once at the start of every script / notebook
cell that involves randomness (data splitting, model initialisation, training).

Design decisions
----------------
- Covers Python random, NumPy, PyTorch CPU, and (if available) CUDA.
- ``torch.backends.cudnn.deterministic = True`` trades a small speed cost
  for reproducible CUDA ops — appropriate for research benchmarking.
- Experiment metadata is written to a JSON sidecar alongside every CSV result
  so runs are self-documenting.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """
    Set all relevant random seeds for full reproducibility.

    Parameters
    ----------
    seed : int  The global seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False   # disable auto-tuner
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Global seed set to %d", seed)


def get_env_metadata() -> dict:
    """
    Collect environment metadata for experiment reproducibility records.

    Returns
    -------
    dict with keys: timestamp, python_version, torch_version, cuda_available,
                    cuda_version, gpu_name, platform, seed (placeholder).
    """
    meta = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version.split()[0],
        "torch_version":  torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version":   torch.version.cuda or "N/A",
        "gpu_name":       torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "platform":       platform.platform(),
    }
    return meta


def save_experiment_metadata(
    output_path: str,
    cfg_dict: dict,
    extra: Optional[dict] = None,
) -> None:
    """
    Write experiment metadata + config snapshot to a JSON file.

    Parameters
    ----------
    output_path : path to the output .json file.
    cfg_dict    : config as a plain dict (use ``namespace_to_dict``).
    extra       : any additional key-value pairs to include.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "environment": get_env_metadata(),
        "config":      cfg_dict,
    }
    if extra:
        payload.update(extra)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Experiment metadata saved → %s", output_path)


def namespace_to_dict(ns) -> dict:
    """
    Recursively convert a SimpleNamespace to a plain dict for JSON serialisation.

    Parameters
    ----------
    ns : SimpleNamespace or any value.

    Returns
    -------
    dict (or original value if not a namespace/list).
    """
    from types import SimpleNamespace
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [namespace_to_dict(i) for i in ns]
    return ns
