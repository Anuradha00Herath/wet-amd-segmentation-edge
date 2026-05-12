"""
utils/config_loader.py
----------------------
Load and validate the YAML configuration file.
Provides a single `load_config()` function that returns an
``argparse.Namespace``-style object so fields are accessible as
``cfg.data.image_dir`` etc.

Design decisions:
  - Uses PyYAML for minimal dependencies.
  - Converts nested dicts to SimpleNamespace for dot-access ergonomics.
  - Validates required top-level keys so config errors surface early.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Keys that MUST be present at the top level of the YAML file.
_REQUIRED_KEYS = {"project", "data", "classes", "model", "inference", "evaluation"}


def _dict_to_namespace(d: Any) -> Any:
    """Recursively convert a nested dict to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{str(k): _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    return d


def load_config(config_path: str | Path = "configs/config.yaml") -> SimpleNamespace:
    """
    Load YAML config and return as a nested SimpleNamespace.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.

    Returns
    -------
    SimpleNamespace
        Nested namespace mirroring the YAML structure.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    KeyError
        If required top-level keys are missing.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw: dict = yaml.safe_load(f)

    missing = _REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise KeyError(f"Config is missing required keys: {missing}")

    cfg = _dict_to_namespace(raw)

    # ── Resolve device ────────────────────────────────────────────────────────
    import torch
    if cfg.inference.device == "auto":
        cfg.inference.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Config loaded from %s | device=%s", config_path, cfg.inference.device)

    return cfg


def get_class_info(cfg: SimpleNamespace) -> dict:
    """
    Build the CLASS_INFO dict (compatible with baseline notebook format)
    from the config namespace.

    Returns
    -------
    dict
        ``{class_id: {"name": str, "rgb": tuple[int,int,int]}}``
    """
    colors = vars(cfg.classes.colors_rgb)  # {"0": [...], "1": [...], ...}
    names  = vars(cfg.classes.names)
    return {
        int(k): {"name": names[k], "rgb": tuple(colors[k])}
        for k in colors
    }
