"""
utils/config_loader.py
----------------------
Load and validate YAML configuration files.

Supports both:
  - configs/config.yaml          (Phase 1 segmentation baseline)
  - configs/detector_config.yaml (Phase 2 ROI detector)

Design decisions:
  - Uses PyYAML for minimal dependencies.
  - Converts nested dicts to SimpleNamespace for dot-access ergonomics.
  - Integer YAML keys (e.g. class ids ``0:``) are coerced to strings so
    they are valid SimpleNamespace keyword arguments.
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

_REQUIRED_KEYS_SEG = {"project", "data", "classes", "model", "inference", "evaluation"}
_REQUIRED_KEYS_DET = {"project", "data", "roi_dataset", "bbox", "training", "detector"}


def _dict_to_namespace(d: Any) -> Any:
    """Recursively convert a nested dict to SimpleNamespace for dot-access.

    YAML integer keys (e.g. class ids ``0:``, ``1:``) are coerced to strings
    so they are valid SimpleNamespace keyword arguments.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{str(k): _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    return d


def load_config(
    config_path: str | Path = "configs/config.yaml",
    config_type: str = "auto",
) -> SimpleNamespace:
    """
    Load YAML config and return as a nested SimpleNamespace.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML config file.
    config_type : "auto" | "segmentation" | "detector"
        Controls which required-key set is validated.
        "auto" infers from filename (contains "detector" → detector config).

    Returns
    -------
    SimpleNamespace
        Nested namespace mirroring the YAML structure.

    Raises
    ------
    FileNotFoundError  If the YAML file does not exist.
    KeyError           If required top-level keys are missing.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw: dict = yaml.safe_load(f)

    if config_type == "auto":
        config_type = "detector" if "detector" in config_path.name else "segmentation"

    required = _REQUIRED_KEYS_DET if config_type == "detector" else _REQUIRED_KEYS_SEG
    missing  = required - set(raw.keys())
    if missing:
        raise KeyError(f"Config '{config_path.name}' missing required keys: {missing}")

    cfg = _dict_to_namespace(raw)

    import torch
    _resolve_device(cfg, torch.cuda.is_available())
    logger.info("Config loaded: %s | type=%s", config_path, config_type)
    return cfg


def _resolve_device(cfg: SimpleNamespace, cuda_available: bool) -> None:
    """Resolve 'auto' device strings in any section that has a device field."""
    for attr in ("inference", "training", "detector"):
        section = getattr(cfg, attr, None)
        if section and getattr(section, "device", None) == "auto":
            section.device = "cuda" if cuda_available else "cpu"


def get_class_info(cfg: SimpleNamespace) -> dict:
    """
    Build the CLASS_INFO dict (Phase 1 segmentation format).

    Returns
    -------
    dict  ``{class_id: {"name": str, "rgb": tuple[int,int,int]}}``
    """
    colors = vars(cfg.classes.colors_rgb)
    names  = vars(cfg.classes.names)
    return {
        int(k): {"name": names[k], "rgb": tuple(colors[k])}
        for k in colors
    }


def get_detector_class_info(cfg: SimpleNamespace) -> dict:
    """
    Build the detector class dict from a detector config.

    Returns
    -------
    dict  ``{class_id (int): "class_name (str)"}``
    """
    return {int(k): v for k, v in vars(cfg.classes.detector_classes).items()}


def get_seg_class_info_from_detector_cfg(cfg: SimpleNamespace) -> dict:
    """
    Build a segmentation CLASS_INFO from a detector config (for bbox generation).

    Returns
    -------
    dict  ``{class_id: {"name": str, "rgb": tuple}}``
    """
    colors = vars(cfg.classes.colors_rgb)
    return {
        int(k): {"name": f"class_{k}", "rgb": tuple(colors[k])}
        for k in colors
    }
