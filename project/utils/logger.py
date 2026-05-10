"""
logger.py
---------
Unified logging utilities providing:
  - Console + file logging via Python's logging module
  - CSV metric logging for experiment tracking
  - Checkpoint saving / loading helpers

All other modules should import from here rather than calling
`print()` or `logging.basicConfig()` directly.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
#  Console / file logger                                                        #
# --------------------------------------------------------------------------- #

def get_logger(name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with console output and optional file output.

    Args:
        name:    Logger name (typically __name__ of the calling module).
        log_dir: Directory for the .log file. Pass None to skip file logging.
        level:   Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already configured
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{name}.log")
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# --------------------------------------------------------------------------- #
#  CSV metric logger                                                            #
# --------------------------------------------------------------------------- #

class CSVLogger:
    """
    Append-mode CSV logger for tracking per-epoch metrics.

    Example:
        csv_log = CSVLogger("results/csv/training.csv",
                            fieldnames=["epoch", "train_loss", "val_dice"])
        csv_log.log({"epoch": 1, "train_loss": 0.45, "val_dice": 0.72})
    """

    def __init__(self, filepath: str, fieldnames: list) -> None:
        self.filepath = filepath
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._initialise()

    def _initialise(self) -> None:
        """Write header row if file does not exist."""
        if not Path(self.filepath).exists():
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """Append a single row dict to the CSV file."""
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writerow(row)

    def reset(self) -> None:
        """Delete existing file and re-write header (start fresh)."""
        if Path(self.filepath).exists():
            os.remove(self.filepath)
        self._initialise()


# --------------------------------------------------------------------------- #
#  Checkpoint utilities                                                         #
# --------------------------------------------------------------------------- #

def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        state:         Dictionary containing model state_dict, epoch, metrics, etc.
        filepath:      Path to save the checkpoint.
        is_best:       If True, also save a copy to best_filepath.
        best_filepath: Where to copy the best checkpoint.
    """
    import torch
    import shutil

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

    if is_best and best_filepath is not None:
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    filepath: str,
    map_location: Any = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint from disk.

    Args:
        filepath:     Path to the .pth file.
        map_location: Device mapping (default 'cpu' for portability).

    Returns:
        The loaded state dictionary.

    Raises:
        FileNotFoundError: If filepath does not exist.
    """
    import torch

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    return torch.load(filepath, map_location=map_location)
