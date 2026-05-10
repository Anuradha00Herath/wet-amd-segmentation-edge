"""
config.py
---------
Centralised configuration for the OCT wetAMD segmentation research project.
All hyper-parameters, paths and experiment settings live here so every module
can import a single source of truth.

Usage:
    from utils.config import Config
    cfg = Config()
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    #  Paths                                                               #
    # ------------------------------------------------------------------ #
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Resolved lazily so the dataclass stays portable across machines
    @property
    def data_dir(self) -> str:
        return os.path.join(self.project_root, "data")

    @property
    def images_dir(self) -> str:
        return os.path.join(self.data_dir, "images")

    @property
    def masks_dir(self) -> str:
        return os.path.join(self.data_dir, "mask")

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.project_root, "models", "checkpoints")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.project_root, "results")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.results_dir, "logs")

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.results_dir, "plots")

    @property
    def csv_dir(self) -> str:
        return os.path.join(self.results_dir, "csv")

    baseline_checkpoint: str = "baseline.pth"

    # ------------------------------------------------------------------ #
    #  Dataset                                                             #
    # ------------------------------------------------------------------ #
    image_size: Tuple[int, int] = (256, 256)   # (H, W)
    val_split: float = 0.2                     # fraction used for validation
    num_workers: int = 2                       # DataLoader workers (set 0 on Colab if issues arise)
    pin_memory: bool = True

    # Normalisation statistics – placeholder values; update after computing
    # actual dataset mean/std with utils/helpers.py::compute_dataset_stats()
    norm_mean: Tuple[float, ...] = (0.5,)
    norm_std: Tuple[float, ...] = (0.5,)

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    seed: int = 42
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10

    # ------------------------------------------------------------------ #
    #  Model                                                               #
    # ------------------------------------------------------------------ #
    model_name: str = "baseline"       # used for naming saved artefacts
    in_channels: int = 1              # greyscale OCT
    out_channels: int = 1             # binary mask
    # Extend with additional architecture hyper-params as needed

    # ------------------------------------------------------------------ #
    #  Quantization (populated in later phases)                            #
    # ------------------------------------------------------------------ #
    ptq_calib_batches: int = 10       # number of batches for PTQ calibration
    qat_epochs: int = 5
    mixed_precision_bits: List[int] = field(default_factory=lambda: [8, 4])

    # ------------------------------------------------------------------ #
    #  Benchmarking                                                        #
    # ------------------------------------------------------------------ #
    benchmark_runs: int = 100         # warm-up + timed runs
    benchmark_warmup: int = 10

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    log_interval: int = 10            # log every N batches
    save_best_only: bool = True

    def ensure_dirs(self) -> None:
        """Create all output directories if they do not exist."""
        for d in [
            self.checkpoints_dir,
            self.logs_dir,
            self.plots_dir,
            self.csv_dir,
        ]:
            os.makedirs(d, exist_ok=True)

    def __repr__(self) -> str:
        lines = ["Config("]
        for k, v in self.__dict__.items():
            lines.append(f"  {k}={v!r},")
        # Also include properties
        for prop in ["data_dir", "images_dir", "masks_dir", "checkpoints_dir",
                     "results_dir", "logs_dir", "plots_dir", "csv_dir"]:
            lines.append(f"  {prop}={getattr(self, prop)!r},")
        lines.append(")")
        return "\n".join(lines)
