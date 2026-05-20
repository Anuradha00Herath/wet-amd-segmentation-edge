"""
pruning_scheduler.py
--------------------
Progressive pruning schedules for gradual model compression.

Implements:
  - Linear schedule: uniform increase per epoch
  - Cosine schedule: smooth increase following cosine curve
  - Step schedule: discrete steps at defined epochs
  - One-shot schedule: single pruning at target ratio

Reuses:
  - utils.logger (get_logger)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

from utils.logger import get_logger

_log = get_logger(__name__)


@dataclass
class PruningScheduleConfig:
    """Configuration for a progressive pruning schedule."""
    initial_ratio:  float = 0.0     # starting pruning ratio
    target_ratio:   float = 0.3     # final pruning ratio
    start_epoch:    int   = 5       # epoch to begin pruning
    end_epoch:      int   = 25      # epoch to reach target ratio
    schedule_type:  str   = "linear"  # linear | cosine | step | oneshot
    step_epochs:    List[int]   = field(default_factory=lambda: [10, 15, 20])
    step_ratios:    List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    prune_freq:     int   = 1       # prune every N epochs


class PruningScheduler:
    """
    Computes pruning ratio for each training epoch.

    Usage:
        scheduler = PruningScheduler(config)
        for epoch in range(total_epochs):
            ratio = scheduler.get_ratio(epoch)
            if scheduler.should_prune(epoch):
                pipeline.prune(model, ratio)
    """

    def __init__(self, config: PruningScheduleConfig) -> None:
        self.cfg = config

    def get_ratio(self, epoch: int) -> float:
        """Return the target pruning ratio for a given epoch."""
        cfg = self.cfg

        if epoch < cfg.start_epoch:
            return cfg.initial_ratio
        if epoch >= cfg.end_epoch:
            return cfg.target_ratio

        progress = (epoch - cfg.start_epoch) / max(cfg.end_epoch - cfg.start_epoch, 1)

        if cfg.schedule_type == "linear":
            ratio = cfg.initial_ratio + progress * (cfg.target_ratio - cfg.initial_ratio)

        elif cfg.schedule_type == "cosine":
            # Smooth ramp from initial to target
            ratio = cfg.initial_ratio + (cfg.target_ratio - cfg.initial_ratio) * (
                1 - math.cos(math.pi * progress)
            ) / 2

        elif cfg.schedule_type == "step":
            ratio = cfg.initial_ratio
            for step_epoch, step_ratio in zip(cfg.step_epochs, cfg.step_ratios):
                if epoch >= step_epoch:
                    ratio = step_ratio

        elif cfg.schedule_type == "oneshot":
            ratio = cfg.target_ratio if epoch >= cfg.start_epoch else cfg.initial_ratio

        else:
            ratio = cfg.initial_ratio + progress * (cfg.target_ratio - cfg.initial_ratio)

        return min(ratio, cfg.target_ratio)

    def should_prune(self, epoch: int) -> bool:
        """Return True if pruning should be applied this epoch."""
        cfg = self.cfg
        if epoch < cfg.start_epoch or epoch > cfg.end_epoch:
            return False
        return (epoch - cfg.start_epoch) % cfg.prune_freq == 0

    def get_schedule(self, total_epochs: int) -> List[float]:
        """Return the full schedule as a list of ratios per epoch."""
        return [self.get_ratio(e) for e in range(total_epochs)]

    def summary(self) -> str:
        cfg = self.cfg
        return (
            f"PruningSchedule({cfg.schedule_type}): "
            f"{cfg.initial_ratio:.0%} → {cfg.target_ratio:.0%} "
            f"epochs {cfg.start_epoch}–{cfg.end_epoch}"
        )


# Pre-defined experiment schedules
PRUNING_EXPERIMENTS = {
    "oneshot_10pct":  PruningScheduleConfig(target_ratio=0.1, schedule_type="oneshot", start_epoch=0),
    "oneshot_20pct":  PruningScheduleConfig(target_ratio=0.2, schedule_type="oneshot", start_epoch=0),
    "oneshot_30pct":  PruningScheduleConfig(target_ratio=0.3, schedule_type="oneshot", start_epoch=0),
    "oneshot_50pct":  PruningScheduleConfig(target_ratio=0.5, schedule_type="oneshot", start_epoch=0),
    "oneshot_70pct":  PruningScheduleConfig(target_ratio=0.7, schedule_type="oneshot", start_epoch=0),
    "progressive_30": PruningScheduleConfig(target_ratio=0.3, schedule_type="cosine",  start_epoch=2, end_epoch=15),
    "progressive_50": PruningScheduleConfig(target_ratio=0.5, schedule_type="cosine",  start_epoch=2, end_epoch=20),
}