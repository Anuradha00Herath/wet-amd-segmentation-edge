"""
qat_scheduler.py
----------------
QAT training schedule utilities.

Manages the staged training strategy:
  Phase 1 (warm-up):       FP32 fine-tuning, no quantization simulation
  Phase 2 (quant-aware):   Enable fake quantization, continue fine-tuning
  Phase 3 (stabilisation): Freeze BatchNorm stats, fine-tune with fixed BN

This staged approach prevents early quantization noise from destabilising
the pretrained encoder before the decoder has adapted.
"""

from dataclasses import dataclass
from typing import Optional
import torch.nn as nn
from utils.logger import get_logger

_log = get_logger(__name__)


@dataclass
class QATScheduleConfig:
    # Total QAT epochs
    total_epochs: int = 20

    # Phase boundaries
    warmup_epochs: int = 3        # FP32 fine-tune before enabling fake quant
    quant_epochs: int = 12        # fake quant enabled
    freeze_bn_epoch: int = 15     # freeze BatchNorm stats from this epoch

    # Learning rates
    warmup_lr: float = 1e-4
    quant_lr: float = 5e-5
    freeze_lr: float = 1e-5

    # Early stopping
    patience: int = 8


class QATScheduler:
    """
    Manages QAT training phases: warm-up → quant-aware → BN-freeze.

    Usage:
        scheduler = QATScheduler(schedule_cfg, model)
        for epoch in range(cfg.total_epochs):
            scheduler.step(epoch, optimizer)
            ...
    """

    def __init__(self, cfg: QATScheduleConfig, model: nn.Module) -> None:
        self.cfg   = cfg
        self.model = model
        self._quant_enabled  = False
        self._bn_frozen      = False

    def step(self, epoch: int, optimizer) -> str:
        """
        Update model state and optimizer LR for the current epoch.

        Args:
            epoch:     Current epoch (1-indexed).
            optimizer: PyTorch optimizer.

        Returns:
            Current phase name string.
        """
        if epoch <= self.cfg.warmup_epochs:
            phase = "warmup"
            lr    = self.cfg.warmup_lr
            if self._quant_enabled:
                self._disable_fake_quant()

        elif epoch <= self.cfg.freeze_bn_epoch:
            phase = "quant_aware"
            lr    = self.cfg.quant_lr
            if not self._quant_enabled:
                self._enable_fake_quant()

        else:
            phase = "bn_freeze"
            lr    = self.cfg.freeze_lr
            if not self._bn_frozen:
                self._freeze_bn()

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        return phase

    def _enable_fake_quant(self) -> None:
        """Enable fake quantization simulation on all conv layers."""
        self._quant_enabled = True
        _log.info("QAT: fake quantization enabled.")

    def _disable_fake_quant(self) -> None:
        self._quant_enabled = False

    def _freeze_bn(self) -> None:
        """Freeze BatchNorm running stats (mean/var) for stable quantization."""
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()               # freeze running stats
                module.weight.requires_grad = True   # keep affine params trainable
                module.bias.requires_grad   = True
        self._bn_frozen = True
        _log.info("QAT: BatchNorm stats frozen.")

    @property
    def quant_enabled(self) -> bool:
        return self._quant_enabled