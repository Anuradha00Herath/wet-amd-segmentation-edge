"""
qat_trainer.py
--------------
QAT fine-tuning training and validation loops.

Reuses from Phase 1/2:
  - utils.logger        (CSVLogger, save_checkpoint, get_logger)
  - utils.dataset       (build_dataloaders)
  - evaluation.metrics  (compute_all_metrics, MetricAccumulator)
  - evaluation.visualization (plot_training_curves)
  - main.CombinedLoss   (CrossEntropy + Dice)

New in Phase 4:
  - Fake-quantization hook management
  - QATScheduler integration
  - QAT-specific CSV logging
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from utils.config import Config
from utils.logger import get_logger, CSVLogger
from evaluation.metrics import compute_all_metrics, MetricAccumulator
from evaluation.visualization import plot_training_curves
from quantization.fake_quant_config import FakeQuantConfig
from quantization.qat_utils import (
    register_fake_quant_hooks, remove_hooks, save_qat_checkpoint
)
from quantization.qat_scheduler import QATScheduler, QATScheduleConfig

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Loss (reuse from main.py — imported to avoid duplication)                  #
# --------------------------------------------------------------------------- #

class CombinedLoss(nn.Module):
    """CrossEntropy + Dice loss. Identical to main.py — defined here to avoid
    circular imports when qat_trainer is used standalone."""

    def __init__(self, num_classes: int = 6, alpha: float = 0.5) -> None:
        super().__init__()
        self.ce    = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.num_classes = num_classes

    def _dice_loss(self, logits, targets):
        probs   = torch.softmax(logits, dim=1)
        one_hot = nn.functional.one_hot(targets, self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        inter   = (probs * one_hot).sum(dim=(2, 3))
        union   = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice    = (2.0 * inter + 1.0) / (union + 1.0)
        return 1.0 - dice.mean()

    def forward(self, logits, targets):
        return self.alpha * self.ce(logits, targets) \
             + (1 - self.alpha) * self._dice_loss(logits, targets)


# --------------------------------------------------------------------------- #
#  QAT Trainer                                                                 #
# --------------------------------------------------------------------------- #

class QATTrainer:
    """
    Manages the full QAT fine-tuning loop.

    Stages:
      1. Warm-up (FP32, no fake quant)
      2. Quant-aware (fake quant enabled on Conv2d weights)
      3. BN-freeze (BatchNorm frozen, fine-tune remaining params)

    Args:
        model:        Pretrained FP32 model.
        cfg:          Project Config.
        fq_cfg:       FakeQuantConfig.
        schedule_cfg: QATScheduleConfig.
        device:       Training device.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        fq_cfg: FakeQuantConfig,
        schedule_cfg: QATScheduleConfig,
        device: torch.device,
    ) -> None:
        self.model        = model.to(device)
        self.cfg          = cfg
        self.fq_cfg       = fq_cfg
        self.schedule_cfg = schedule_cfg
        self.device       = device
        self.criterion    = CombinedLoss(num_classes=cfg.out_channels)
        self._hooks: list = []

        # CSV logger
        self.csv_log = CSVLogger(
            filepath=os.path.join(cfg.csv_dir, "qat_training.csv"),
            fieldnames=[
                "epoch", "phase", "train_loss", "val_loss",
                "val_mean_dice", "val_mean_iou", "val_pixel_acc", "lr",
            ],
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        label: str = "qat",
    ) -> Tuple[nn.Module, Dict[str, List]]:
        """
        Run the full QAT training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader:   Validation DataLoader.
            label:        Checkpoint label prefix.

        Returns:
            (trained_model, history_dict)
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.schedule_cfg.warmup_lr,
            weight_decay=self.cfg.weight_decay,
        )

        scheduler = QATScheduler(self.schedule_cfg, self.model)

        best_dice        = 0.0
        patience_counter = 0
        history = {
            "train_loss": [], "val_loss": [],
            "train_dice": [], "val_dice": [],
        }

        for epoch in range(1, self.schedule_cfg.total_epochs + 1):
            phase = scheduler.step(epoch, optimizer)

            # Enable/disable fake quant hooks
            if scheduler.quant_enabled and not self._hooks:
                self._hooks = register_fake_quant_hooks(self.model, self.fq_cfg)
            elif not scheduler.quant_enabled and self._hooks:
                remove_hooks(self._hooks)
                self._hooks = []

            # ── Train ────────────────────────────────────────────────────────
            train_loss, train_metrics = self._train_epoch(
                train_loader, optimizer, epoch
            )

            # ── Validate ─────────────────────────────────────────────────────
            val_loss, val_metrics = self._val_epoch(val_loader)
            val_dice = val_metrics["mean_dice"]
            current_lr = optimizer.param_groups[0]["lr"]

            # History
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_dice"].append(train_metrics.get("mean_dice", 0.0))
            history["val_dice"].append(val_dice)

            _log.info(
                f"Epoch {epoch:03d}/{self.schedule_cfg.total_epochs} [{phase}] | "
                f"TrLoss {train_loss:.4f} | VaLoss {val_loss:.4f} | "
                f"VaDice {val_dice:.4f} | LR {current_lr:.2e}"
            )

            # CSV log
            self.csv_log.log({
                "epoch":         epoch,
                "phase":         phase,
                "train_loss":    round(train_loss, 6),
                "val_loss":      round(val_loss,   6),
                "val_mean_dice": round(val_metrics.get("mean_dice", 0), 6),
                "val_mean_iou":  round(val_metrics.get("mean_iou",  0), 6),
                "val_pixel_acc": round(val_metrics.get("pixel_acc", 0), 6),
                "lr":            current_lr,
            })

            # Checkpoint
            is_best = val_dice > best_dice
            if is_best:
                best_dice        = val_dice
                patience_counter = 0
            else:
                patience_counter += 1

            save_qat_checkpoint(
                self.model, optimizer, epoch, val_metrics,
                self.cfg, is_best=is_best, label=label,
            )

            if patience_counter >= self.schedule_cfg.patience:
                _log.info(f"Early stopping at epoch {epoch}.")
                break

        # Remove hooks before returning
        if self._hooks:
            remove_hooks(self._hooks)
            self._hooks = []

        _log.info(f"QAT training complete. Best Val Dice: {best_dice:.4f}")
        return self.model, history

    def _train_epoch(
        self, loader: DataLoader, optimizer, epoch: int
    ) -> Tuple[float, Dict]:
        self.model.train()
        total_loss = 0.0
        acc = MetricAccumulator()

        for step, (images, masks) in enumerate(loader, 1):
            images, masks = images.to(self.device), masks.to(self.device)
            optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred_mask   = logits.argmax(dim=1)
            acc.update(compute_all_metrics(pred_mask, masks))

            if step % self.cfg.log_interval == 0:
                _log.info(
                    f"  Epoch {epoch} | Step {step}/{len(loader)} "
                    f"| Loss {loss.item():.4f}"
                )

        return total_loss / len(loader), acc.mean()

    def _val_epoch(self, loader: DataLoader) -> Tuple[float, Dict]:
        self.model.eval()
        total_loss = 0.0
        acc = MetricAccumulator()

        with torch.no_grad():
            for images, masks in loader:
                images, masks = images.to(self.device), masks.to(self.device)
                logits     = self.model(images)
                total_loss += self.criterion(logits, masks).item()
                pred_mask  = logits.argmax(dim=1)
                acc.update(compute_all_metrics(pred_mask, masks))

        return total_loss / len(loader), acc.mean()