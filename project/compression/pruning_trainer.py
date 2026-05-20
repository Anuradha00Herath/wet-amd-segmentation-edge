"""
pruning_trainer.py
------------------
Fine-tuning pipeline for pruned models.

After structured pruning, accuracy drops. Fine-tuning recovers accuracy
by re-training the remaining weights.

Supports:
  - Standard fine-tuning (fixed pruning, re-train weights)
  - Progressive pruning + fine-tuning (iterative)
  - Combined pruning + quantization (prune → fine-tune → quantize)

Reuses:
  - models.baseline_model      (build_model)
  - utils.dataset              (build_dataloaders)
  - utils.logger               (get_logger, CSVLogger)
  - evaluation.metrics         (compute_all_metrics, MetricAccumulator)
  - compression.structured_pruning (StructuredPruningPipeline)
  - compression.pruning_scheduler  (PruningScheduler)
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.config import Config
from utils.logger import get_logger, CSVLogger
from utils.dataset import build_dataloaders
from evaluation.metrics import compute_all_metrics, MetricAccumulator
from compression.structured_pruning import StructuredPruningPipeline
from compression.pruning_scheduler import PruningScheduler, PruningScheduleConfig

_log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Combined loss                                                               #
# --------------------------------------------------------------------------- #

class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int = 6, ce_weight: float = 0.5) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.softmax(logits, dim=1)
        n_cls   = probs.shape[1]
        targets_oh = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        intersection = (probs * targets_oh).sum(dim=(0, 2, 3))
        union        = (probs + targets_oh).sum(dim=(0, 2, 3))
        dice         = (2 * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + (1 - self.ce_weight) * self.dice_loss(logits, targets)


# --------------------------------------------------------------------------- #
#  Fine-tuning trainer                                                         #
# --------------------------------------------------------------------------- #

class PruningTrainer:
    """
    Fine-tune a pruned model to recover accuracy.

    Usage:
        trainer = PruningTrainer(pruned_model, cfg)
        result  = trainer.fine_tune(epochs=20)
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model  = model
        self.cfg    = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fine_tune(
        self,
        epochs: int = 20,
        lr: float = 1e-4,
        label: str = "pruned",
    ) -> Dict[str, Any]:
        """
        Fine-tune the pruned model.

        Args:
            epochs: Number of fine-tuning epochs.
            lr:     Learning rate.
            label:  Label for checkpoint naming.

        Returns:
            Dict with best Dice, final metrics, and training history.
        """
        train_loader, val_loader, _ = build_dataloaders(self.cfg)
        criterion = CombinedLoss(num_classes=self.cfg.out_channels)
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        csv_path = os.path.join(self.cfg.csv_dir, f"finetune_{label}.csv")
        csv_log  = CSVLogger(csv_path, fieldnames=[
            "epoch", "train_loss", "val_dice", "val_iou", "lr"
        ])

        best_dice  = 0.0
        best_ckpt  = os.path.join(self.cfg.checkpoints_dir, f"{label}_best.pth")
        history    = []

        for epoch in range(1, epochs + 1):
            # Train
            self.model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images = images.to(self.device)
                masks  = masks.to(self.device)
                optimizer.zero_grad()
                logits = self.model(images)
                loss   = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)

            # Validate
            self.model.eval()
            acc = MetricAccumulator()
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device)
                    logits = self.model(images)
                    pred   = logits.argmax(dim=1).cpu()
                    acc.update(compute_all_metrics(pred, masks))

            val_metrics = acc.mean()
            val_dice    = val_metrics.get("mean_dice", 0)
            val_iou     = val_metrics.get("mean_iou", 0)

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            csv_log.log({
                "epoch": epoch, "train_loss": round(train_loss, 6),
                "val_dice": round(val_dice, 6), "val_iou": round(val_iou, 6),
                "lr": current_lr,
            })
            history.append({"epoch": epoch, "val_dice": val_dice, "train_loss": train_loss})

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"model_state": self.model.state_dict(), "epoch": epoch, "dice": val_dice}, best_ckpt)

            _log.info(f"Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} | dice={val_dice:.4f} | lr={current_lr:.2e}")

        _log.info(f"Fine-tuning complete. Best Dice: {best_dice:.4f} → {best_ckpt}")
        return {"best_dice": best_dice, "best_checkpoint": best_ckpt, "history": history}


# --------------------------------------------------------------------------- #
#  Progressive pruning + fine-tuning                                           #
# --------------------------------------------------------------------------- #

class ProgressivePruningTrainer:
    """
    Iterative pruning: prune a little → fine-tune → prune more → fine-tune.

    This produces better accuracy than one-shot pruning at the same final ratio.

    Usage:
        trainer = ProgressivePruningTrainer(model, cfg)
        result  = trainer.run(target_ratio=0.5, steps=5, finetune_epochs=10)
    """

    def __init__(self, model: nn.Module, cfg: Config) -> None:
        self.model = model
        self.cfg   = cfg

    def run(
        self,
        target_ratio: float = 0.3,
        steps: int = 3,
        finetune_epochs: int = 10,
        lr: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Run progressive pruning.

        Args:
            target_ratio:    Final pruning ratio.
            steps:           Number of pruning steps.
            finetune_epochs: Fine-tuning epochs per step.
            lr:              Fine-tuning learning rate.

        Returns:
            Dict with final metrics and per-step history.
        """
        step_ratio  = target_ratio / steps
        current_model = self.model
        all_results   = []

        for step in range(1, steps + 1):
            ratio = step_ratio * step
            _log.info(f"\n── Progressive step {step}/{steps}: ratio={ratio:.0%} ──")

            pipeline = StructuredPruningPipeline(current_model, self.cfg)
            pruned_model, prune_stats = pipeline.prune(step_ratio)

            trainer = PruningTrainer(pruned_model, self.cfg)
            ft_result = trainer.fine_tune(epochs=finetune_epochs, lr=lr, label=f"progressive_step{step}")

            all_results.append({
                "step":           step,
                "cumulative_ratio": ratio,
                "prune_stats":    prune_stats,
                "best_dice":      ft_result["best_dice"],
            })

            # Load best checkpoint for next iteration
            ckpt = torch.load(ft_result["best_checkpoint"], map_location="cpu")
            pruned_model.load_state_dict(ckpt["model_state"])
            current_model = pruned_model

            _log.info(f"Step {step} best Dice: {ft_result['best_dice']:.4f}")

        return {
            "target_ratio": target_ratio,
            "final_model":  current_model,
            "steps":        all_results,
            "final_dice":   all_results[-1]["best_dice"] if all_results else 0,
        }