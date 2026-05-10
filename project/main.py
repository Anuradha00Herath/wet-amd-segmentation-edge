"""
main.py
-------
Unified command-line entry point for the OCT wetAMD segmentation
research framework.

Modes
-----
  train      – Train the baseline model from scratch.
  evaluate   – Evaluate a saved checkpoint on the test set.
  benchmark  – Measure inference latency and model stats.
  profile    – Run torch.profiler for operator-level breakdown.
  export     – Export model to ONNX (Phase 2 placeholder).

Usage (examples)
----------------
  python main.py train
  python main.py train --epochs 60 --batch_size 8
  python main.py evaluate --checkpoint models/checkpoints/best_baseline.pth
  python main.py benchmark
  python main.py profile

Google Colab
------------
  !python main.py train --epochs 60
"""

import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils.config import Config
from utils.seed import set_seed
from utils.helpers import get_device, device_info, print_model_summary
from utils.logger import get_logger, CSVLogger, save_checkpoint
from utils.dataset import build_dataloaders, build_inference_loader

from models.model_loader import load_model, get_model, SegmentationInference
from models.baseline_model import CLASS_NAMES, NUM_CLASSES

from evaluation.metrics import compute_all_metrics, MetricAccumulator
from evaluation.benchmark import Benchmarker
from evaluation.profiler import profile_model
from evaluation.visualization import (
    plot_predictions, plot_training_curves, plot_per_class_dice
)


# --------------------------------------------------------------------------- #
#  CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCT wetAMD Segmentation – Research Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["train", "evaluate", "benchmark", "profile", "export"],
        help="Run mode.",
    )

    # Training overrides
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--seed",       type=int,   default=None)

    # Evaluation
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pth checkpoint for evaluate / benchmark / profile modes.",
    )

    # Hardware
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")

    # Visualisation
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip saving plots.")

    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Loss                                                                         #
# --------------------------------------------------------------------------- #

class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, num_classes: int = NUM_CLASSES, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.softmax(logits, dim=1)
        one_hot = nn.functional.one_hot(targets, self.num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()               # (B, C, H, W)
        inter   = (probs * one_hot).sum(dim=(2, 3))
        union   = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice    = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """CrossEntropy + Dice loss (alpha * CE + (1-alpha) * Dice)."""

    def __init__(self, num_classes: int = NUM_CLASSES, alpha: float = 0.5) -> None:
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ce(logits, targets) \
             + (1 - self.alpha) * self.dice(logits, targets)


# --------------------------------------------------------------------------- #
#  Encoder freeze / unfreeze                                                   #
# --------------------------------------------------------------------------- #

def freeze_encoder(model: nn.Module) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_encoder(model: nn.Module) -> None:
    for p in model.encoder.parameters():
        p.requires_grad = True


def make_optimizer(model: nn.Module, cfg: Config, frozen: bool) -> AdamW:
    if frozen:
        return AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
    return AdamW([
        {"params": model.encoder.parameters(),          "lr": cfg.encoder_lr},
        {"params": model.decoder.parameters(),          "lr": cfg.learning_rate},
        {"params": model.segmentation_head.parameters(),"lr": cfg.learning_rate},
    ], weight_decay=cfg.weight_decay)


# --------------------------------------------------------------------------- #
#  Training                                                                     #
# --------------------------------------------------------------------------- #

def run_train(cfg: Config, device: torch.device, args: argparse.Namespace) -> None:
    log = get_logger("train", log_dir=cfg.logs_dir)

    if args.epochs:     cfg.num_epochs    = args.epochs
    if args.lr:         cfg.learning_rate = args.lr
    if args.batch_size: cfg.batch_size    = args.batch_size

    log.info(f"Device : {device_info(device)}")
    log.info(f"Epochs : {cfg.num_epochs} | LR : {cfg.learning_rate} | "
             f"Batch  : {cfg.batch_size}")

    # Data — 70 / 15 / 15 split
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    log.info(f"Train: {len(train_loader)} batches | "
             f"Val: {len(val_loader)} batches | "
             f"Test: {len(test_loader)} batches")

    # Model
    model = get_model(cfg.model_name, cfg.in_channels, cfg.out_channels).to(device)
    print_model_summary(model, (cfg.in_channels, *cfg.image_size), device)

    # Loss
    criterion = CombinedLoss(num_classes=cfg.out_channels)

    # Phase 1: freeze encoder, train decoder only
    freeze_encoder(model)
    log.info(f"Phase 1 — decoder only (epochs 1–{cfg.freeze_epochs})")
    optimizer  = make_optimizer(model, cfg, frozen=True)
    scheduler  = CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.freeze_epochs, eta_min=1e-6
    )

    # CSV logger
    csv_fields = (
        ["epoch", "phase", "train_loss", "val_loss", "val_mean_dice",
         "val_mean_iou", "val_pixel_acc", "lr"]
        + [f"val_dice_{n.lower().replace(' ','_')}" for n in CLASS_NAMES]
    )
    csv_log = CSVLogger(
        filepath=f"{cfg.csv_dir}/train_metrics.csv",
        fieldnames=csv_fields,
    )

    best_dice        = 0.0
    patience_counter = 0
    train_losses, val_losses   = [], []
    train_dices,  val_dices    = [], []

    best_ckpt = f"{cfg.checkpoints_dir}/best_{cfg.model_name}.pth"
    last_ckpt = f"{cfg.checkpoints_dir}/last_{cfg.model_name}.pth"

    for epoch in range(1, cfg.num_epochs + 1):

        # Switch to Phase 2: unfreeze encoder
        if epoch == cfg.freeze_epochs + 1:
            unfreeze_encoder(model)
            optimizer = make_optimizer(model, cfg, frozen=False)
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            log.info(f"Phase 2 — full fine-tuning (epoch {epoch}+)")

        phase = "frozen" if epoch <= cfg.freeze_epochs else "full"

        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_acc  = MetricAccumulator()

        for step, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)                          # (B, 6, H, W)
            loss   = criterion(logits, masks)               # masks: (B, H, W) int64
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pred_mask   = logits.argmax(dim=1)              # (B, H, W)
            train_acc.update(compute_all_metrics(pred_mask, masks))

            if step % cfg.log_interval == 0:
                log.info(
                    f"Epoch {epoch}/{cfg.num_epochs} [{phase}] | "
                    f"Step {step}/{len(train_loader)} | Loss {loss.item():.4f}"
                )

        scheduler.step()
        train_loss   /= len(train_loader)
        train_metrics = train_acc.mean()

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_acc  = MetricAccumulator()

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits     = model(images)
                val_loss  += criterion(logits, masks).item()
                pred_mask  = logits.argmax(dim=1)
                val_acc.update(compute_all_metrics(pred_mask, masks))

        val_loss    /= len(val_loader)
        val_metrics  = val_acc.mean()
        val_dice     = val_metrics["mean_dice"]
        current_lr   = optimizer.param_groups[0]["lr"]

        # History
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_metrics.get("mean_dice", 0.0))
        val_dices.append(val_dice)

        log.info(
            f"Epoch {epoch:04d} [{phase}] | "
            f"TrLoss {train_loss:.4f} | VaLoss {val_loss:.4f} | "
            f"VaDice {val_dice:.4f} | VaIoU {val_metrics['mean_iou']:.4f} | "
            f"LR {current_lr:.2e}"
        )

        # CSV row
        row = {
            "epoch": epoch, "phase": phase,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "val_mean_dice": round(val_metrics["mean_dice"],  6),
            "val_mean_iou":  round(val_metrics["mean_iou"],   6),
            "val_pixel_acc": round(val_metrics["pixel_acc"],  6),
            "lr": current_lr,
        }
        for name in CLASS_NAMES:
            key = f"dice_{name.lower().replace(' ', '_')}"
            row[f"val_{key}"] = round(val_metrics.get(key, 0.0), 6)
        csv_log.log(row)

        # Checkpoint
        is_best = val_dice > best_dice
        if is_best:
            best_dice        = val_dice
            patience_counter = 0
        else:
            patience_counter += 1

        state = {
            "epoch":                epoch,
            "model_state":          model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dice":                 val_dice,
            "config":               cfg.__dict__,
        }
        save_checkpoint(state, last_ckpt,
                        is_best=is_best, best_filepath=best_ckpt)

        if patience_counter >= cfg.early_stopping_patience:
            log.info(f"Early stopping at epoch {epoch}.")
            break

    log.info(f"Training complete. Best Val Dice: {best_dice:.4f}")

    if not args.no_plot:
        plot_training_curves(
            train_losses, val_losses, train_dices, val_dices, cfg
        )


# --------------------------------------------------------------------------- #
#  Evaluation                                                                   #
# --------------------------------------------------------------------------- #

def run_evaluate(cfg: Config, device: torch.device, args: argparse.Namespace) -> None:
    log = get_logger("evaluate", log_dir=cfg.logs_dir)

    model = load_model(cfg, checkpoint_path=args.checkpoint, device=device)
    inf   = SegmentationInference(model, device)

    # Evaluate on test set
    _, _, test_loader = build_dataloaders(cfg)
    acc = MetricAccumulator()

    sample_images, sample_gt, sample_pred = [], [], []

    with torch.no_grad():
        for images, masks in test_loader:
            probs, pred_mask = inf.predict(images)
            acc.update(compute_all_metrics(pred_mask, masks))
            if len(sample_images) < 4:
                sample_images.append(images)
                sample_gt.append(masks)
                sample_pred.append(pred_mask)

    metrics = acc.mean()
    log.info("Test Set Results:")
    log.info(f"  Mean Dice : {metrics['mean_dice']:.4f}")
    log.info(f"  Mean IoU  : {metrics['mean_iou']:.4f}")
    log.info(f"  Pixel Acc : {metrics['pixel_acc']:.4f}")
    log.info("Per-class Dice:")
    for name in CLASS_NAMES:
        key = f"dice_{name.lower().replace(' ', '_')}"
        log.info(f"  {name:<18} {metrics.get(key, 0.0):.4f}")

    # Save CSV
    csv_log = CSVLogger(
        f"{cfg.csv_dir}/eval_results.csv",
        fieldnames=list(metrics.keys()),
    )
    csv_log.log(metrics)

    if not args.no_plot and sample_images:
        plot_predictions(
            torch.cat(sample_images),
            torch.cat(sample_gt),
            torch.cat(sample_pred),
            cfg, n=4,
        )
        class_dice_scores = [
            metrics.get(f"dice_{n.lower().replace(' ', '_')}", 0.0)
            for n in CLASS_NAMES
        ]
        plot_per_class_dice(class_dice_scores, cfg)


# --------------------------------------------------------------------------- #
#  Benchmark                                                                    #
# --------------------------------------------------------------------------- #

def run_benchmark(cfg: Config, device: torch.device, args: argparse.Namespace) -> None:
    model   = load_model(cfg, checkpoint_path=args.checkpoint, device=device)
    bench   = Benchmarker(cfg, device)
    results = bench.run(model, label=cfg.model_name)
    Benchmarker.print_results(results)

    csv_log = CSVLogger(
        f"{cfg.csv_dir}/benchmark_results.csv",
        fieldnames=list(results.keys()),
    )
    csv_log.log(results)


# --------------------------------------------------------------------------- #
#  Profiler                                                                     #
# --------------------------------------------------------------------------- #

def run_profile(cfg: Config, device: torch.device, args: argparse.Namespace) -> None:
    model = load_model(cfg, checkpoint_path=args.checkpoint, device=device)
    dummy = torch.zeros(1, cfg.in_channels, *cfg.image_size).to(device)
    profile_model(model, dummy, cfg, trace_name=cfg.model_name)


# --------------------------------------------------------------------------- #
#  ONNX Export (Phase 2 placeholder)                                           #
# --------------------------------------------------------------------------- #

def run_export(cfg: Config, device: torch.device, args: argparse.Namespace) -> None:
    raise NotImplementedError(
        "ONNX export is a Phase 2 feature. "
        "Implement using torch.onnx.export() in models/model_loader.py."
    )


# --------------------------------------------------------------------------- #
#  Dispatcher                                                                   #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    cfg  = Config()

    if args.seed:
        cfg.seed = args.seed
    set_seed(cfg.seed)
    cfg.ensure_dirs()

    device = torch.device("cpu") if args.cpu else get_device()

    dispatch = {
        "train":     run_train,
        "evaluate":  run_evaluate,
        "benchmark": run_benchmark,
        "profile":   run_profile,
        "export":    run_export,
    }
    dispatch[args.mode](cfg, device, args)


if __name__ == "__main__":
    main()