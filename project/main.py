"""
main.py
-------
Unified command-line entry point for the OCT wetAMD segmentation
research framework.

Modes
-----
  train      – Train the baseline model from scratch.
  evaluate   – Evaluate a saved checkpoint on the validation set.
  benchmark  – Measure inference latency and model stats.
  profile    – Run torch.profiler for operator-level breakdown.
  export     – Export model to ONNX (Phase 2 placeholder).

Usage (examples)
--------------
  python main.py train
  python main.py train --epochs 100 --lr 0.0005 --batch_size 16
  python main.py evaluate --checkpoint models/checkpoints/best.pth
  python main.py benchmark
  python main.py profile

Google Colab
-----------
  !python main.py train --epochs 50
"""

import argparse
import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.config import Config
from utils.seed import set_seed
from utils.helpers import get_device, device_info, print_model_summary
from utils.logger import get_logger, CSVLogger, save_checkpoint
from utils.dataset import build_dataloaders, build_inference_loader

from models.model_loader import load_model, get_model, SegmentationInference

from evaluation.metrics import compute_all_metrics, MetricAccumulator
from evaluation.benchmark import Benchmarker
from evaluation.profiler import profile_model
from evaluation.visualization import plot_predictions, plot_training_curves


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

    # Evaluation / export
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pth checkpoint for evaluate / export modes.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Sigmoid threshold for binary mask prediction.",
    )

    # Hardware
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")

    # Visualisation
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip saving plots.")

    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Training                                                                     #
# --------------------------------------------------------------------------- #

def run_train(cfg: Config, device: torch.device, args: argparse.Namespace) -> None:
    log = get_logger("train", log_dir=cfg.logs_dir)

    # Override cfg from CLI
    if args.epochs:     cfg.num_epochs = args.epochs
    if args.lr:         cfg.learning_rate = args.lr
    if args.batch_size: cfg.batch_size = args.batch_size

    log.info(f"Device: {device_info(device)}")
    log.info(f"Epochs: {cfg.num_epochs}, LR: {cfg.learning_rate}, "
             f"Batch: {cfg.batch_size}")

    # Data
    train_loader, val_loader = build_dataloaders(cfg)
    log.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = get_model(cfg.model_name, cfg.in_channels, cfg.out_channels).to(device)
    print_model_summary(model, (cfg.in_channels, *cfg.image_size), device)

    # Optimiser, scheduler, loss
    optimiser = Adam(model.parameters(), lr=cfg.learning_rate,
                     weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimiser, mode="max", patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    # CSV logger
    csv_log = CSVLogger(
        filepath=f"{cfg.csv_dir}/train_metrics.csv",
        fieldnames=["epoch", "train_loss", "val_loss",
                    "val_dice", "val_iou", "val_pixel_acc",
                    "val_sensitivity", "val_specificity", "lr"],
    )

    best_dice = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    best_ckpt = f"{cfg.checkpoints_dir}/best_{cfg.model_name}.pth"
    last_ckpt = f"{cfg.checkpoints_dir}/last_{cfg.model_name}.pth"

    for epoch in range(1, cfg.num_epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        train_acc = MetricAccumulator()

        for step, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)
            optimiser.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()

            preds = (torch.sigmoid(logits) >= args.threshold).float()
            train_acc.update(compute_all_metrics(preds, masks))

            if step % cfg.log_interval == 0:
                log.info(
                    f"Epoch {epoch}/{cfg.num_epochs} | Step {step}/{len(train_loader)} "
                    f"| Loss {loss.item():.4f}"
                )

        train_loss /= len(train_loader)
        train_metrics = train_acc.mean()

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_acc = MetricAccumulator()

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_loss += criterion(logits, masks).item()
                preds = (torch.sigmoid(logits) >= args.threshold).float()
                val_acc.update(compute_all_metrics(preds, masks))

        val_loss /= len(val_loader)
        val_metrics = val_acc.mean()
        val_dice = val_metrics["dice"]

        scheduler.step(val_dice)
        current_lr = optimiser.param_groups[0]["lr"]

        # Logging
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_metrics["dice"])
        val_dices.append(val_dice)

        log.info(
            f"Epoch {epoch:04d} | "
            f"TrLoss {train_loss:.4f} | VaLoss {val_loss:.4f} | "
            f"VaDice {val_dice:.4f} | VaIoU {val_metrics['iou']:.4f} | "
            f"LR {current_lr:.2e}"
        )

        csv_log.log({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            **{f"val_{k}": round(v, 6) for k, v in val_metrics.items()},
            "lr": current_lr,
        })

        # Checkpoint
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            patience_counter = 0
        else:
            patience_counter += 1

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "val_dice": val_dice,
            "cfg": cfg.__dict__,
        }
        save_checkpoint(state, last_ckpt,
                        is_best=is_best, best_filepath=best_ckpt)

        # Early stopping
        if patience_counter >= cfg.early_stopping_patience:
            log.info(f"Early stopping triggered at epoch {epoch}.")
            break

    log.info(f"Training complete. Best Val Dice: {best_dice:.4f}")

    # Plots
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
    inf = SegmentationInference(model, device, threshold=args.threshold)

    _, val_loader = build_dataloaders(cfg)
    acc = MetricAccumulator()

    model.eval()
    all_images, all_gt, all_pred = [], [], []

    with torch.no_grad():
        for images, masks in val_loader:
            probs, preds = inf.predict(images)
            acc.update(compute_all_metrics(preds, masks))
            if len(all_images) < 4:
                all_images.append(images)
                all_gt.append(masks)
                all_pred.append(preds)

    metrics = acc.mean()
    log.info("Evaluation results:")
    for k, v in metrics.items():
        log.info(f"  {k:<20} {v:.6f}")

    # Save CSV
    from utils.logger import CSVLogger
    csv_log = CSVLogger(
        f"{cfg.csv_dir}/eval_results.csv",
        fieldnames=list(metrics.keys()),
    )
    csv_log.log(metrics)

    # Qualitative plots
    if not args.no_plot and all_images:
        import torch
        plot_predictions(
            torch.cat(all_images),
            torch.cat(all_gt),
            torch.cat(all_pred),
            cfg, n=min(4, len(all_images)),
        )


# --------------------------------------------------------------------------- #
#  Benchmark                                                                    #
# --------------------------------------------------------------------------- #

def run_benchmark(cfg: Config, device: torch.device, args: argparse.Namespace) -> None:
    model = load_model(cfg, checkpoint_path=args.checkpoint, device=device)
    bench = Benchmarker(cfg, device)
    results = bench.run(model, label=cfg.model_name)
    Benchmarker.print_results(results)

    from utils.logger import CSVLogger
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
    cfg = Config()

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
