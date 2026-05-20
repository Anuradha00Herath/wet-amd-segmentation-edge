"""
pruning_visualization.py
------------------------
Publication-quality plots for pruning experiments.

Generates:
  1. Pruning ratio vs Dice score
  2. Pruning ratio vs Latency
  3. Pruning ratio vs Model size
  4. Pruning ratio vs Energy (estimated)
  5. Pareto frontier: accuracy vs efficiency
  6. Full model comparison bar chart
  7. Progressive pruning training curves
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.config import Config


def plot_pruning_accuracy_tradeoff(
    results: List[Dict[str, Any]],
    cfg: Config,
    filename: str = "pruning_accuracy_tradeoff.png",
    show: bool = False,
) -> None:
    """
    Line plot: pruning ratio vs Dice score and IoU.
    """
    ratios = [r["prune_ratio"] for r in results if "prune_ratio" in r]
    dices  = [r["mean_dice"]   for r in results if "prune_ratio" in r]
    ious   = [r.get("mean_iou", 0) for r in results if "prune_ratio" in r]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([r*100 for r in ratios], dices, "b-o", linewidth=2, markersize=7, label="Dice Score")
    ax.plot([r*100 for r in ratios], ious,  "g-s", linewidth=2, markersize=7, label="IoU")

    ax.set_xlabel("Pruning Ratio (%)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Pruning Ratio vs Segmentation Accuracy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    for x, y in zip([r*100 for r in ratios], dices):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")

    plt.tight_layout()
    _save(fig, cfg, filename, show)


def plot_pruning_efficiency(
    results: List[Dict[str, Any]],
    cfg: Config,
    filename: str = "pruning_efficiency.png",
    show: bool = False,
) -> None:
    """
    3-panel: pruning ratio vs latency, size, FPS.
    """
    pruned = [r for r in results if "prune_ratio" in r]
    ratios = [r["prune_ratio"] * 100 for r in pruned]
    ms_vals = [r.get("mean_ms", 0)  for r in pruned]
    sizes   = [r.get("size_mb", 0)  for r in pruned]
    fps_v   = [r.get("fps", 0)      for r in pruned]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(ratios, ms_vals, "r-o", linewidth=2, markersize=7)
    axes[0].set_xlabel("Pruning Ratio (%)"); axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Latency", fontweight="bold"); axes[0].grid(alpha=0.3)

    axes[1].plot(ratios, sizes, "b-o", linewidth=2, markersize=7)
    axes[1].set_xlabel("Pruning Ratio (%)"); axes[1].set_ylabel("Model Size (MB)")
    axes[1].set_title("Model Size", fontweight="bold"); axes[1].grid(alpha=0.3)

    axes[2].plot(ratios, fps_v, "g-o", linewidth=2, markersize=7)
    axes[2].set_xlabel("Pruning Ratio (%)"); axes[2].set_ylabel("FPS")
    axes[2].set_title("Throughput", fontweight="bold"); axes[2].grid(alpha=0.3)

    plt.suptitle("Pruning Ratio vs Efficiency Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, cfg, filename, show)


def plot_compression_comparison(
    all_results: Dict[str, Dict[str, float]],
    cfg: Config,
    filename: str = "compression_comparison.png",
    show: bool = False,
) -> None:
    """
    Grouped bar chart comparing:
    FP32 | PTQ INT8 | QAT INT8 | Pruned 30% | Pruned+PTQ

    Args:
        all_results: Dict of model_name → {mean_dice, mean_ms, size_mb, fps}
    """
    names  = list(all_results.keys())
    dices  = [all_results[n].get("mean_dice", 0) for n in names]
    sizes  = [all_results[n].get("size_mb", 0)   for n in names]
    fps_v  = [all_results[n].get("fps", 0)        for n in names]
    x      = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#3498db", "#e74c3c", "#27ae60", "#f39c12", "#9b59b6", "#1abc9c"]

    for ax, vals, ylabel, title in zip(
        axes,
        [dices, sizes, fps_v],
        ["Mean Dice", "Model Size (MB)", "FPS"],
        ["Accuracy", "Model Size", "Throughput"],
    ):
        bars = ax.bar(x, vals, color=colors[:len(names)], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.bar_label(bars, fmt="%.3f" if "Dice" in ylabel else "%.1f", fontsize=8, padding=2)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Model Compression Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, cfg, filename, show)


def plot_pareto_frontier(
    results: List[Dict[str, Any]],
    cfg: Config,
    x_metric: str = "mean_ms",
    y_metric: str = "mean_dice",
    filename: str = "pruning_pareto_frontier.png",
    show: bool = False,
) -> None:
    """Pareto frontier: accuracy vs latency for all compression methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "fp32": "#3498db", "ptq": "#e74c3c", "qat": "#27ae60",
        "pruned": "#f39c12", "pruned_ptq": "#9b59b6",
    }

    for r in results:
        label = r.get("label", "")
        color = next((v for k, v in colors.items() if k in label.lower()), "#95a5a6")
        ax.scatter(r.get(x_metric, 0), r.get(y_metric, 0),
                   s=120, c=color, zorder=5, alpha=0.85)
        ax.annotate(label, (r.get(x_metric, 0), r.get(y_metric, 0)),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)

    # Draw Pareto frontier
    pareto = _get_pareto(results, x_metric, y_metric)
    if len(pareto) > 1:
        px = [r[x_metric] for r in sorted(pareto, key=lambda r: r[x_metric])]
        py = [r[y_metric] for r in sorted(pareto, key=lambda r: r[x_metric])]
        ax.plot(px, py, "r--", linewidth=1.5, alpha=0.7, label="Pareto frontier")

    x_label = {"mean_ms": "Latency (ms)", "size_mb": "Model Size (MB)"}.get(x_metric, x_metric)
    y_label = {"mean_dice": "Mean Dice"}.get(y_metric, y_metric)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title("Pareto Frontier: Accuracy vs Efficiency", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, cfg, filename, show)


def plot_progressive_training_curves(
    history: List[Dict],
    cfg: Config,
    filename: str = "progressive_pruning_curves.png",
    show: bool = False,
) -> None:
    """Plot fine-tuning training curves for progressive pruning steps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    steps  = list(set(h.get("step", 1) for h in history))
    colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))

    for step, color in zip(steps, colors):
        step_data = [h for h in history if h.get("step", 1) == step]
        epochs    = [h["epoch"] for h in step_data]
        dices     = [h["val_dice"] for h in step_data]
        losses    = [h["train_loss"] for h in step_data]

        axes[0].plot(epochs, dices,  color=color, linewidth=2, label=f"Step {step}")
        axes[1].plot(epochs, losses, color=color, linewidth=2, label=f"Step {step}")

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Dice Score")
    axes[0].set_title("Validation Dice", fontweight="bold"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss", fontweight="bold"); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.suptitle("Progressive Pruning Fine-tuning Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, cfg, filename, show)


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _get_pareto(results, x_metric, y_metric):
    pareto = []
    for r in results:
        dominated = any(
            other[x_metric] <= r[x_metric] and other[y_metric] >= r[y_metric]
            and (other[x_metric] < r[x_metric] or other[y_metric] > r[y_metric])
            for other in results if other is not r
        )
        if not dominated:
            pareto.append(r)
    return pareto


def _save(fig, cfg: Config, filename: str, show: bool) -> None:
    os.makedirs(cfg.plots_dir, exist_ok=True)
    path = os.path.join(cfg.plots_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[viz] Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)