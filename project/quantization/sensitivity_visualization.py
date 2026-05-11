"""
sensitivity_visualization.py
-----------------------------
Publication-quality plots for sensitivity analysis and mixed precision results.

Reuses:
  - evaluation.visualization._save (internal helper pattern)
  - utils.config.Config

All plots saved to cfg.plots_dir automatically.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.config import Config


# --------------------------------------------------------------------------- #
#  Sensitivity bar chart                                                       #
# --------------------------------------------------------------------------- #

def plot_sensitivity_ranking(
    results: List[Dict[str, Any]],
    cfg: Config,
    filename: str = "sensitivity_ranking.png",
    show: bool = False,
) -> None:
    """
    Horizontal bar chart: sensitivity (Dice recovery) per layer group.

    Args:
        results:  Output of LayerSensitivityAnalyser.run().
        cfg:      Config.
        filename: Output filename.
        show:     Display inline in Colab.
    """
    groups = [r["group"] for r in results]
    sens   = [r["sensitivity"] for r in results]
    dices  = [r["dice_with_fp32"] for r in results]

    colors = ["#e74c3c" if s > 0.01 else "#f39c12" if s > 0.005
              else "#27ae60" for s in sens]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(groups) * 0.8)))

    # Left: sensitivity
    bars = axes[0].barh(groups, sens, color=colors)
    axes[0].set_xlabel("Sensitivity (Dice recovery when kept FP32)")
    axes[0].set_title("Layer Sensitivity Ranking", fontweight="bold")
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    for bar, s in zip(bars, sens):
        axes[0].text(
            max(s + 0.001, 0.001), bar.get_y() + bar.get_height() / 2,
            f"{s:+.4f}", va="center", fontsize=9
        )

    # Right: absolute Dice with FP32 group
    axes[1].barh(groups, dices, color="#3498db")
    axes[1].set_xlabel("Dice Score (group kept FP32)")
    axes[1].set_title("Dice Score by Group", fontweight="bold")
    axes[1].set_xlim(0, 1)
    for bar, d in zip(axes[1].patches, dices):
        axes[1].text(
            d + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{d:.4f}", va="center", fontsize=9
        )

    # Legend
    legend_elements = [
        mpatches.Patch(color="#e74c3c", label="High sensitivity (>0.01)"),
        mpatches.Patch(color="#f39c12", label="Medium sensitivity (0.005–0.01)"),
        mpatches.Patch(color="#27ae60", label="Low sensitivity (<0.005)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    _save(fig, cfg, filename, show)


# --------------------------------------------------------------------------- #
#  Mixed precision comparison chart                                            #
# --------------------------------------------------------------------------- #

def plot_mixed_precision_comparison(
    results: List[Dict[str, Any]],
    cfg: Config,
    filename: str = "mixed_precision_comparison.png",
    show: bool = False,
) -> None:
    """
    Grouped bar chart comparing all mixed precision experiments.

    Args:
        results:  Output of MixedPrecisionPipeline.run_experiments().
        cfg:      Config.
        filename: Output filename.
        show:     Display inline.
    """
    names = [r["experiment"] for r in results]
    dices = [r["mean_dice"]  for r in results]
    sizes = [r["size_mb"]    for r in results]
    fps   = [r["fps"]        for r in results]

    x     = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Dice
    bars = axes[0].bar(x, dices, color="#3498db", alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_ylabel("Mean Dice"); axes[0].set_title("Accuracy", fontweight="bold")
    axes[0].set_ylim(max(0, min(dices) - 0.05), 1.0)
    axes[0].bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    axes[0].grid(axis="y", alpha=0.3)

    # Model size
    bars = axes[1].bar(x, sizes, color="#e74c3c", alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_ylabel("Model Size (MB)"); axes[1].set_title("Model Size", fontweight="bold")
    axes[1].bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    axes[1].grid(axis="y", alpha=0.3)

    # FPS
    bars = axes[2].bar(x, fps, color="#27ae60", alpha=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=30, ha="right")
    axes[2].set_ylabel("FPS"); axes[2].set_title("Throughput", fontweight="bold")
    axes[2].bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle("Mixed Precision Experiment Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, cfg, filename, show)


# --------------------------------------------------------------------------- #
#  Pareto frontier plot                                                        #
# --------------------------------------------------------------------------- #

def plot_pareto_frontier(
    results: List[Dict[str, Any]],
    pareto_results: List[Dict[str, Any]],
    cfg: Config,
    x_metric: str = "mean_ms",
    y_metric: str = "mean_dice",
    filename: str = "pareto_frontier.png",
    show: bool = False,
) -> None:
    """
    Scatter plot with Pareto frontier highlighted.

    Args:
        results:        All experiment results.
        pareto_results: Pareto-optimal subset.
        cfg:            Config.
        x_metric:       X-axis metric (minimise).
        y_metric:       Y-axis metric (maximise).
        filename:       Output filename.
        show:           Display inline.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x_all  = [r[x_metric] for r in results]
    y_all  = [r[y_metric] for r in results]
    names  = [r["experiment"] for r in results]

    pareto_names = {r["experiment"] for r in pareto_results}

    # Plot all points
    for x, y, name in zip(x_all, y_all, names):
        color  = "#e74c3c" if name in pareto_names else "#95a5a6"
        marker = "★" if name in pareto_names else "o"
        ax.scatter(x, y, c=color, s=120, zorder=5)
        ax.annotate(
            name, (x, y),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, ha="left",
        )

    # Draw Pareto frontier line
    if len(pareto_results) > 1:
        px = sorted([r[x_metric] for r in pareto_results])
        py = [r[y_metric] for r in sorted(pareto_results, key=lambda r: r[x_metric])]
        ax.plot(px, py, "r--", linewidth=1.5, alpha=0.7, label="Pareto frontier")

    x_label = {"mean_ms": "Latency (ms)", "size_mb": "Model Size (MB)"}.get(x_metric, x_metric)
    y_label = {"mean_dice": "Mean Dice", "mean_iou": "Mean IoU"}.get(y_metric, y_metric)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title("Pareto Frontier: Accuracy vs Efficiency", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    legend_elements = [
        mpatches.Patch(color="#e74c3c", label="Pareto-optimal"),
        mpatches.Patch(color="#95a5a6", label="Dominated"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    _save(fig, cfg, filename, show)


# --------------------------------------------------------------------------- #
#  Accuracy vs compression scatter                                             #
# --------------------------------------------------------------------------- #

def plot_accuracy_vs_compression(
    results: List[Dict[str, Any]],
    cfg: Config,
    filename: str = "accuracy_vs_compression.png",
    show: bool = False,
) -> None:
    """
    Scatter: compression ratio (x) vs Dice score (y) for all experiments.
    Bubble size = FPS (larger = faster).
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for r in results:
        ax.scatter(
            r["compression"], r["mean_dice"],
            s=r["fps"] * 3,
            alpha=0.7, edgecolors="black", linewidths=0.5,
        )
        ax.annotate(
            r["experiment"],
            (r["compression"], r["mean_dice"]),
            textcoords="offset points", xytext=(5, 3), fontsize=8,
        )

    ax.set_xlabel("Compression Ratio (×)", fontsize=11)
    ax.set_ylabel("Mean Dice Score", fontsize=11)
    ax.set_title("Accuracy vs Compression (bubble size = FPS)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, cfg, filename, show)


# --------------------------------------------------------------------------- #
#  FP32 / PTQ / QAT / Mixed comparison radar                                  #
# --------------------------------------------------------------------------- #

def plot_full_model_comparison(
    model_results: Dict[str, Dict[str, float]],
    metrics: List[str],
    cfg: Config,
    filename: str = "full_model_comparison_radar.png",
    show: bool = False,
) -> None:
    """
    Radar (spider) chart comparing all model variants across metrics.

    Args:
        model_results: Dict of model_name → {metric: value}.
        metrics:       List of metric names to include.
        cfg:           Config.
        filename:      Output filename.
        show:          Display inline.
    """
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    colors  = ["#3498db", "#e74c3c", "#27ae60", "#f39c12", "#9b59b6"]

    for (model_name, scores), color in zip(model_results.items(), colors):
        values  = [scores.get(m, 0) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=model_name)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [m.replace("_", "\n") for m in metrics], size=9
    )
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    _save(fig, cfg, filename, show)


# --------------------------------------------------------------------------- #
#  Internal helper                                                             #
# --------------------------------------------------------------------------- #

def _save(fig, cfg: Config, filename: str, show: bool) -> None:
    os.makedirs(cfg.plots_dir, exist_ok=True)
    path = os.path.join(cfg.plots_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[viz] Saved → {path}")
    if show:
        plt.show()
    plt.close(fig)
