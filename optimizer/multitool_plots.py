"""
Visualization for multi-tool parameter optimization.

Extends the single-tool optimization plots with cross-tool comparisons:
  - Per-tool ROC overlays (best params per tool)
  - Cross-tool metric comparison heatmap
  - Best-params-per-tool bar chart
  - Per-tool score distributions
"""

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .optimization_plots import (
    generate_all_plots as generate_single_tool_plots,
)

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def generate_multitool_plots(
    results_by_tool: dict,
    output_dir: str,
) -> None:
    """
    Generate all plots for multi-tool optimization.

    Args:
        results_by_tool: dict of backend_name -> list[ValidationResult]
        output_dir: base output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Per-tool plots in subdirectories
    for tool_name, results in results_by_tool.items():
        if results:
            tool_dir = os.path.join(output_dir, tool_name)
            generate_single_tool_plots(results, tool_dir)

    # Cross-tool comparison plots
    plot_cross_tool_roc(results_by_tool, output_dir)
    plot_cross_tool_metrics(results_by_tool, output_dir)
    plot_best_params_comparison(results_by_tool, output_dir)
    plot_cross_tool_score_distributions(results_by_tool, output_dir)

    logger.info("All multi-tool plots saved to %s", output_dir)


def plot_cross_tool_roc(results_by_tool: dict, output_dir: str) -> None:
    """
    Overlay the best ROC curve from each tool on a single plot.
    Shows which tool achieves the best discrimination with its optimal params.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.500)")

    colors = plt.cm.Set1(np.linspace(0, 0.8, len(results_by_tool)))

    for (tool_name, results), color in zip(results_by_tool.items(), colors):
        if not results:
            continue
        best = results[0]  # already sorted, best first
        m = best.metrics
        if m.fpr is not None and m.tpr is not None:
            ax.plot(
                m.fpr, m.tpr,
                color=color, linewidth=2.0,
                label=f"{tool_name}: AUC={m.roc_auc:.3f}, BEDROC={m.bedroc:.3f}",
            )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Best ROC Curve Per Docking Tool (Optimized Parameters)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    path = os.path.join(output_dir, "cross_tool_roc.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved cross-tool ROC to %s", path)


def plot_cross_tool_metrics(results_by_tool: dict, output_dir: str) -> None:
    """
    Heatmap comparing the best metrics achieved by each tool.
    """
    rows = []
    for tool_name, results in results_by_tool.items():
        if not results:
            continue
        best = results[0]
        rows.append({
            "Tool": tool_name,
            "ROC AUC": best.metrics.roc_auc,
            "LogAUC": best.metrics.log_auc,
            "BEDROC": best.metrics.bedroc,
            "EF 1%": best.metrics.ef_1pct,
            "EF 5%": best.metrics.ef_5pct,
            "EF 10%": best.metrics.ef_10pct,
        })

    if not rows:
        return

    df = pd.DataFrame(rows).set_index("Tool")

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.8)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGn", ax=ax)
    ax.set_title("Best Metrics Per Tool (Optimized Parameters)")
    plt.tight_layout()
    path = os.path.join(output_dir, "cross_tool_metrics.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved cross-tool metrics heatmap to %s", path)


def plot_best_params_comparison(results_by_tool: dict, output_dir: str) -> None:
    """
    Bar chart comparing the optimal parameters found for each tool.
    """
    tools = []
    box_sizes = []
    exhaustiveness = []
    aucs = []

    for tool_name, results in results_by_tool.items():
        if not results:
            continue
        best = results[0]
        tools.append(tool_name)
        box_sizes.append(best.params.box_size[0])  # assume cubic
        exhaustiveness.append(best.params.exhaustiveness)
        aucs.append(best.metrics.roc_auc)

    if not tools:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = range(len(tools))
    colors = plt.cm.Set2(np.linspace(0, 1, len(tools)))

    axes[0].bar(x, box_sizes, color=colors, edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tools)
    axes[0].set_ylabel("Box Size (Å)")
    axes[0].set_title("Optimal Box Size")

    axes[1].bar(x, exhaustiveness, color=colors, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tools)
    axes[1].set_ylabel("Exhaustiveness")
    axes[1].set_title("Optimal Exhaustiveness")

    axes[2].bar(x, aucs, color=colors, edgecolor="black")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tools)
    axes[2].set_ylabel("ROC AUC")
    axes[2].set_title("Best AUC Achieved")
    axes[2].set_ylim([0, 1])

    fig.suptitle("Optimal Parameters Per Tool", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "best_params_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved best params comparison to %s", path)


def plot_cross_tool_score_distributions(
    results_by_tool: dict,
    output_dir: str,
) -> None:
    """
    Active vs decoy score distributions for the best params of each tool.
    """
    tools_with_results = [
        (name, results[0])
        for name, results in results_by_tool.items()
        if results
    ]

    if not tools_with_results:
        return

    n = len(tools_with_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for i, (tool_name, best) in enumerate(tools_with_results):
        ax = axes[i]
        bins = np.linspace(
            min(best.active_scores.min(), best.decoy_scores.min()) - 0.5,
            max(best.active_scores.max(), best.decoy_scores.max()) + 0.5,
            40,
        )
        ax.hist(best.active_scores, bins=bins, alpha=0.6, color="#4CAF50",
                label=f"Actives (n={len(best.active_scores)})", edgecolor="black", linewidth=0.5)
        ax.hist(best.decoy_scores, bins=bins, alpha=0.6, color="#F44336",
                label=f"Decoys (n={len(best.decoy_scores)})", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Docking Score")
        if i == 0:
            ax.set_ylabel("Count")
        ax.set_title(f"{tool_name}: AUC={best.metrics.roc_auc:.3f}")
        ax.legend(fontsize=7)

    fig.suptitle("Active vs Decoy Distributions (Best Parameters Per Tool)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "cross_tool_score_distributions.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved cross-tool score distributions to %s", path)
