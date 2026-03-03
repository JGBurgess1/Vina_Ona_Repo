"""
Visualization for docking parameter optimization results.

Generates:
  - ROC curve overlays (top N parameter sets)
  - Score distributions (actives vs decoys)
  - Parameter sensitivity plots (metric vs parameter value)
  - Enrichment factor comparison bar charts
  - Summary heatmap of all metrics across parameter sets
"""

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def plot_roc_overlay(results: list, output_dir: str, top_n: int = 10) -> None:
    """
    Overlay ROC curves for the top N parameter sets.
    Also plots the random baseline.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Random baseline
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.500)")

    # Plot top N
    n_plot = min(top_n, len(results))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, n_plot))

    for i, result in enumerate(results[:n_plot]):
        m = result.metrics
        if m.fpr is not None and m.tpr is not None:
            ax.plot(
                m.fpr, m.tpr,
                color=cmap[i],
                linewidth=1.5 if i == 0 else 1.0,
                alpha=1.0 if i == 0 else 0.6,
                label=f"#{i+1} AUC={m.roc_auc:.3f} | {result.params.label[:40]}",
            )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Top Parameter Configurations")
    ax.legend(fontsize=6, loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    path = os.path.join(output_dir, "roc_overlay.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved ROC overlay to %s", path)


def plot_roc_semilog(results: list, output_dir: str, top_n: int = 5) -> None:
    """
    Semi-log ROC plot (log10 x-axis) emphasizing early enrichment.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    n_plot = min(top_n, len(results))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, n_plot))

    for i, result in enumerate(results[:n_plot]):
        m = result.metrics
        if m.fpr is not None and m.tpr is not None:
            # Filter out FPR=0 for log scale
            mask = m.fpr > 0
            ax.semilogx(
                m.fpr[mask], m.tpr[mask],
                color=cmap[i],
                linewidth=1.5 if i == 0 else 1.0,
                label=f"#{i+1} LogAUC={m.log_auc:.3f}",
            )

    # Random baseline on log scale
    x = np.logspace(-3, 0, 100)
    ax.semilogx(x, x, "k--", alpha=0.4, label="Random")

    ax.set_xlabel("False Positive Rate (log scale)")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Semi-log ROC (Early Enrichment)")
    ax.legend(fontsize=7)
    ax.set_xlim([1e-3, 1])
    ax.set_ylim([0, 1.05])

    path = os.path.join(output_dir, "roc_semilog.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved semi-log ROC to %s", path)


def plot_score_distributions(results: list, output_dir: str, top_n: int = 4) -> None:
    """
    Histogram of docking scores for actives vs decoys for top parameter sets.
    Good separation = good discrimination.
    """
    n_plot = min(top_n, len(results))
    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4), sharey=True)
    if n_plot == 1:
        axes = [axes]

    for i, (ax, result) in enumerate(zip(axes, results[:n_plot])):
        bins = np.linspace(
            min(result.active_scores.min(), result.decoy_scores.min()) - 0.5,
            max(result.active_scores.max(), result.decoy_scores.max()) + 0.5,
            40,
        )
        ax.hist(
            result.active_scores, bins=bins, alpha=0.6,
            color="#4CAF50", label=f"Actives (n={len(result.active_scores)})",
            edgecolor="black", linewidth=0.5,
        )
        ax.hist(
            result.decoy_scores, bins=bins, alpha=0.6,
            color="#F44336", label=f"Decoys (n={len(result.decoy_scores)})",
            edgecolor="black", linewidth=0.5,
        )
        ax.set_xlabel("Docking Score (kcal/mol)")
        if i == 0:
            ax.set_ylabel("Count")
        ax.set_title(f"#{i+1}: AUC={result.metrics.roc_auc:.3f}", fontsize=10)
        ax.legend(fontsize=7)

    fig.suptitle("Active vs Decoy Score Distributions", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "score_distributions.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved score distributions to %s", path)


def plot_enrichment_bars(results: list, output_dir: str, top_n: int = 10) -> None:
    """Bar chart comparing EF1%, EF5%, EF10% across top parameter sets."""
    n_plot = min(top_n, len(results))

    labels = [f"#{i+1}" for i in range(n_plot)]
    ef1 = [r.metrics.ef_1pct for r in results[:n_plot]]
    ef5 = [r.metrics.ef_5pct for r in results[:n_plot]]
    ef10 = [r.metrics.ef_10pct for r in results[:n_plot]]

    x = np.arange(n_plot)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n_plot), 5))
    ax.bar(x - width, ef1, width, label="EF 1%", color="#2196F3")
    ax.bar(x, ef5, width, label="EF 5%", color="#FF9800")
    ax.bar(x + width, ef10, width, label="EF 10%", color="#9C27B0")

    ax.set_xlabel("Parameter Set Rank")
    ax.set_ylabel("Enrichment Factor")
    ax.set_title("Enrichment Factors by Parameter Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Random (EF=1)")

    plt.tight_layout()
    path = os.path.join(output_dir, "enrichment_factors.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved enrichment factor bars to %s", path)


def plot_parameter_sensitivity(results: list, output_dir: str) -> None:
    """
    Scatter plots showing how each parameter affects the target metric.
    One subplot per parameter dimension.
    """
    if len(results) < 3:
        logger.info("Too few results for sensitivity plots, skipping")
        return

    # Extract parameter values
    box_x = [r.params.box_size[0] for r in results]
    box_y = [r.params.box_size[1] for r in results]
    box_z = [r.params.box_size[2] for r in results]
    cx = [r.params.center[0] for r in results]
    cy = [r.params.center[1] for r in results]
    cz = [r.params.center[2] for r in results]
    exh = [r.params.exhaustiveness for r in results]
    auc = [r.metrics.roc_auc for r in results]
    bedroc = [r.metrics.bedroc for r in results]

    params_data = [
        ("Box Size X", box_x),
        ("Box Size Y", box_y),
        ("Box Size Z", box_z),
        ("Center X", cx),
        ("Center Y", cy),
        ("Center Z", cz),
        ("Exhaustiveness", exh),
    ]

    # Only plot dimensions that actually vary
    params_data = [(name, vals) for name, vals in params_data if len(set(vals)) > 1]

    if not params_data:
        logger.info("No varying parameters for sensitivity plots, skipping")
        return

    n = len(params_data)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, (name, vals) in enumerate(params_data):
        # AUC
        axes[0, j].scatter(vals, auc, alpha=0.6, s=20, c="#2196F3")
        axes[0, j].set_xlabel(name)
        axes[0, j].set_ylabel("ROC AUC")
        axes[0, j].set_title(f"AUC vs {name}")

        # BEDROC
        axes[1, j].scatter(vals, bedroc, alpha=0.6, s=20, c="#FF9800")
        axes[1, j].set_xlabel(name)
        axes[1, j].set_ylabel("BEDROC")
        axes[1, j].set_title(f"BEDROC vs {name}")

    plt.suptitle("Parameter Sensitivity Analysis", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "parameter_sensitivity.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved parameter sensitivity plots to %s", path)


def plot_metrics_heatmap(results: list, output_dir: str, top_n: int = 20) -> None:
    """Heatmap of all metrics across top parameter sets."""
    n_plot = min(top_n, len(results))

    data = []
    for i, r in enumerate(results[:n_plot]):
        data.append({
            "Config": f"#{i+1}",
            "ROC AUC": r.metrics.roc_auc,
            "LogAUC": r.metrics.log_auc,
            "BEDROC": r.metrics.bedroc,
            "EF 1%": r.metrics.ef_1pct,
            "EF 5%": r.metrics.ef_5pct,
            "EF 10%": r.metrics.ef_10pct,
        })

    df = pd.DataFrame(data).set_index("Config")

    fig, ax = plt.subplots(figsize=(8, max(4, n_plot * 0.4)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGn", ax=ax)
    ax.set_title("Metrics Summary: Top Parameter Configurations")

    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved metrics heatmap to %s", path)


def generate_all_plots(results: list, output_dir: str) -> None:
    """Generate the full suite of optimization plots."""
    os.makedirs(output_dir, exist_ok=True)

    plot_roc_overlay(results, output_dir)
    plot_roc_semilog(results, output_dir)
    plot_score_distributions(results, output_dir)
    plot_enrichment_bars(results, output_dir)
    plot_parameter_sensitivity(results, output_dir)
    plot_metrics_heatmap(results, output_dir)

    logger.info("All optimization plots saved to %s", output_dir)
