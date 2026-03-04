"""
Visualization for consensus docking results.

Generates:
  - Rank correlation heatmap between tools
  - Score distribution per tool (overlaid histograms)
  - Consensus method comparison (rank correlation between methods)
  - Top hits Venn-style overlap analysis
  - Scatter plots of pairwise tool scores
"""

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def generate_all_plots(
    consensus_df: pd.DataFrame,
    score_matrix: pd.DataFrame,
    output_dir: str,
) -> None:
    """Generate the full suite of consensus docking plots."""
    os.makedirs(output_dir, exist_ok=True)

    plot_rank_correlation_heatmap(score_matrix, output_dir)
    plot_score_distributions(score_matrix, output_dir)
    plot_pairwise_scatter(score_matrix, output_dir)
    plot_consensus_method_comparison(consensus_df, output_dir)
    plot_top_hits_overlap(score_matrix, output_dir)
    plot_rank_stability(consensus_df, score_matrix, output_dir)

    logger.info("All consensus plots saved to %s", output_dir)


def plot_rank_correlation_heatmap(score_matrix: pd.DataFrame, output_dir: str) -> None:
    """
    Heatmap of Spearman rank correlations between all pairs of docking tools.
    High correlation = tools agree on relative ranking of compounds.
    """
    tools = score_matrix.columns.tolist()
    n = len(tools)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Use only ligands that succeeded in both tools
            mask = score_matrix[[tools[i], tools[j]]].notna().all(axis=1)
            if mask.sum() < 3:
                corr_matrix[i, j] = np.nan
            else:
                corr_matrix[i, j], _ = stats.spearmanr(
                    score_matrix.loc[mask, tools[i]],
                    score_matrix.loc[mask, tools[j]],
                )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pd.DataFrame(corr_matrix, index=tools, columns=tools),
        annot=True, fmt=".3f", cmap="RdYlGn", vmin=-1, vmax=1,
        ax=ax,
    )
    ax.set_title("Spearman Rank Correlation Between Docking Tools")
    plt.tight_layout()
    path = os.path.join(output_dir, "rank_correlation_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved rank correlation heatmap to %s", path)


def plot_score_distributions(score_matrix: pd.DataFrame, output_dir: str) -> None:
    """Overlaid histograms of raw scores from each tool."""
    tools = score_matrix.columns.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(tools)))
    for tool, color in zip(tools, colors):
        scores = score_matrix[tool].dropna()
        ax.hist(
            scores, bins=50, alpha=0.5, color=color, edgecolor="black",
            linewidth=0.5, label=f"{tool} (n={len(scores)}, μ={scores.mean():.2f})",
        )

    ax.set_xlabel("Docking Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distributions by Docking Tool")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "score_distributions.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved score distributions to %s", path)


def plot_pairwise_scatter(score_matrix: pd.DataFrame, output_dir: str) -> None:
    """Pairwise scatter plots of scores between all tool pairs."""
    tools = score_matrix.columns.tolist()
    n = len(tools)
    if n < 2:
        return

    n_pairs = n * (n - 1) // 2
    cols = min(3, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            ax = axes[pair_idx]
            mask = score_matrix[[tools[i], tools[j]]].notna().all(axis=1)
            x = score_matrix.loc[mask, tools[i]]
            y = score_matrix.loc[mask, tools[j]]

            ax.scatter(x, y, alpha=0.3, s=8, edgecolors="none")
            if len(x) > 2:
                r, p = stats.spearmanr(x, y)
                ax.set_title(f"{tools[i]} vs {tools[j]}\nρ={r:.3f}", fontsize=10)
            else:
                ax.set_title(f"{tools[i]} vs {tools[j]}")
            ax.set_xlabel(tools[i])
            ax.set_ylabel(tools[j])
            pair_idx += 1

    for k in range(pair_idx, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle("Pairwise Score Comparison", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "pairwise_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved pairwise scatter to %s", path)


def plot_consensus_method_comparison(
    consensus_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Heatmap of Spearman correlations between consensus methods.
    Shows how much the different consensus strategies agree.
    """
    methods = ["avg_rank", "z_score_avg", "ecr_score", "best_of_n", "vote_count"]
    labels = ["Avg Rank", "Z-Score Avg", "ECR", "Best-of-N", "Majority Vote"]

    available = [m for m in methods if m in consensus_df.columns]
    available_labels = [labels[methods.index(m)] for m in available]

    if len(available) < 2:
        return

    corr_matrix = np.zeros((len(available), len(available)))
    for i, m1 in enumerate(available):
        for j, m2 in enumerate(available):
            mask = consensus_df[[m1, m2]].notna().all(axis=1)
            if mask.sum() > 2:
                corr_matrix[i, j], _ = stats.spearmanr(
                    consensus_df.loc[mask, m1],
                    consensus_df.loc[mask, m2],
                )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pd.DataFrame(corr_matrix, index=available_labels, columns=available_labels),
        annot=True, fmt=".3f", cmap="RdYlGn", vmin=-1, vmax=1, ax=ax,
    )
    ax.set_title("Correlation Between Consensus Methods")
    plt.tight_layout()
    path = os.path.join(output_dir, "consensus_method_correlation.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved consensus method correlation to %s", path)


def plot_top_hits_overlap(
    score_matrix: pd.DataFrame,
    output_dir: str,
    top_fraction: float = 0.05,
) -> None:
    """
    Bar chart showing overlap of top X% hits between tools.
    For each pair of tools, shows the fraction of top hits that appear in both.
    """
    tools = score_matrix.columns.tolist()
    n = len(tools)
    if n < 2:
        return

    cutoff = max(1, int(np.ceil(len(score_matrix) * top_fraction)))

    # Get top hits per tool
    top_sets = {}
    for tool in tools:
        col = score_matrix[tool].dropna()
        top_ligands = col.nsmallest(cutoff).index.tolist()
        top_sets[tool] = set(top_ligands)

    # Compute pairwise Jaccard overlap
    pairs = []
    overlaps = []
    for i in range(n):
        for j in range(i + 1, n):
            s1 = top_sets[tools[i]]
            s2 = top_sets[tools[j]]
            union = len(s1 | s2)
            intersection = len(s1 & s2)
            jaccard = intersection / union if union > 0 else 0
            pairs.append(f"{tools[i]}\nvs\n{tools[j]}")
            overlaps.append(jaccard)

    fig, ax = plt.subplots(figsize=(max(6, len(pairs) * 1.5), 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(pairs)))
    ax.bar(range(len(pairs)), overlaps, color=colors, edgecolor="black")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=8)
    ax.set_ylabel(f"Jaccard Overlap (top {top_fraction:.0%})")
    ax.set_title(f"Top {top_fraction:.0%} Hit Overlap Between Tools")
    ax.set_ylim([0, 1])

    for i, v in enumerate(overlaps):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "top_hits_overlap.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved top hits overlap to %s", path)


def plot_rank_stability(
    consensus_df: pd.DataFrame,
    score_matrix: pd.DataFrame,
    output_dir: str,
    top_n: int = 50,
) -> None:
    """
    Parallel coordinates plot showing how the top N consensus hits
    rank across individual tools. Stable hits rank well in all tools.
    """
    tools = score_matrix.columns.tolist()
    rank_cols = [f"rank_{t}" for t in tools if f"rank_{t}" in consensus_df.columns]

    if not rank_cols:
        return

    top = consensus_df.head(top_n)
    n_ligands = len(score_matrix)

    fig, ax = plt.subplots(figsize=(max(6, len(rank_cols) * 2), 6))

    x = range(len(rank_cols))
    for _, row in top.iterrows():
        ranks = [row[rc] for rc in rank_cols]
        # Normalize to percentile (0% = best, 100% = worst)
        pcts = [100.0 * r / n_ligands if not np.isnan(r) else np.nan for r in ranks]
        ax.plot(x, pcts, alpha=0.3, linewidth=0.8, color="#2196F3")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("rank_", "") for c in rank_cols], fontsize=10)
    ax.set_ylabel("Percentile Rank (lower = better)")
    ax.set_title(f"Rank Stability: Top {top_n} Consensus Hits Across Tools")
    ax.invert_yaxis()
    ax.set_ylim([100, 0])
    ax.axhline(y=10, color="red", linestyle="--", alpha=0.5, label="Top 10%")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "rank_stability.png")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved rank stability plot to %s", path)
