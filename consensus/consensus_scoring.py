"""
Consensus scoring methods for combining results from multiple docking tools.

Implements five consensus strategies:
  1. Average Rank (RbR)     — average fractional rank across tools
  2. Z-Score Normalization  — normalize scores to Z-scores, then average
  3. Exponential Consensus Ranking (ECR) — exponentially weighted rank fusion
  4. Best-of-N              — take the best normalized score from any tool
  5. Majority Voting        — count how many tools rank a ligand in the top X%

All methods handle missing data (ligand failed in one tool) gracefully.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Consensus scores for a single ligand across all methods."""
    ligand: str
    avg_rank: float
    z_score_avg: float
    ecr_score: float
    best_of_n: float
    vote_count: int
    per_tool_scores: dict    # tool_name -> raw score
    per_tool_ranks: dict     # tool_name -> fractional rank
    n_tools_succeeded: int


def build_score_matrix(
    all_results: dict,
    ligand_names: list,
) -> pd.DataFrame:
    """
    Build a DataFrame of raw scores: rows=ligands, columns=tools.
    Missing values (failed dockings) are NaN.

    Args:
        all_results: dict of tool_name -> list[BackendResult]
        ligand_names: ordered list of ligand identifiers

    Returns:
        DataFrame with shape (n_ligands, n_tools)
    """
    data = {}
    for tool_name, results in all_results.items():
        score_map = {}
        for r in results:
            key = _ligand_key(r.ligand_path)
            if r.success and r.score is not None:
                score_map[key] = r.score
        data[tool_name] = [score_map.get(name, np.nan) for name in ligand_names]

    df = pd.DataFrame(data, index=ligand_names)
    logger.info(
        "Score matrix: %d ligands x %d tools, %.1f%% missing",
        len(df), len(df.columns),
        100.0 * df.isna().sum().sum() / df.size,
    )
    return df


def _ligand_key(path: str) -> str:
    """Extract a consistent ligand identifier from a file path."""
    import os
    name = os.path.basename(path)
    # Strip common extensions
    for ext in [".pdbqt", ".sdf", ".sd", ".mol2", ".pdb"]:
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name


def compute_fractional_ranks(scores: pd.Series) -> pd.Series:
    """
    Compute fractional ranks from scores. Lower score = better = rank 1.
    NaN values get rank NaN. Ties get average rank.
    """
    return scores.rank(method="average", ascending=True, na_option="keep")


def consensus_average_rank(score_matrix: pd.DataFrame) -> pd.Series:
    """
    Average Rank (Rank-by-Rank, RbR).

    For each tool, rank all ligands by score (lower = better = rank 1).
    The consensus score is the average rank across tools.
    Lower average rank = better consensus hit.
    """
    rank_matrix = score_matrix.apply(compute_fractional_ranks, axis=0)
    avg_ranks = rank_matrix.mean(axis=1, skipna=True)
    return avg_ranks


def consensus_zscore(score_matrix: pd.DataFrame) -> pd.Series:
    """
    Z-Score Normalization.

    Normalize each tool's scores to Z-scores (mean=0, std=1), then average.
    This accounts for different score scales across tools.
    More negative Z-score = better consensus hit.
    """
    z_matrix = score_matrix.apply(
        lambda col: (col - col.mean()) / col.std() if col.std() > 0 else col * 0,
        axis=0,
    )
    z_avg = z_matrix.mean(axis=1, skipna=True)
    return z_avg


def consensus_ecr(score_matrix: pd.DataFrame, sigma: float = 0.05) -> pd.Series:
    """
    Exponential Consensus Ranking (ECR).

    Converts ranks to exponential scores: exp(-rank / (sigma * N)),
    then averages across tools. Emphasizes top-ranked compounds.

    sigma controls the steepness: smaller sigma = more emphasis on top ranks.
    Default sigma=0.05 means compounds outside the top 5% contribute negligibly.

    Reference: Palacio-Rodriguez et al., J. Chem. Inf. Model. 2019.
    """
    n = len(score_matrix)
    rank_matrix = score_matrix.apply(compute_fractional_ranks, axis=0)

    ecr_matrix = rank_matrix.apply(
        lambda col: np.exp(-col / (sigma * n)),
        axis=0,
    )
    ecr_avg = ecr_matrix.mean(axis=1, skipna=True)
    return ecr_avg


def consensus_best_of_n(score_matrix: pd.DataFrame) -> pd.Series:
    """
    Best-of-N.

    For each ligand, take the best (most negative) Z-normalized score
    from any tool. Identifies compounds that score well in at least one program.
    """
    z_matrix = score_matrix.apply(
        lambda col: (col - col.mean()) / col.std() if col.std() > 0 else col * 0,
        axis=0,
    )
    best = z_matrix.min(axis=1, skipna=True)
    return best


def consensus_majority_vote(
    score_matrix: pd.DataFrame,
    top_fraction: float = 0.10,
) -> pd.Series:
    """
    Majority Voting.

    For each tool, identify the top X% of ligands. The consensus score
    is the number of tools that rank a ligand in their top X%.
    Higher count = more tools agree it's a hit.
    """
    n = len(score_matrix)
    cutoff = max(1, int(np.ceil(n * top_fraction)))

    vote_matrix = pd.DataFrame(index=score_matrix.index)
    for tool in score_matrix.columns:
        col = score_matrix[tool].dropna()
        if len(col) == 0:
            vote_matrix[tool] = 0
            continue
        threshold = col.nsmallest(cutoff).iloc[-1]
        vote_matrix[tool] = (score_matrix[tool] <= threshold).astype(int)

    votes = vote_matrix.sum(axis=1)
    return votes


def compute_all_consensus(
    all_results: dict,
    ligand_names: list,
    vote_fraction: float = 0.10,
    ecr_sigma: float = 0.05,
) -> tuple:
    """
    Compute all consensus methods and return structured results.

    Args:
        all_results: dict of tool_name -> list[BackendResult]
        ligand_names: ordered list of ligand identifiers
        vote_fraction: top fraction for majority voting (default: 10%)
        ecr_sigma: sigma for ECR (default: 0.05)

    Returns:
        (consensus_df, score_matrix)
        consensus_df: DataFrame with all consensus scores per ligand
        score_matrix: raw score matrix
    """
    score_matrix = build_score_matrix(all_results, ligand_names)

    avg_rank = consensus_average_rank(score_matrix)
    z_avg = consensus_zscore(score_matrix)
    ecr = consensus_ecr(score_matrix, sigma=ecr_sigma)
    best_n = consensus_best_of_n(score_matrix)
    votes = consensus_majority_vote(score_matrix, top_fraction=vote_fraction)

    # Per-tool ranks
    rank_matrix = score_matrix.apply(compute_fractional_ranks, axis=0)

    # Build consensus DataFrame
    consensus_df = pd.DataFrame({
        "ligand": ligand_names,
        "avg_rank": avg_rank.values,
        "z_score_avg": z_avg.values,
        "ecr_score": ecr.values,
        "best_of_n": best_n.values,
        "vote_count": votes.values.astype(int),
        "n_tools_succeeded": score_matrix.notna().sum(axis=1).values.astype(int),
    })

    # Add per-tool raw scores and ranks
    for tool in score_matrix.columns:
        consensus_df[f"score_{tool}"] = score_matrix[tool].values
        consensus_df[f"rank_{tool}"] = rank_matrix[tool].values

    # Sort by average rank (primary consensus method)
    consensus_df = consensus_df.sort_values("avg_rank").reset_index(drop=True)
    consensus_df.insert(0, "consensus_rank", range(1, len(consensus_df) + 1))

    logger.info(
        "Consensus computed: %d ligands, %d tools, %d methods",
        len(consensus_df), len(score_matrix.columns), 5,
    )

    return consensus_df, score_matrix
