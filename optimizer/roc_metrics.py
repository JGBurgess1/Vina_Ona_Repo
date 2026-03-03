"""
ROC and enrichment metrics for evaluating docking discrimination
between known actives and property-matched decoys.

Metrics:
  - ROC AUC: area under the receiver operating characteristic curve
  - LogAUC: semi-log AUC emphasizing early enrichment (log10 x-axis)
  - BEDROC: Boltzmann-enhanced discrimination of ROC (alpha=20 by default)
  - Enrichment Factor at X%: ratio of actives found in top X% vs random
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentMetrics:
    """Full set of enrichment metrics for one parameter configuration."""
    roc_auc: float
    log_auc: float
    bedroc: float
    ef_1pct: float
    ef_5pct: float
    ef_10pct: float
    n_actives: int
    n_decoys: int
    # ROC curve data for plotting
    fpr: Optional[np.ndarray] = None
    tpr: Optional[np.ndarray] = None


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> tuple:
    """
    Compute ROC AUC. Docking scores are negated because lower (more negative)
    scores indicate better binding, but sklearn expects higher = more positive.

    Args:
        labels: binary array (1=active, 0=decoy)
        scores: docking scores (kcal/mol, more negative = better)

    Returns:
        (auc, fpr, tpr, thresholds)
    """
    # Negate scores: better binders (more negative) become higher values
    neg_scores = -scores
    fpr, tpr, thresholds = roc_curve(labels, neg_scores)
    auc = roc_auc_score(labels, neg_scores)
    return auc, fpr, tpr, thresholds


def compute_log_auc(fpr: np.ndarray, tpr: np.ndarray, min_fpr: float = 0.001) -> float:
    """
    Semi-log AUC: integrates the ROC curve on a log10 x-axis.
    Emphasizes early enrichment (low false positive rates).

    The x-axis is log10(FPR) from min_fpr to 1.0.
    Normalized so random performance = 0.145 (for min_fpr=0.001).
    """
    # Filter to FPR >= min_fpr
    mask = fpr >= min_fpr
    if mask.sum() < 2:
        return 0.0

    fpr_filtered = fpr[mask]
    tpr_filtered = tpr[mask]

    log_fpr = np.log10(fpr_filtered)
    log_auc = np.trapz(tpr_filtered, log_fpr)

    # Normalize by the total log range
    log_range = np.log10(1.0) - np.log10(min_fpr)
    return log_auc / log_range


def compute_bedroc(labels: np.ndarray, scores: np.ndarray, alpha: float = 20.0) -> float:
    """
    Boltzmann-Enhanced Discrimination of ROC (BEDROC).

    Weights early enrichment exponentially. alpha controls the weighting:
    higher alpha = more emphasis on top-ranked compounds.
    alpha=20 corresponds to ~80% of the score coming from the top 8% of the list.

    Reference: Truchon & Bayly, J. Chem. Inf. Model. 2007, 47, 488-508.
    """
    n = len(labels)
    n_actives = int(labels.sum())

    if n_actives == 0 or n_actives == n:
        return 0.0

    # Sort by score (ascending = best binders first, most negative first)
    order = np.argsort(scores)
    sorted_labels = labels[order]

    # Ranks of actives (1-indexed, normalized to [0, 1])
    active_ranks = np.where(sorted_labels == 1)[0]
    ri = (active_ranks + 1) / n  # normalized ranks

    # BEDROC formula
    s = np.sum(np.exp(-alpha * ri))
    ra = n_actives / n

    # Random and max BEDROC
    random_sum = (ra * (1 - np.exp(-alpha))) / (np.exp(alpha / n) - 1)
    factor = (ra * np.sinh(alpha / 2)) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ra))
    max_sum = (1 - np.exp(-alpha * ra)) / (1 - np.exp(-alpha / n))

    if max_sum - random_sum == 0:
        return 0.0

    bedroc = (s - random_sum) / (max_sum - random_sum)
    return float(np.clip(bedroc, 0.0, 1.0))


def compute_enrichment_factor(
    labels: np.ndarray,
    scores: np.ndarray,
    fraction: float,
) -> float:
    """
    Enrichment factor at a given fraction of the ranked list.

    EF = (actives in top X%) / (expected actives in top X% by random).
    """
    n = len(labels)
    n_actives = int(labels.sum())

    if n_actives == 0 or n == 0:
        return 0.0

    # Sort by score ascending (most negative = best binder first)
    order = np.argsort(scores)
    sorted_labels = labels[order]

    cutoff = max(1, int(np.ceil(n * fraction)))
    actives_in_top = sorted_labels[:cutoff].sum()

    expected = n_actives * fraction
    if expected == 0:
        return 0.0

    return float(actives_in_top / expected)


def compute_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    store_curve: bool = True,
) -> EnrichmentMetrics:
    """
    Compute the full suite of enrichment metrics.

    Args:
        labels: binary array (1=active, 0=decoy)
        scores: docking scores (kcal/mol)
        store_curve: if True, store FPR/TPR arrays for plotting

    Returns:
        EnrichmentMetrics dataclass
    """
    n_actives = int(labels.sum())
    n_decoys = len(labels) - n_actives

    auc, fpr, tpr, _ = compute_roc_auc(labels, scores)
    log_auc = compute_log_auc(fpr, tpr)
    bedroc = compute_bedroc(labels, scores, alpha=20.0)
    ef_1 = compute_enrichment_factor(labels, scores, 0.01)
    ef_5 = compute_enrichment_factor(labels, scores, 0.05)
    ef_10 = compute_enrichment_factor(labels, scores, 0.10)

    logger.info(
        "Metrics: AUC=%.3f, LogAUC=%.3f, BEDROC=%.3f, EF1%%=%.1f, EF5%%=%.1f, EF10%%=%.1f "
        "(actives=%d, decoys=%d)",
        auc, log_auc, bedroc, ef_1, ef_5, ef_10, n_actives, n_decoys,
    )

    return EnrichmentMetrics(
        roc_auc=auc,
        log_auc=log_auc,
        bedroc=bedroc,
        ef_1pct=ef_1,
        ef_5pct=ef_5,
        ef_10pct=ef_10,
        n_actives=n_actives,
        n_decoys=n_decoys,
        fpr=fpr if store_curve else None,
        tpr=tpr if store_curve else None,
    )
