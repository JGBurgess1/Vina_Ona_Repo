"""
Validation docking runner for parameter optimization.

Docks known actives and property-matched decoys (e.g., LUDe decoys)
against a receptor using a given parameter set, then computes
enrichment metrics to evaluate discrimination quality.

Integrates directly with DockingEngine and DockingConfig from the
Vina MPI pipeline.
"""

import glob
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .param_optimizer import OptimizationConfig, ParameterSet
from .roc_metrics import EnrichmentMetrics, compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single parameter set evaluation."""
    params: ParameterSet
    metrics: EnrichmentMetrics
    active_scores: np.ndarray
    decoy_scores: np.ndarray
    all_scores: np.ndarray
    all_labels: np.ndarray
    n_active_failures: int
    n_decoy_failures: int
    wall_time: float


def discover_ligands(directory: str, pattern: str = "*.pdbqt") -> list:
    """Find all ligand files in a directory."""
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {directory}"
        )
    return paths


def _import_docking_engine():
    """Lazy import to avoid loading the vina C++ bindings at module import time."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from docking_engine import DockingConfig, DockingEngine
    return DockingConfig, DockingEngine


def dock_validation_set(
    opt_config: OptimizationConfig,
    params: ParameterSet,
    active_paths: list,
    decoy_paths: list,
) -> ValidationResult:
    """
    Dock actives and decoys with a specific parameter set and compute metrics.

    Creates a DockingConfig from the optimization config + parameter set,
    runs docking via DockingEngine, then evaluates enrichment.
    """
    t_start = time.time()

    DockingConfig, DockingEngine = _import_docking_engine()

    # Build DockingConfig for this parameter set
    dock_config = DockingConfig(
        receptor_pdbqt=opt_config.receptor_pdbqt,
        center=params.center,
        box_size=params.box_size,
        spacing=opt_config.spacing,
        scoring_function=opt_config.scoring_function,
        exhaustiveness=params.exhaustiveness,
        n_poses=opt_config.n_poses,
        min_rmsd=1.0,
        max_evals=0,
        energy_range=3.0,
        seed=opt_config.seed,
        write_poses=opt_config.write_poses,
        output_dir=os.path.join(opt_config.output_dir, "poses"),
        maps_dir=os.path.join(opt_config.output_dir, "maps"),
    )

    # Initialize engine and prepare maps
    engine = DockingEngine(dock_config, rank=0)
    map_prefix = engine.prepare_maps()
    engine.initialize(map_prefix)

    # Dock actives
    logger.info("  Docking %d actives...", len(active_paths))
    active_results = engine.dock_batch(active_paths)
    active_scores = []
    n_active_fail = 0
    for r in active_results:
        if r.success and r.best_energy is not None:
            active_scores.append(r.best_energy)
        else:
            n_active_fail += 1

    # Dock decoys
    logger.info("  Docking %d decoys...", len(decoy_paths))
    decoy_results = engine.dock_batch(decoy_paths)
    decoy_scores = []
    n_decoy_fail = 0
    for r in decoy_results:
        if r.success and r.best_energy is not None:
            decoy_scores.append(r.best_energy)
        else:
            n_decoy_fail += 1

    if n_active_fail > 0 or n_decoy_fail > 0:
        logger.warning(
            "  Docking failures: %d actives, %d decoys", n_active_fail, n_decoy_fail
        )

    active_scores = np.array(active_scores, dtype=np.float64)
    decoy_scores = np.array(decoy_scores, dtype=np.float64)

    if len(active_scores) == 0 or len(decoy_scores) == 0:
        raise ValueError(
            f"No successful dockings: {len(active_scores)} actives, {len(decoy_scores)} decoys"
        )

    # Build labels and combined scores
    all_scores = np.concatenate([active_scores, decoy_scores])
    all_labels = np.concatenate([
        np.ones(len(active_scores), dtype=int),
        np.zeros(len(decoy_scores), dtype=int),
    ])

    # Compute enrichment metrics
    metrics = compute_all_metrics(all_labels, all_scores, store_curve=True)

    wall_time = time.time() - t_start
    logger.info(
        "  %s: AUC=%.3f, BEDROC=%.3f, EF1%%=%.1f (%.1fs)",
        params.label, metrics.roc_auc, metrics.bedroc, metrics.ef_1pct, wall_time,
    )

    return ValidationResult(
        params=params,
        metrics=metrics,
        active_scores=active_scores,
        decoy_scores=decoy_scores,
        all_scores=all_scores,
        all_labels=all_labels,
        n_active_failures=n_active_fail,
        n_decoy_failures=n_decoy_fail,
        wall_time=wall_time,
    )


def run_optimization(
    opt_config: OptimizationConfig,
    param_sets: list,
    active_paths: list,
    decoy_paths: list,
) -> list:
    """
    Evaluate all parameter sets and return sorted results.

    Args:
        opt_config: optimization configuration
        param_sets: list of ParameterSet to evaluate
        active_paths: paths to active ligand PDBQT files
        decoy_paths: paths to decoy ligand PDBQT files

    Returns:
        List of ValidationResult sorted by the target metric (best first).
    """
    results = []
    total = len(param_sets)

    for i, params in enumerate(param_sets):
        logger.info(
            "Evaluating parameter set %d/%d: %s", i + 1, total, params.label
        )
        try:
            result = dock_validation_set(opt_config, params, active_paths, decoy_paths)
            results.append(result)
        except Exception as e:
            logger.error("  Failed: %s", e)
            continue

    if not results:
        raise RuntimeError("All parameter sets failed")

    # Sort by target metric (descending — higher is better for all metrics)
    metric_key = opt_config.metric
    results.sort(
        key=lambda r: getattr(r.metrics, metric_key, r.metrics.roc_auc),
        reverse=True,
    )

    return results
