"""
Validation docking runner for parameter optimization.

Docks known actives and property-matched decoys (e.g., LUDe decoys)
against a receptor using a given parameter set, then computes
enrichment metrics to evaluate discrimination quality.

Supports both serial and MPI-parallel execution:
  - Serial: run_optimization() — evaluates parameter sets one at a time
  - MPI:    run_optimization_mpi() — distributes parameter sets across ranks

Integrates directly with DockingEngine and DockingConfig from the
Vina MPI pipeline.
"""

import glob
import logging
import os
import time
from dataclasses import dataclass, asdict
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


def _to_serializable(result: ValidationResult) -> dict:
    """Convert a ValidationResult to a pickle-safe dict for MPI transfer."""
    return {
        "params": result.params.to_dict(),
        "metrics": {
            "roc_auc": result.metrics.roc_auc,
            "log_auc": result.metrics.log_auc,
            "bedroc": result.metrics.bedroc,
            "ef_1pct": result.metrics.ef_1pct,
            "ef_5pct": result.metrics.ef_5pct,
            "ef_10pct": result.metrics.ef_10pct,
            "n_actives": result.metrics.n_actives,
            "n_decoys": result.metrics.n_decoys,
            "fpr": result.metrics.fpr.tolist() if result.metrics.fpr is not None else None,
            "tpr": result.metrics.tpr.tolist() if result.metrics.tpr is not None else None,
        },
        "active_scores": result.active_scores.tolist(),
        "decoy_scores": result.decoy_scores.tolist(),
        "all_scores": result.all_scores.tolist(),
        "all_labels": result.all_labels.tolist(),
        "n_active_failures": result.n_active_failures,
        "n_decoy_failures": result.n_decoy_failures,
        "wall_time": result.wall_time,
    }


def _from_serializable(d: dict) -> ValidationResult:
    """Reconstruct a ValidationResult from a serialized dict."""
    params = ParameterSet(
        center=d["params"]["center"],
        box_size=d["params"]["box_size"],
        exhaustiveness=d["params"]["exhaustiveness"],
        label=d["params"]["label"],
    )
    m = d["metrics"]
    metrics = EnrichmentMetrics(
        roc_auc=m["roc_auc"],
        log_auc=m["log_auc"],
        bedroc=m["bedroc"],
        ef_1pct=m["ef_1pct"],
        ef_5pct=m["ef_5pct"],
        ef_10pct=m["ef_10pct"],
        n_actives=m["n_actives"],
        n_decoys=m["n_decoys"],
        fpr=np.array(m["fpr"]) if m["fpr"] is not None else None,
        tpr=np.array(m["tpr"]) if m["tpr"] is not None else None,
    )
    return ValidationResult(
        params=params,
        metrics=metrics,
        active_scores=np.array(d["active_scores"]),
        decoy_scores=np.array(d["decoy_scores"]),
        all_scores=np.array(d["all_scores"]),
        all_labels=np.array(d["all_labels"]),
        n_active_failures=d["n_active_failures"],
        n_decoy_failures=d["n_decoy_failures"],
        wall_time=d["wall_time"],
    )


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
    mpi_rank: int = 0,
) -> ValidationResult:
    """
    Dock actives and decoys with a specific parameter set and compute metrics.

    Creates a DockingConfig from the optimization config + parameter set,
    runs docking via DockingEngine, then evaluates enrichment.
    """
    t_start = time.time()

    DockingConfig, DockingEngine = _import_docking_engine()

    # Per-rank maps directory to avoid file collisions in parallel execution
    maps_dir = os.path.join(opt_config.output_dir, "maps", f"rank_{mpi_rank}")

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
        maps_dir=maps_dir,
    )

    # Initialize engine and prepare maps
    engine = DockingEngine(dock_config, rank=mpi_rank)
    map_prefix = engine.prepare_maps()
    engine.initialize(map_prefix)

    # Dock actives
    logger.info("  Rank %d: docking %d actives...", mpi_rank, len(active_paths))
    active_results = engine.dock_batch(active_paths)
    active_scores = []
    n_active_fail = 0
    for r in active_results:
        if r.success and r.best_energy is not None:
            active_scores.append(r.best_energy)
        else:
            n_active_fail += 1

    # Dock decoys
    logger.info("  Rank %d: docking %d decoys...", mpi_rank, len(decoy_paths))
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
            "  Rank %d: docking failures: %d actives, %d decoys",
            mpi_rank, n_active_fail, n_decoy_fail,
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
        "  Rank %d: %s — AUC=%.3f, BEDROC=%.3f, EF1%%=%.1f (%.1fs)",
        mpi_rank, params.label, metrics.roc_auc, metrics.bedroc,
        metrics.ef_1pct, wall_time,
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


def _sort_results(results: list, metric_key: str) -> list:
    """Sort results by target metric, descending (higher is better)."""
    results.sort(
        key=lambda r: getattr(r.metrics, metric_key, r.metrics.roc_auc),
        reverse=True,
    )
    return results


def _chunk_round_robin(items: list, n_chunks: int) -> list:
    """Split items into n_chunks lists using round-robin assignment."""
    chunks = [[] for _ in range(n_chunks)]
    for i, item in enumerate(items):
        chunks[i % n_chunks].append(item)
    return chunks


def run_optimization(
    opt_config: OptimizationConfig,
    param_sets: list,
    active_paths: list,
    decoy_paths: list,
) -> list:
    """
    Evaluate all parameter sets serially and return sorted results.

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

    return _sort_results(results, opt_config.metric)


def run_optimization_mpi(
    opt_config: OptimizationConfig,
    param_sets: list,
    active_paths: list,
    decoy_paths: list,
    comm=None,
) -> Optional[list]:
    """
    Evaluate parameter sets in parallel across MPI ranks.

    Distributes parameter sets evenly via scatter, each rank evaluates
    its chunk independently, then results are gathered back to rank 0.

    Args:
        opt_config: optimization configuration
        param_sets: list of ParameterSet to evaluate (only needed on rank 0)
        active_paths: paths to active ligand PDBQT files
        decoy_paths: paths to decoy ligand PDBQT files
        comm: MPI communicator (if None, imports MPI.COMM_WORLD)

    Returns:
        Sorted list of ValidationResult on rank 0, None on other ranks.
    """
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Broadcast shared data from rank 0 ---
    if rank == 0:
        bcast_data = {
            "opt_config": opt_config,
            "active_paths": active_paths,
            "decoy_paths": decoy_paths,
        }
    else:
        bcast_data = None

    bcast_data = comm.bcast(bcast_data, root=0)
    opt_config = bcast_data["opt_config"]
    active_paths = bcast_data["active_paths"]
    decoy_paths = bcast_data["decoy_paths"]

    # --- Scatter parameter sets ---
    if rank == 0:
        chunks = _chunk_round_robin(param_sets, size)
        logger.info(
            "Distributing %d parameter sets across %d ranks (min=%d, max=%d per rank)",
            len(param_sets), size,
            min(len(c) for c in chunks),
            max(len(c) for c in chunks),
        )
    else:
        chunks = None

    my_param_sets = comm.scatter(chunks, root=0)

    logger.info(
        "Rank %d: received %d parameter sets to evaluate", rank, len(my_param_sets)
    )

    # --- Each rank evaluates its parameter sets ---
    my_results = []
    for i, params in enumerate(my_param_sets):
        logger.info(
            "Rank %d: evaluating %d/%d — %s",
            rank, i + 1, len(my_param_sets), params.label,
        )
        try:
            result = dock_validation_set(
                opt_config, params, active_paths, decoy_paths, mpi_rank=rank,
            )
            my_results.append(result)
        except Exception as e:
            logger.error("Rank %d: failed %s — %s", rank, params.label, e)
            continue

    n_success = len(my_results)
    n_fail = len(my_param_sets) - n_success
    logger.info(
        "Rank %d: finished — %d success, %d failed", rank, n_success, n_fail
    )

    # --- Serialize for MPI transfer ---
    my_serialized = [_to_serializable(r) for r in my_results]

    # --- Gather all results to rank 0 ---
    all_serialized_lists = comm.gather(my_serialized, root=0)

    if rank == 0:
        all_results = []
        for result_list in all_serialized_lists:
            for d in result_list:
                all_results.append(_from_serializable(d))

        total_success = len(all_results)
        total_attempted = len(param_sets)
        logger.info(
            "Gathered results: %d/%d parameter sets succeeded",
            total_success, total_attempted,
        )

        if not all_results:
            raise RuntimeError("All parameter sets failed across all ranks")

        return _sort_results(all_results, opt_config.metric)

    return None
