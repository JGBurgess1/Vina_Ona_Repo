"""
Multi-tool validation docking for parameter optimization.

Bridges the consensus DockingBackend interface with the optimizer's
ValidationResult/EnrichmentMetrics system. Allows parameter optimization
for any docking tool, not just Vina.

Each (backend, parameter_set) combination is an independent unit of work
that can be distributed across MPI ranks.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .param_optimizer import OptimizationConfig, ParameterSet
from .roc_metrics import EnrichmentMetrics, compute_all_metrics
from .validation_docker import (
    ValidationResult,
    _to_serializable,
    _from_serializable,
    _sort_results,
    _chunk_round_robin,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolParamJob:
    """A single (backend_name, parameter_set) job to evaluate."""
    backend_name: str
    params: ParameterSet

    def to_dict(self) -> dict:
        return {
            "backend_name": self.backend_name,
            "params": self.params.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ToolParamJob":
        return cls(
            backend_name=d["backend_name"],
            params=ParameterSet(
                center=d["params"]["center"],
                box_size=d["params"]["box_size"],
                exhaustiveness=d["params"]["exhaustiveness"],
                label=d["params"]["label"],
            ),
        )


@dataclass
class ToolValidationResult:
    """Extends ValidationResult with the backend name."""
    backend_name: str
    result: ValidationResult


def _create_backend_for_params(
    backend_name: str,
    opt_config: OptimizationConfig,
    params: ParameterSet,
    mpi_rank: int = 0,
):
    """
    Create a DockingBackend instance configured for a specific parameter set.
    Imports backends lazily to avoid circular imports.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from consensus.backends.base import BackendConfig
    from consensus.backends.vina_backend import VinaBackend
    from consensus.backends.smina_backend import SminaBackend
    from consensus.backends.gnina_backend import GninaBackend
    from consensus.backends.rdock_backend import RDockBackend

    registry = {
        "vina": VinaBackend,
        "smina": SminaBackend,
        "gnina": GninaBackend,
        "rdock": RDockBackend,
    }

    cls = registry.get(backend_name)
    if cls is None:
        raise ValueError(f"Unknown backend: {backend_name}")

    # Per-rank, per-param output dir to avoid file collisions
    out_dir = os.path.join(
        opt_config.output_dir, "tool_optimization",
        backend_name, f"rank_{mpi_rank}",
    )

    config = BackendConfig(
        receptor_pdbqt=opt_config.receptor_pdbqt,
        center=params.center,
        box_size=params.box_size,
        exhaustiveness=params.exhaustiveness,
        n_poses=opt_config.n_poses,
        seed=opt_config.seed,
        scoring_function=opt_config.scoring_function,
        output_dir=out_dir,
        extra=getattr(opt_config, "backend_options", {}),
    )

    return cls(config)


def dock_validation_set_backend(
    opt_config: OptimizationConfig,
    backend_name: str,
    params: ParameterSet,
    active_paths: list,
    decoy_paths: list,
    mpi_rank: int = 0,
) -> ValidationResult:
    """
    Dock actives and decoys using any DockingBackend and compute metrics.

    This is the multi-tool equivalent of validation_docker.dock_validation_set().
    """
    t_start = time.time()

    backend = _create_backend_for_params(
        backend_name, opt_config, params, mpi_rank
    )

    if not backend.is_available():
        raise RuntimeError(f"Backend {backend_name} is not available")

    backend.prepare()

    # Dock actives
    logger.info(
        "  Rank %d [%s]: docking %d actives...",
        mpi_rank, backend_name, len(active_paths),
    )
    active_results = backend.dock_batch(active_paths)
    active_scores = []
    n_active_fail = 0
    for r in active_results:
        if r.success and r.score is not None:
            active_scores.append(r.score)
        else:
            n_active_fail += 1

    # Dock decoys
    logger.info(
        "  Rank %d [%s]: docking %d decoys...",
        mpi_rank, backend_name, len(decoy_paths),
    )
    decoy_results = backend.dock_batch(decoy_paths)
    decoy_scores = []
    n_decoy_fail = 0
    for r in decoy_results:
        if r.success and r.score is not None:
            decoy_scores.append(r.score)
        else:
            n_decoy_fail += 1

    if n_active_fail > 0 or n_decoy_fail > 0:
        logger.warning(
            "  Rank %d [%s]: failures: %d actives, %d decoys",
            mpi_rank, backend_name, n_active_fail, n_decoy_fail,
        )

    active_scores = np.array(active_scores, dtype=np.float64)
    decoy_scores = np.array(decoy_scores, dtype=np.float64)

    if len(active_scores) == 0 or len(decoy_scores) == 0:
        raise ValueError(
            f"[{backend_name}] No successful dockings: "
            f"{len(active_scores)} actives, {len(decoy_scores)} decoys"
        )

    all_scores = np.concatenate([active_scores, decoy_scores])
    all_labels = np.concatenate([
        np.ones(len(active_scores), dtype=int),
        np.zeros(len(decoy_scores), dtype=int),
    ])

    metrics = compute_all_metrics(all_labels, all_scores, store_curve=True)

    wall_time = time.time() - t_start
    logger.info(
        "  Rank %d [%s] %s: AUC=%.3f, BEDROC=%.3f, EF1%%=%.1f (%.1fs)",
        mpi_rank, backend_name, params.label,
        metrics.roc_auc, metrics.bedroc, metrics.ef_1pct, wall_time,
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


def _to_tool_serializable(backend_name: str, result: ValidationResult) -> dict:
    """Serialize a ToolValidationResult for MPI transfer."""
    d = _to_serializable(result)
    d["backend_name"] = backend_name
    return d


def _from_tool_serializable(d: dict) -> tuple:
    """Deserialize to (backend_name, ValidationResult)."""
    backend_name = d.pop("backend_name")
    result = _from_serializable(d)
    return backend_name, result


# ---------------------------------------------------------------------------
# Serial execution
# ---------------------------------------------------------------------------

def run_multitool_optimization(
    opt_config: OptimizationConfig,
    backend_names: list,
    param_sets: list,
    active_paths: list,
    decoy_paths: list,
) -> dict:
    """
    Evaluate all parameter sets for all backends serially.

    Returns:
        dict of backend_name -> list[ValidationResult] (sorted by metric)
    """
    results_by_tool = {name: [] for name in backend_names}

    total = len(backend_names) * len(param_sets)
    combo_i = 0

    for backend_name in backend_names:
        for params in param_sets:
            combo_i += 1
            logger.info(
                "Evaluating [%d/%d] %s: %s",
                combo_i, total, backend_name, params.label,
            )
            try:
                result = dock_validation_set_backend(
                    opt_config, backend_name, params,
                    active_paths, decoy_paths,
                )
                results_by_tool[backend_name].append(result)
            except Exception as e:
                logger.error("  [%s] Failed: %s", backend_name, e)
                continue

    # Sort each tool's results
    for name in backend_names:
        if results_by_tool[name]:
            results_by_tool[name] = _sort_results(
                results_by_tool[name], opt_config.metric
            )

    return results_by_tool


# ---------------------------------------------------------------------------
# MPI execution
# ---------------------------------------------------------------------------

def run_multitool_optimization_mpi(
    opt_config: OptimizationConfig,
    backend_names: list,
    param_sets: list,
    active_paths: list,
    decoy_paths: list,
    comm=None,
) -> Optional[dict]:
    """
    Evaluate all (backend, param_set) combinations in parallel across MPI ranks.

    Generates the Cartesian product of backends × param_sets, distributes
    these jobs across ranks, each rank evaluates its chunk, results are
    gathered and grouped by tool.

    Returns:
        dict of backend_name -> list[ValidationResult] on rank 0, None elsewhere.
    """
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Broadcast shared data ---
    if rank == 0:
        bcast_data = {
            "opt_config": opt_config,
            "active_paths": active_paths,
            "decoy_paths": decoy_paths,
            "backend_names": backend_names,
        }
    else:
        bcast_data = None

    bcast_data = comm.bcast(bcast_data, root=0)
    opt_config = bcast_data["opt_config"]
    active_paths = bcast_data["active_paths"]
    decoy_paths = bcast_data["decoy_paths"]
    backend_names = bcast_data["backend_names"]

    # --- Generate and scatter jobs ---
    if rank == 0:
        jobs = []
        for bname in backend_names:
            for params in param_sets:
                jobs.append(ToolParamJob(backend_name=bname, params=params))

        chunks = _chunk_round_robin(jobs, size)
        logger.info(
            "Distributing %d jobs (%d backends x %d param sets) across %d ranks",
            len(jobs), len(backend_names), len(param_sets), size,
        )
    else:
        chunks = None

    my_jobs = comm.scatter(chunks, root=0)
    logger.info("Rank %d: received %d jobs", rank, len(my_jobs))

    # --- Evaluate jobs ---
    my_results = []
    for i, job in enumerate(my_jobs):
        logger.info(
            "Rank %d: [%d/%d] %s — %s",
            rank, i + 1, len(my_jobs), job.backend_name, job.params.label,
        )
        try:
            result = dock_validation_set_backend(
                opt_config, job.backend_name, job.params,
                active_paths, decoy_paths, mpi_rank=rank,
            )
            my_results.append(_to_tool_serializable(job.backend_name, result))
        except Exception as e:
            logger.error(
                "Rank %d: [%s] %s failed: %s",
                rank, job.backend_name, job.params.label, e,
            )

    logger.info("Rank %d: completed %d/%d jobs", rank, len(my_results), len(my_jobs))

    # --- Gather ---
    all_serialized = comm.gather(my_results, root=0)

    if rank == 0:
        results_by_tool = {name: [] for name in backend_names}
        for rank_results in all_serialized:
            for d in rank_results:
                bname, result = _from_tool_serializable(d)
                results_by_tool[bname].append(result)

        # Sort each tool's results
        for name in backend_names:
            if results_by_tool[name]:
                results_by_tool[name] = _sort_results(
                    results_by_tool[name], opt_config.metric
                )
            n = len(results_by_tool[name])
            logger.info("Gathered %s: %d results", name, n)

        return results_by_tool

    return None
