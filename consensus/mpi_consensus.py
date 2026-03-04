"""
MPI orchestration for parallel consensus docking.

Distributes ligands across MPI ranks. Each rank runs ALL enabled backends
on its assigned ligand chunk. This ensures each ligand gets docked by
every tool, which is required for consensus scoring.

Alternative strategy (distributing backends across ranks) would require
all ligands to be available on all ranks and complicates result gathering.
The per-ligand distribution is simpler and scales better with large libraries.
"""

import logging
import time
from typing import Optional

from .backends.base import BackendConfig, BackendResult, DockingBackend

logger = logging.getLogger(__name__)


def _chunk_round_robin(items: list, n_chunks: int) -> list:
    """Split items into n_chunks lists using round-robin."""
    chunks = [[] for _ in range(n_chunks)]
    for i, item in enumerate(items):
        chunks[i % n_chunks].append(item)
    return chunks


def _serialize_results(results_by_tool: dict) -> dict:
    """Convert results to pickle-safe dicts for MPI transfer."""
    serialized = {}
    for tool_name, results in results_by_tool.items():
        serialized[tool_name] = []
        for r in results:
            serialized[tool_name].append({
                "ligand_path": r.ligand_path,
                "backend_name": r.backend_name,
                "success": r.success,
                "score": r.score,
                "extra_scores": r.extra_scores,
                "pose_file": r.pose_file,
                "error": r.error,
            })
    return serialized


def _deserialize_results(serialized: dict) -> dict:
    """Reconstruct BackendResult objects from serialized dicts."""
    results = {}
    for tool_name, result_dicts in serialized.items():
        results[tool_name] = []
        for d in result_dicts:
            results[tool_name].append(BackendResult(**d))
    return results


def run_consensus_serial(
    backends: list,
    ligand_paths: list,
) -> dict:
    """
    Run all backends on all ligands serially.
    Returns dict of tool_name -> list[BackendResult].
    """
    all_results = {}
    for backend in backends:
        logger.info("Running %s on %d ligands...", backend.name, len(ligand_paths))
        t0 = time.time()
        backend.prepare()
        results = backend.dock_batch(ligand_paths)
        elapsed = time.time() - t0

        n_success = sum(1 for r in results if r.success)
        logger.info(
            "%s complete: %d/%d success in %.1fs",
            backend.name, n_success, len(results), elapsed,
        )
        all_results[backend.name] = results

    return all_results


def run_consensus_mpi(
    backends: list,
    ligand_paths: list,
    comm=None,
) -> Optional[dict]:
    """
    Run consensus docking in parallel across MPI ranks.

    Each rank receives a chunk of ligands and runs ALL backends on that chunk.
    Results are gathered to rank 0.

    Args:
        backends: list of DockingBackend instances (same on all ranks)
        ligand_paths: full list of ligand paths (only needed on rank 0)
        comm: MPI communicator

    Returns:
        dict of tool_name -> list[BackendResult] on rank 0, None on workers.
    """
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Scatter ligand paths ---
    if rank == 0:
        chunks = _chunk_round_robin(ligand_paths, size)
        logger.info(
            "Distributing %d ligands across %d ranks for consensus docking "
            "(%d backends: %s)",
            len(ligand_paths), size, len(backends),
            ", ".join(b.name for b in backends),
        )
    else:
        chunks = None

    my_ligands = comm.scatter(chunks, root=0)
    logger.info("Rank %d: received %d ligands", rank, len(my_ligands))

    # --- Each rank runs all backends on its ligands ---
    my_results = {}
    for backend in backends:
        t0 = time.time()
        backend.prepare()
        results = backend.dock_batch(my_ligands)
        elapsed = time.time() - t0

        n_success = sum(1 for r in results if r.success)
        logger.info(
            "Rank %d: %s — %d/%d success in %.1fs",
            rank, backend.name, n_success, len(results), elapsed,
        )
        my_results[backend.name] = results

    # --- Serialize and gather ---
    my_serialized = _serialize_results(my_results)
    all_serialized = comm.gather(my_serialized, root=0)

    if rank == 0:
        # Merge results from all ranks
        merged = {b.name: [] for b in backends}
        for rank_results in all_serialized:
            deserialized = _deserialize_results(rank_results)
            for tool_name, results in deserialized.items():
                merged[tool_name].extend(results)

        total_ligands = len(ligand_paths)
        for tool_name, results in merged.items():
            n_success = sum(1 for r in results if r.success)
            logger.info(
                "Gathered %s: %d/%d success", tool_name, n_success, total_ligands
            )

        return merged

    return None
