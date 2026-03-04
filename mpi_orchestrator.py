"""
MPI orchestration for parallel docking across hundreds of cores.

Uses a scatter/gather pattern:
  - Rank 0 discovers ligands, computes affinity maps, distributes work
  - All ranks (including 0) dock their assigned chunk
  - Rank 0 gathers all results and writes output

Run with:
    mpiexec -n 600 python run_docking.py --config config.yaml
"""

import logging
import time
from dataclasses import asdict
from typing import Optional

from mpi4py import MPI

from docking_engine import DockingConfig, DockingEngine, DockingResult
from logging_config import ProgressTracker

logger = logging.getLogger(__name__)


def chunk_list(lst: list, n_chunks: int) -> list:
    """
    Split a list into n_chunks roughly equal parts.
    Returns a list of n_chunks lists. Some may be empty if
    len(lst) < n_chunks.
    """
    chunks = [[] for _ in range(n_chunks)]
    for i, item in enumerate(lst):
        chunks[i % n_chunks].append(item)
    return chunks


class MPIOrchestrator:
    """
    Coordinates parallel docking across MPI ranks.
    """

    def __init__(self, config: DockingConfig, ligand_paths: Optional[list] = None):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.config = config
        self._ligand_paths = ligand_paths  # only needed on rank 0

    def run(self) -> Optional[list]:
        """
        Execute the full parallel docking pipeline.
        Returns the aggregated list of DockingResult dicts on rank 0, None elsewhere.
        """
        t_start = time.time()

        # --- Phase 1: Map preparation (rank 0 only) ---
        map_prefix = None
        if self.rank == 0:
            logger.info(
                "Starting docking campaign: %d ligands across %d MPI ranks",
                len(self._ligand_paths) if self._ligand_paths else 0,
                self.size,
            )
            engine = DockingEngine(self.config, rank=self.rank)
            map_prefix = engine.prepare_maps()
            logger.info("Map preparation complete")

        # Broadcast the map prefix path to all ranks
        map_prefix = self.comm.bcast(map_prefix, root=0)

        # --- Phase 2: Distribute ligand paths ---
        if self.rank == 0:
            chunks = chunk_list(self._ligand_paths, self.size)
            logger.info(
                "Distributing ligands: min=%d, max=%d per rank",
                min(len(c) for c in chunks),
                max(len(c) for c in chunks),
            )
        else:
            chunks = None

        my_ligands = self.comm.scatter(chunks, root=0)
        logger.info(
            "Rank %d: received %d ligands to dock", self.rank, len(my_ligands)
        )

        # --- Phase 3: Initialize engine and dock ---
        engine = DockingEngine(self.config, rank=self.rank)
        engine.initialize(map_prefix)

        tracker = ProgressTracker(
            total=len(my_ligands),
            label="Docking",
            rank=self.rank,
        )

        t_dock_start = time.time()
        my_results = []
        for i, ligand_path in enumerate(my_ligands):
            result = engine.dock_ligand(ligand_path)
            my_results.append(result)
            tracker.update(i + 1)
        t_dock_end = time.time()

        n_success = sum(1 for r in my_results if r.success)
        n_fail = len(my_results) - n_success
        tracker.finish(n_success=n_success, n_failed=n_fail)

        # Convert to serializable dicts for MPI gather
        my_result_dicts = [asdict(r) for r in my_results]

        # --- Phase 4: Gather results ---
        all_result_lists = self.comm.gather(my_result_dicts, root=0)

        if self.rank == 0:
            # Flatten the list of lists
            all_results = []
            for result_list in all_result_lists:
                all_results.extend(result_list)

            t_end = time.time()
            total_success = sum(1 for r in all_results if r["success"])
            total_fail = len(all_results) - total_success

            logger.info(
                "Docking campaign complete: %d total, %d success, %d failed, %.1fs elapsed",
                len(all_results),
                total_success,
                total_fail,
                t_end - t_start,
            )

            return all_results

        return None
