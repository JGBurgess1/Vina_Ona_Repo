#!/usr/bin/env python
"""
Parallel molecular docking using AutoDock Vina and MPI.

Docks tens of thousands of ligands against a receptor in parallel
across an HPC cluster. Uses MPI scatter/gather to distribute ligand
batches evenly across all available cores.

Usage:
    mpiexec -n 600 python run_docking.py --config config.yaml --ligands /path/to/ligands/

    # With recursive ligand search:
    mpiexec -n 600 python run_docking.py --config config.yaml --ligands /path/to/ligands/ --recursive

    # Custom output:
    mpiexec -n 600 python run_docking.py --config config.yaml --ligands /path/to/ligands/ --output results.csv
"""

import argparse
import logging
import sys
import time

from mpi4py import MPI

from docking_engine import DockingConfig
from input_handler import discover_ligands, discover_ligands_recursive, load_config, validate_inputs
from mpi_orchestrator import MPIOrchestrator
from results_writer import print_summary, write_results_csv


def setup_logging(rank: int, verbose: bool = False) -> None:
    """Configure logging. Only rank 0 logs to stdout at INFO; workers log at WARNING."""
    level = logging.DEBUG if verbose else logging.INFO
    if rank != 0:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format=f"[Rank {rank:04d}] %(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel molecular docking with AutoDock Vina + MPI",
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--ligands", "-l",
        required=True,
        help="Directory containing ligand PDBQT files",
    )
    parser.add_argument(
        "--output", "-o",
        default="output/results.csv",
        help="Output CSV file path (default: output/results.csv)",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search for ligands recursively in subdirectories",
    )
    parser.add_argument(
        "--pattern", "-p",
        default="*.pdbqt",
        help="Glob pattern for ligand files (default: *.pdbqt)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging on all ranks",
    )
    return parser.parse_args()


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args()
    setup_logging(rank, args.verbose)
    logger = logging.getLogger(__name__)

    # Only rank 0 handles input discovery and validation
    config = None
    ligand_paths = None

    if rank == 0:
        logger.info("MPI docking: %d ranks available", size)
        t_start = time.time()

        try:
            config = load_config(args.config)
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            comm.Abort(1)
            return 1

        try:
            if args.recursive:
                ligand_paths = discover_ligands_recursive(args.ligands, args.pattern)
            else:
                ligand_paths = discover_ligands(args.ligands, args.pattern)
        except FileNotFoundError as e:
            logger.error("Ligand discovery failed: %s", e)
            comm.Abort(1)
            return 1

        try:
            validate_inputs(config, ligand_paths)
        except FileNotFoundError as e:
            logger.error("Input validation failed: %s", e)
            comm.Abort(1)
            return 1

        logger.info(
            "Found %d ligands, distributing across %d ranks (~%d per rank)",
            len(ligand_paths),
            size,
            len(ligand_paths) // size,
        )

    # Broadcast config to all ranks
    config = comm.bcast(config, root=0)

    # Run the parallel docking
    orchestrator = MPIOrchestrator(config, ligand_paths)
    all_results = orchestrator.run()

    # Rank 0 writes output
    if rank == 0 and all_results is not None:
        write_results_csv(all_results, args.output)
        print_summary(all_results)

        t_end = time.time()
        logger.info("Total wall time: %.1fs", t_end - t_start)

    return 0


if __name__ == "__main__":
    sys.exit(main())
