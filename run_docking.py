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
from logging_config import configure_logging, log_config_summary, log_final_summary, log_phase
from mpi_orchestrator import MPIOrchestrator
from results_writer import print_summary, write_results_csv


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
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for log files (default: logs)",
    )
    return parser.parse_args()


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args()
    configure_logging(
        log_dir=args.log_dir,
        log_name="docking_campaign",
        rank=rank,
        mpi_size=size,
        verbose=args.verbose,
    )
    logger = logging.getLogger(__name__)

    # Only rank 0 handles input discovery and validation
    config = None
    ligand_paths = None

    if rank == 0:
        t_start = time.time()
        log_phase(logger, 1, "Input discovery and validation")

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

        log_config_summary(
            logger,
            mpi_ranks=size,
            ligands=len(ligand_paths),
            ligands_per_rank=f"~{len(ligand_paths) // size}",
            receptor=config.receptor_pdbqt,
            exhaustiveness=config.exhaustiveness,
            scoring=config.scoring_function,
            output=args.output,
        )

    # Broadcast config to all ranks
    config = comm.bcast(config, root=0)

    # Run the parallel docking
    if rank == 0:
        log_phase(logger, 2, "Parallel docking")

    orchestrator = MPIOrchestrator(config, ligand_paths)
    all_results = orchestrator.run()

    # Rank 0 writes output
    if rank == 0 and all_results is not None:
        log_phase(logger, 3, "Writing results")
        write_results_csv(all_results, args.output)
        print_summary(all_results)

        t_end = time.time()
        total_success = sum(1 for r in all_results if r["success"])
        total_fail = len(all_results) - total_success
        log_final_summary(
            logger,
            program="Docking Campaign",
            wall_time=t_end - t_start,
            total_ligands=len(all_results),
            successful=total_success,
            failed=total_fail,
            output_file=args.output,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
