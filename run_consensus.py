#!/usr/bin/env python
"""
Consensus docking using multiple open-source docking tools.

Docks a ligand library with multiple tools (Vina, Smina, GNINA, rDock),
normalizes scores, and computes consensus rankings using five methods.

Supports MPI parallelism: ligands are distributed across ranks, each rank
runs all enabled backends on its assigned ligands.

Usage:
    # Serial (single core):
    python run_consensus.py \
        --config config/consensus_example.yaml \
        --ligands /data/ligands/

    # MPI parallel:
    mpiexec -n 64 python run_consensus.py \
        --config config/consensus_example.yaml \
        --ligands /data/ligands/ \
        --mpi

    # Specific backends only:
    python run_consensus.py \
        --config config/consensus_example.yaml \
        --ligands /data/ligands/ \
        --backends vina smina gnina

    # Use optimized config from parameter optimization:
    python run_consensus.py \
        --config optimization_results/optimized_config.yaml \
        --ligands /data/ligands/ \
        --backends vina smina gnina
"""

import argparse
import csv
import glob
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

from consensus.backends.base import BackendConfig
from consensus.backends.vina_backend import VinaBackend
from consensus.backends.smina_backend import SminaBackend
from consensus.backends.gnina_backend import GninaBackend
from consensus.backends.rdock_backend import RDockBackend
from consensus.consensus_scoring import compute_all_consensus
from consensus.consensus_plots import generate_all_plots
from consensus.mpi_consensus import run_consensus_serial, run_consensus_mpi


# Registry of available backends
BACKEND_REGISTRY = {
    "vina": VinaBackend,
    "smina": SminaBackend,
    "gnina": GninaBackend,
    "rdock": RDockBackend,
}


def setup_logging(rank: int = 0, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if rank != 0:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s %(levelname)s [Rank {rank:03d}|%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consensus docking with multiple open-source tools",
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to YAML config (docking or consensus config)",
    )
    parser.add_argument(
        "--ligands", "-l", required=True,
        help="Directory containing ligand PDBQT files",
    )
    parser.add_argument(
        "--backends", "-b", nargs="+", default=None,
        choices=list(BACKEND_REGISTRY.keys()),
        help="Backends to use (default: all available)",
    )
    parser.add_argument(
        "--mpi", action="store_true",
        help="Enable MPI parallelism",
    )
    parser.add_argument(
        "--output-dir", "-o", default="consensus_results",
        help="Output directory (default: consensus_results/)",
    )
    parser.add_argument(
        "--pattern", "-p", default="*.pdbqt",
        help="Glob pattern for ligand files (default: *.pdbqt)",
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Search for ligands recursively",
    )
    parser.add_argument(
        "--vote-fraction", type=float, default=0.10,
        help="Top fraction for majority voting (default: 0.10)",
    )
    parser.add_argument(
        "--ecr-sigma", type=float, default=0.05,
        help="Sigma for ECR scoring (default: 0.05)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config. Accepts both docking configs and consensus-specific configs."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_backend_config(raw_config: dict, output_dir: str) -> BackendConfig:
    """Build a BackendConfig from a YAML config dict."""
    return BackendConfig(
        receptor_pdbqt=raw_config["receptor"],
        center=raw_config["center"],
        box_size=raw_config["box_size"],
        exhaustiveness=raw_config.get("exhaustiveness", 8),
        n_poses=raw_config.get("n_poses", 1),
        seed=raw_config.get("seed", 42),
        scoring_function=raw_config.get("scoring_function", "vina"),
        output_dir=output_dir,
        extra=raw_config.get("backend_options", {}),
    )


def discover_ligands(directory: str, pattern: str, recursive: bool) -> list:
    """Find ligand files."""
    if recursive:
        search = os.path.join(directory, "**", pattern)
        paths = sorted(glob.glob(search, recursive=True))
    else:
        search = os.path.join(directory, pattern)
        paths = sorted(glob.glob(search))

    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")
    return paths


def create_backends(
    backend_names: list,
    config: BackendConfig,
) -> list:
    """
    Instantiate requested backends, checking availability.
    Returns list of available DockingBackend instances.
    """
    backends = []
    for name in backend_names:
        cls = BACKEND_REGISTRY.get(name)
        if cls is None:
            logging.warning("Unknown backend: %s", name)
            continue
        backend = cls(config)
        if backend.is_available():
            backends.append(backend)
            logging.info("Backend enabled: %s", name)
        else:
            logging.warning(
                "Backend %s not available (executable not found), skipping", name
            )
    return backends


def write_consensus_csv(consensus_df: pd.DataFrame, output_path: str) -> None:
    """Write consensus results to CSV."""
    consensus_df.to_csv(output_path, index=False, float_format="%.4f")


def write_vina_compatible_csv(consensus_df: pd.DataFrame, output_path: str) -> None:
    """
    Write a simplified CSV compatible with Vina_ML_Pipeline's load_vina_results().
    Uses the average Z-score as the primary score.
    """
    simple = consensus_df[["consensus_rank", "ligand", "z_score_avg"]].copy()
    simple.columns = ["rank", "ligand", "best_energy_kcal"]
    simple.to_csv(output_path, index=False)


def print_summary(
    consensus_df: pd.DataFrame,
    score_matrix: pd.DataFrame,
    n_ranks: int = 1,
) -> None:
    """Print consensus docking summary."""
    tools = score_matrix.columns.tolist()

    print(f"\n{'='*78}")
    print(f"CONSENSUS DOCKING RESULTS")
    print(f"Tools: {', '.join(tools)} | MPI ranks: {n_ranks}")
    print(f"{'='*78}")

    # Per-tool stats
    print(f"\nPer-Tool Statistics:")
    print(f"  {'Tool':<12} {'N':<8} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for tool in tools:
        scores = score_matrix[tool].dropna()
        print(
            f"  {tool:<12} {len(scores):<8} {scores.mean():<10.3f} "
            f"{scores.median():<10.3f} {scores.min():<10.3f} {scores.max():<10.3f}"
        )

    # Top 10 consensus hits
    print(f"\nTop 10 Consensus Hits (by Average Rank):")
    print(f"  {'Rank':<6} {'Ligand':<30} {'AvgRank':<9} {'Z-Avg':<9} "
          f"{'ECR':<9} {'Votes':<7} {'Tools':<6}")
    print(f"  {'-'*6} {'-'*30} {'-'*9} {'-'*9} {'-'*9} {'-'*7} {'-'*6}")

    for _, row in consensus_df.head(10).iterrows():
        print(
            f"  {int(row['consensus_rank']):<6} {row['ligand']:<30} "
            f"{row['avg_rank']:<9.1f} {row['z_score_avg']:<9.3f} "
            f"{row['ecr_score']:<9.4f} {int(row['vote_count']):<7} "
            f"{int(row['n_tools_succeeded']):<6}"
        )

    print(f"{'='*78}\n")


def main() -> int:
    args = parse_args()

    rank = 0
    size = 1
    comm = None
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    setup_logging(rank, args.verbose)
    logger = logging.getLogger(__name__)

    t_start = time.time()

    # ---------------------------------------------------------------
    # Phase 1: Load config, discover ligands, create backends
    # ---------------------------------------------------------------
    raw_config = None
    ligand_paths = None
    backend_config = None

    if rank == 0:
        logger.info("Phase 1: Setup")
        if args.mpi:
            logger.info("MPI mode: %d ranks", size)

        raw_config = load_config(args.config)
        os.makedirs(args.output_dir, exist_ok=True)
        backend_config = build_backend_config(raw_config, args.output_dir)
        ligand_paths = discover_ligands(args.ligands, args.pattern, args.recursive)
        logger.info("Found %d ligands", len(ligand_paths))

    # Broadcast config to all ranks for backend creation
    if args.mpi:
        raw_config = comm.bcast(raw_config, root=0)
        backend_config = comm.bcast(
            build_backend_config(raw_config, args.output_dir) if rank != 0 else backend_config,
            root=0,
        )

    # Determine which backends to use
    requested = args.backends or list(BACKEND_REGISTRY.keys())
    backends = create_backends(requested, backend_config or build_backend_config(raw_config, args.output_dir))

    if not backends:
        if rank == 0:
            logger.error("No docking backends available. Install at least one of: vina, smina, gnina, rbdock")
        return 1

    if rank == 0:
        logger.info(
            "Enabled backends: %s", ", ".join(b.name for b in backends)
        )

    # ---------------------------------------------------------------
    # Phase 2: Run docking with all backends
    # ---------------------------------------------------------------
    if rank == 0:
        logger.info("Phase 2: Consensus docking")

    if args.mpi:
        all_results = run_consensus_mpi(backends, ligand_paths, comm=comm)
    else:
        all_results = run_consensus_serial(backends, ligand_paths)

    # ---------------------------------------------------------------
    # Phase 3: Compute consensus scores (rank 0 only)
    # ---------------------------------------------------------------
    if rank == 0 and all_results is not None:
        logger.info("Phase 3: Computing consensus scores")

        # Build ligand name list
        from consensus.consensus_scoring import _ligand_key
        ligand_names = [_ligand_key(p) for p in ligand_paths]

        consensus_df, score_matrix = compute_all_consensus(
            all_results, ligand_names,
            vote_fraction=args.vote_fraction,
            ecr_sigma=args.ecr_sigma,
        )

        # ---------------------------------------------------------------
        # Phase 4: Write outputs and generate plots
        # ---------------------------------------------------------------
        logger.info("Phase 4: Writing results and generating plots")

        # Full consensus CSV
        consensus_csv = os.path.join(args.output_dir, "consensus_results.csv")
        write_consensus_csv(consensus_df, consensus_csv)

        # ML Pipeline compatible CSV
        ml_csv = os.path.join(args.output_dir, "consensus_scores_for_ml.csv")
        write_vina_compatible_csv(consensus_df, ml_csv)

        # Plots
        plot_dir = os.path.join(args.output_dir, "plots")
        generate_all_plots(consensus_df, score_matrix, plot_dir)

        # Summary
        print_summary(consensus_df, score_matrix, n_ranks=size)

        t_end = time.time()
        logger.info("Consensus docking complete in %.1fs", t_end - t_start)

        print(f"Outputs:")
        print(f"  Consensus results: {consensus_csv}")
        print(f"  ML-compatible CSV: {ml_csv}")
        print(f"    -> Use with: python run_ml_pipeline.py --scores {ml_csv} --smiles ligands.smi")
        print(f"  Plots:             {plot_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
