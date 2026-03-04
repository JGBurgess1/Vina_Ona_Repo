#!/usr/bin/env python
"""
Docking parameter optimization using known actives and property-matched decoys.

Evaluates docking parameter combinations (box size, center, exhaustiveness)
by docking actives and decoys, computing ROC AUC and enrichment metrics,
and selecting the configuration that best discriminates actives from decoys.

Supports both serial and MPI-parallel execution. In MPI mode, parameter sets
are distributed across ranks — each rank independently evaluates its assigned
configurations, then results are gathered and ranked.

The optimized parameters are written as a YAML config file compatible with
run_docking.py (Vina MPI pipeline) and the results CSV is compatible with
run_ml_pipeline.py (Vina ML Pipeline).

Usage:
    # Serial (single core):
    python run_optimize.py \
        --config config/optimize_example.yaml \
        --actives data/actives/ \
        --decoys data/decoys/

    # MPI parallel (distribute parameter sets across ranks):
    mpiexec -n 32 python run_optimize.py \
        --config config/optimize_example.yaml \
        --actives data/actives/ \
        --decoys data/decoys/ \
        --mpi

    # With iterative refinement (2 rounds):
    mpiexec -n 32 python run_optimize.py \
        --config config/optimize_example.yaml \
        --actives data/actives/ \
        --decoys data/decoys/ \
        --mpi --refine 2

    # Custom metric and output:
    mpiexec -n 32 python run_optimize.py \
        --config config/optimize_example.yaml \
        --actives data/actives/ \
        --decoys data/decoys/ \
        --mpi --metric bedroc --output-dir my_optimization/

    # Multi-tool optimization (optimize params for each backend independently):
    mpiexec -n 64 python run_optimize.py \
        --config config/optimize_example.yaml \
        --actives data/actives/ \
        --decoys data/decoys/ \
        --mpi --backends vina smina gnina

    # Optimize for consensus docking (outputs per-tool configs):
    python run_optimize.py \
        --config config/optimize_example.yaml \
        --actives data/actives/ \
        --decoys data/decoys/ \
        --backends vina smina gnina rdock
"""

import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
import yaml

from optimizer.param_optimizer import (
    OptimizationConfig,
    generate_parameter_grid,
    load_optimization_config,
    refine_around_best,
    write_optimized_config,
)
from optimizer.optimization_plots import generate_all_plots
from optimizer.validation_docker import (
    ValidationResult,
    discover_ligands,
    run_optimization,
    run_optimization_mpi,
)

AVAILABLE_BACKENDS = ["vina", "smina", "gnina", "rdock"]


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
        description="Optimize docking parameters using actives/decoys validation",
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to optimization YAML config file",
    )
    parser.add_argument(
        "--actives", "-a", required=True,
        help="Directory containing active ligand PDBQT files",
    )
    parser.add_argument(
        "--decoys", "-d", required=True,
        help="Directory containing decoy ligand PDBQT files (e.g., LUDe decoys)",
    )
    parser.add_argument(
        "--backends", "-b", nargs="+", default=None,
        choices=AVAILABLE_BACKENDS,
        help="Docking backends to optimize (default: Vina only). "
             "When multiple backends are specified, each is optimized independently.",
    )
    parser.add_argument(
        "--mpi", action="store_true",
        help="Enable MPI parallelism (distribute parameter sets across ranks)",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory (default: from config file)",
    )
    parser.add_argument(
        "--refine", type=int, default=None,
        help="Number of refinement rounds (overrides config)",
    )
    parser.add_argument(
        "--metric", "-m", default=None,
        choices=["roc_auc", "bedroc", "log_auc", "ef_1pct", "ef_5pct"],
        help="Metric to optimize (overrides config)",
    )
    parser.add_argument(
        "--pattern", "-p", default="*.pdbqt",
        help="Glob pattern for ligand files (default: *.pdbqt)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def write_results_csv(results: list, output_path: str) -> None:
    """
    Write optimization results to CSV.
    Format is compatible with the Vina ML Pipeline's data_loader.
    """
    fieldnames = [
        "rank", "label",
        "box_size_x", "box_size_y", "box_size_z",
        "center_x", "center_y", "center_z",
        "exhaustiveness",
        "roc_auc", "log_auc", "bedroc",
        "ef_1pct", "ef_5pct", "ef_10pct",
        "n_actives", "n_decoys",
        "active_failures", "decoy_failures",
        "wall_time_sec",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results, 1):
            writer.writerow({
                "rank": i,
                "label": r.params.label,
                "box_size_x": r.params.box_size[0],
                "box_size_y": r.params.box_size[1],
                "box_size_z": r.params.box_size[2],
                "center_x": r.params.center[0],
                "center_y": r.params.center[1],
                "center_z": r.params.center[2],
                "exhaustiveness": r.params.exhaustiveness,
                "roc_auc": f"{r.metrics.roc_auc:.4f}",
                "log_auc": f"{r.metrics.log_auc:.4f}",
                "bedroc": f"{r.metrics.bedroc:.4f}",
                "ef_1pct": f"{r.metrics.ef_1pct:.2f}",
                "ef_5pct": f"{r.metrics.ef_5pct:.2f}",
                "ef_10pct": f"{r.metrics.ef_10pct:.2f}",
                "n_actives": r.metrics.n_actives,
                "n_decoys": r.metrics.n_decoys,
                "active_failures": r.n_active_failures,
                "decoy_failures": r.n_decoy_failures,
                "wall_time_sec": f"{r.wall_time:.1f}",
            })


def write_docking_scores_csv(results: list, output_path: str) -> None:
    """
    Write per-ligand docking scores from the best parameter set.
    Compatible with Vina_ML_Pipeline's load_vina_results().
    """
    best = results[0]
    n_actives = len(best.active_scores)
    n_decoys = len(best.decoy_scores)

    fieldnames = ["rank", "ligand", "best_energy_kcal", "is_active"]

    all_scores = np.concatenate([best.active_scores, best.decoy_scores])
    all_labels = np.concatenate([
        np.ones(n_actives, dtype=int),
        np.zeros(n_decoys, dtype=int),
    ])
    all_names = (
        [f"active_{i:04d}" for i in range(n_actives)]
        + [f"decoy_{i:04d}" for i in range(n_decoys)]
    )

    # Sort by score (best first)
    order = np.argsort(all_scores)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, idx in enumerate(order, 1):
            writer.writerow({
                "rank": rank,
                "ligand": all_names[idx],
                "best_energy_kcal": f"{all_scores[idx]:.3f}",
                "is_active": int(all_labels[idx]),
            })


def print_summary(results: list, metric_name: str, n_ranks: int = 1) -> None:
    """Print optimization summary to stdout."""
    print(f"\n{'='*72}")
    print(f"DOCKING PARAMETER OPTIMIZATION RESULTS")
    print(f"Optimized metric: {metric_name} | MPI ranks: {n_ranks}")
    print(f"{'='*72}")

    print(f"\n  {'Rank':<5} {'AUC':<8} {'BEDROC':<8} {'LogAUC':<8} "
          f"{'EF1%':<7} {'EF5%':<7} {'Box':<18} {'Exh':<5}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*18} {'-'*5}")

    for i, r in enumerate(results[:10], 1):
        m = r.metrics
        bs = r.params.box_size
        print(
            f"  {i:<5} {m.roc_auc:<8.4f} {m.bedroc:<8.4f} {m.log_auc:<8.4f} "
            f"{m.ef_1pct:<7.1f} {m.ef_5pct:<7.1f} "
            f"{bs[0]}x{bs[1]}x{bs[2]:<8} {r.params.exhaustiveness:<5}"
        )

    best = results[0]
    print(f"\n  BEST CONFIGURATION:")
    print(f"    Center:          {best.params.center}")
    print(f"    Box size:        {best.params.box_size}")
    print(f"    Exhaustiveness:  {best.params.exhaustiveness}")
    print(f"    ROC AUC:         {best.metrics.roc_auc:.4f}")
    print(f"    BEDROC:          {best.metrics.bedroc:.4f}")
    print(f"    EF 1%:           {best.metrics.ef_1pct:.1f}")
    print(f"{'='*72}\n")


def _run_round(opt_config, param_sets, active_paths, decoy_paths, use_mpi, comm):
    """Execute one optimization round (serial or MPI)."""
    if use_mpi:
        return run_optimization_mpi(
            opt_config, param_sets, active_paths, decoy_paths, comm=comm,
        )
    else:
        return run_optimization(
            opt_config, param_sets, active_paths, decoy_paths,
        )


def main() -> int:
    args = parse_args()

    # Determine MPI rank before logging setup
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
    # Phase 1: Load configuration and discover ligands
    # Rank 0 handles file I/O; MPI bcast is done inside
    # run_optimization_mpi().
    # ---------------------------------------------------------------
    opt_config = None
    active_paths = None
    decoy_paths = None

    if rank == 0:
        logger.info("Phase 1: Loading configuration and discovering ligands")
        if args.mpi:
            logger.info("MPI mode: %d ranks available", size)

        opt_config = load_optimization_config(args.config)
        if args.output_dir:
            opt_config.output_dir = args.output_dir
        if args.metric:
            opt_config.metric = args.metric
        if args.refine is not None:
            opt_config.n_refinement_rounds = args.refine

        os.makedirs(opt_config.output_dir, exist_ok=True)

        active_paths = discover_ligands(args.actives, args.pattern)
        decoy_paths = discover_ligands(args.decoys, args.pattern)

        logger.info(
            "Found %d actives and %d decoys", len(active_paths), len(decoy_paths)
        )

    # ---------------------------------------------------------------
    # Branch: multi-tool or single-tool optimization
    # ---------------------------------------------------------------
    use_multitool = args.backends is not None and len(args.backends) > 0

    if use_multitool:
        return _main_multitool(
            args, opt_config, active_paths, decoy_paths,
            rank, size, comm, t_start,
        )
    else:
        return _main_single_tool(
            args, opt_config, active_paths, decoy_paths,
            rank, size, comm, t_start,
        )


def _main_single_tool(args, opt_config, active_paths, decoy_paths,
                       rank, size, comm, t_start) -> int:
    """Original single-tool (Vina) optimization path."""
    logger = logging.getLogger(__name__)

    # Phase 2: Generate parameter grid and run optimization
    param_sets = None
    if rank == 0:
        logger.info("Phase 2: Parameter grid search (Vina)")
        param_sets = generate_parameter_grid(opt_config)

    all_results = _run_round(
        opt_config, param_sets, active_paths, decoy_paths, args.mpi, comm,
    )

    # Phase 3: Iterative refinement
    n_refinement_rounds = 1
    if rank == 0 and opt_config is not None:
        n_refinement_rounds = opt_config.n_refinement_rounds

    if args.mpi:
        n_refinement_rounds = comm.bcast(n_refinement_rounds, root=0)

    for round_i in range(1, n_refinement_rounds):
        refined_sets = None
        if rank == 0:
            logger.info(
                "Phase 3: Refinement round %d/%d",
                round_i, n_refinement_rounds - 1,
            )
            best_params = all_results[0].params
            refined_sets = refine_around_best(
                opt_config, best_params, zoom=opt_config.refinement_zoom,
            )

        refined_results = _run_round(
            opt_config, refined_sets, active_paths, decoy_paths, args.mpi, comm,
        )

        if rank == 0:
            all_results.extend(refined_results)
            metric_key = opt_config.metric
            all_results.sort(
                key=lambda r: getattr(r.metrics, metric_key, r.metrics.roc_auc),
                reverse=True,
            )

    # Phase 4: Write outputs (rank 0 only)
    if rank == 0 and all_results is not None:
        logger.info("Phase 4: Writing results and generating plots")

        results_csv = os.path.join(opt_config.output_dir, "optimization_results.csv")
        write_results_csv(all_results, results_csv)

        scores_csv = os.path.join(opt_config.output_dir, "best_docking_scores.csv")
        write_docking_scores_csv(all_results, scores_csv)

        best_config_path = os.path.join(opt_config.output_dir, "optimized_config.yaml")
        write_optimized_config(opt_config, all_results[0].params, best_config_path)

        plot_dir = os.path.join(opt_config.output_dir, "plots")
        generate_all_plots(all_results, plot_dir)

        print_summary(all_results, opt_config.metric, n_ranks=size)

        t_end = time.time()
        logger.info("Optimization complete in %.1fs", t_end - t_start)

        print(f"Outputs:")
        print(f"  Optimized config:  {best_config_path}")
        print(f"    -> Use with: mpiexec -n 600 python run_docking.py --config {best_config_path}")
        print(f"  Docking scores:    {scores_csv}")
        print(f"    -> Use with: python run_ml_pipeline.py --scores {scores_csv} --smiles ligands.smi")
        print(f"  Full results:      {results_csv}")
        print(f"  Plots:             {plot_dir}/")

    return 0


def _main_multitool(args, opt_config, active_paths, decoy_paths,
                     rank, size, comm, t_start) -> int:
    """
    Multi-tool optimization: optimize parameters for each backend independently.
    Produces per-tool optimized configs and cross-tool comparison.
    """
    logger = logging.getLogger(__name__)

    from optimizer.multitool_validator import (
        run_multitool_optimization,
        run_multitool_optimization_mpi,
    )
    from optimizer.multitool_plots import generate_multitool_plots

    backend_names = args.backends

    # Phase 2: Generate parameter grid
    param_sets = None
    if rank == 0:
        param_sets = generate_parameter_grid(opt_config)
        logger.info(
            "Phase 2: Multi-tool grid search — %d backends x %d param sets = %d jobs",
            len(backend_names), len(param_sets),
            len(backend_names) * len(param_sets),
        )

    # Run optimization
    if args.mpi:
        results_by_tool = run_multitool_optimization_mpi(
            opt_config, backend_names, param_sets,
            active_paths, decoy_paths, comm=comm,
        )
    else:
        results_by_tool = run_multitool_optimization(
            opt_config, backend_names, param_sets,
            active_paths, decoy_paths,
        )

    # Phase 3: Iterative refinement per tool
    n_refinement_rounds = 1
    if rank == 0 and opt_config is not None:
        n_refinement_rounds = opt_config.n_refinement_rounds

    if args.mpi:
        n_refinement_rounds = comm.bcast(n_refinement_rounds, root=0)

    for round_i in range(1, n_refinement_rounds):
        if rank == 0:
            logger.info(
                "Phase 3: Refinement round %d/%d",
                round_i, n_refinement_rounds - 1,
            )
            # Refine around each tool's best params
            refined_sets_by_tool = {}
            for bname in backend_names:
                if results_by_tool[bname]:
                    best = results_by_tool[bname][0].params
                    refined = refine_around_best(
                        opt_config, best, zoom=opt_config.refinement_zoom,
                    )
                    refined_sets_by_tool[bname] = refined

            # Flatten into a single param_sets list for all tools
            all_refined = []
            refined_backend_names = []
            for bname, rsets in refined_sets_by_tool.items():
                all_refined.extend(rsets)
                refined_backend_names.extend([bname] * len(rsets))
        else:
            all_refined = None
            refined_backend_names = None

        # For refinement, we need to run per-tool with their specific refined sets
        # Reuse the multitool runner with the union of all refined sets
        if rank == 0:
            # Group refined sets per tool
            refined_by_tool = {}
            for bname in backend_names:
                if bname in (refined_sets_by_tool if rank == 0 else {}):
                    refined_by_tool[bname] = refined_sets_by_tool[bname]

            # Run each tool's refinement with its own refined grid
            for bname, rsets in refined_by_tool.items():
                if args.mpi:
                    from optimizer.multitool_validator import run_multitool_optimization_mpi
                    refined_results = run_multitool_optimization_mpi(
                        opt_config, [bname], rsets,
                        active_paths, decoy_paths, comm=comm,
                    )
                else:
                    from optimizer.multitool_validator import run_multitool_optimization
                    refined_results = run_multitool_optimization(
                        opt_config, [bname], rsets,
                        active_paths, decoy_paths,
                    )

                if refined_results and bname in refined_results:
                    results_by_tool[bname].extend(refined_results[bname])
                    metric_key = opt_config.metric
                    results_by_tool[bname].sort(
                        key=lambda r: getattr(r.metrics, metric_key, r.metrics.roc_auc),
                        reverse=True,
                    )

    # Phase 4: Write outputs (rank 0 only)
    if rank == 0 and results_by_tool is not None:
        logger.info("Phase 4: Writing multi-tool results and plots")

        # Per-tool optimized configs and results CSVs
        for bname in backend_names:
            results = results_by_tool.get(bname, [])
            if not results:
                logger.warning("No results for %s", bname)
                continue

            # Per-tool results CSV
            tool_csv = os.path.join(
                opt_config.output_dir, f"optimization_results_{bname}.csv"
            )
            write_results_csv(results, tool_csv)

            # Per-tool optimized config
            tool_config_path = os.path.join(
                opt_config.output_dir, f"optimized_config_{bname}.yaml"
            )
            write_optimized_config(opt_config, results[0].params, tool_config_path)

        # Cross-tool comparison CSV
        comparison_csv = os.path.join(opt_config.output_dir, "tool_comparison.csv")
        _write_tool_comparison_csv(results_by_tool, comparison_csv)

        # Plots
        plot_dir = os.path.join(opt_config.output_dir, "plots")
        generate_multitool_plots(results_by_tool, plot_dir)

        # Summary
        _print_multitool_summary(results_by_tool, opt_config.metric, n_ranks=size)

        t_end = time.time()
        logger.info("Multi-tool optimization complete in %.1fs", t_end - t_start)

        print(f"\nOutputs:")
        for bname in backend_names:
            if results_by_tool.get(bname):
                cfg = os.path.join(opt_config.output_dir, f"optimized_config_{bname}.yaml")
                print(f"  {bname} config: {cfg}")
        print(f"  Tool comparison:   {comparison_csv}")
        print(f"  Plots:             {plot_dir}/")
        print(f"\n  Use per-tool configs with run_consensus.py:")
        print(f"    python run_consensus.py --config <tool_config> --ligands /data/ligands/")

    return 0


def _write_tool_comparison_csv(results_by_tool: dict, output_path: str) -> None:
    """Write a side-by-side comparison of the best params per tool."""
    fieldnames = [
        "tool", "roc_auc", "log_auc", "bedroc",
        "ef_1pct", "ef_5pct", "ef_10pct",
        "box_size_x", "box_size_y", "box_size_z",
        "center_x", "center_y", "center_z",
        "exhaustiveness", "n_param_sets_evaluated",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tool_name, results in results_by_tool.items():
            if not results:
                continue
            best = results[0]
            writer.writerow({
                "tool": tool_name,
                "roc_auc": f"{best.metrics.roc_auc:.4f}",
                "log_auc": f"{best.metrics.log_auc:.4f}",
                "bedroc": f"{best.metrics.bedroc:.4f}",
                "ef_1pct": f"{best.metrics.ef_1pct:.2f}",
                "ef_5pct": f"{best.metrics.ef_5pct:.2f}",
                "ef_10pct": f"{best.metrics.ef_10pct:.2f}",
                "box_size_x": best.params.box_size[0],
                "box_size_y": best.params.box_size[1],
                "box_size_z": best.params.box_size[2],
                "center_x": best.params.center[0],
                "center_y": best.params.center[1],
                "center_z": best.params.center[2],
                "exhaustiveness": best.params.exhaustiveness,
                "n_param_sets_evaluated": len(results),
            })


def _print_multitool_summary(results_by_tool: dict, metric_name: str, n_ranks: int = 1) -> None:
    """Print multi-tool optimization summary."""
    print(f"\n{'='*78}")
    print(f"MULTI-TOOL PARAMETER OPTIMIZATION RESULTS")
    print(f"Optimized metric: {metric_name} | MPI ranks: {n_ranks}")
    print(f"{'='*78}")

    print(f"\n  {'Tool':<10} {'AUC':<8} {'BEDROC':<8} {'LogAUC':<8} "
          f"{'EF1%':<7} {'Box':<18} {'Exh':<5} {'Evaluated':<10}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*18} {'-'*5} {'-'*10}")

    for tool_name, results in results_by_tool.items():
        if not results:
            print(f"  {tool_name:<10} (no results)")
            continue
        best = results[0]
        m = best.metrics
        bs = best.params.box_size
        print(
            f"  {tool_name:<10} {m.roc_auc:<8.4f} {m.bedroc:<8.4f} {m.log_auc:<8.4f} "
            f"{m.ef_1pct:<7.1f} {bs[0]}x{bs[1]}x{bs[2]:<8} "
            f"{best.params.exhaustiveness:<5} {len(results):<10}"
        )

    # Identify overall best tool
    best_tool = None
    best_metric = -1
    for tool_name, results in results_by_tool.items():
        if results:
            val = getattr(results[0].metrics, metric_name, results[0].metrics.roc_auc)
            if val > best_metric:
                best_metric = val
                best_tool = tool_name

    if best_tool:
        print(f"\n  BEST TOOL: {best_tool} ({metric_name}={best_metric:.4f})")

    print(f"{'='*78}\n")


if __name__ == "__main__":
    sys.exit(main())
