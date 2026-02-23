"""
Results collection and output writing.

Writes a CSV summary sorted by binding energy (best first) and a
separate file listing failed ligands for re-processing.
"""

import csv
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def write_results_csv(
    results: list,
    output_path: str,
    sort_by_energy: bool = True,
) -> None:
    """
    Write docking results to a CSV file.

    Args:
        results: list of result dicts from MPIOrchestrator.run()
        output_path: path to output CSV file
        sort_by_energy: if True, sort successful results by best_energy ascending
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if sort_by_energy:
        successful.sort(key=lambda r: r["best_energy"])

    fieldnames = [
        "rank",
        "ligand",
        "best_energy_kcal",
        "n_poses",
        "total",
        "inter",
        "intra",
        "torsions",
        "intra_best_pose",
        "pose_file",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, r in enumerate(successful, 1):
            energies = r.get("energies") or []
            row = {
                "rank": i,
                "ligand": os.path.basename(r["ligand_path"]),
                "best_energy_kcal": f"{r['best_energy']:.3f}",
                "n_poses": r["n_poses"],
                "total": f"{energies[0]:.3f}" if len(energies) > 0 else "",
                "inter": f"{energies[1]:.3f}" if len(energies) > 1 else "",
                "intra": f"{energies[2]:.3f}" if len(energies) > 2 else "",
                "torsions": f"{energies[3]:.3f}" if len(energies) > 3 else "",
                "intra_best_pose": f"{energies[4]:.3f}" if len(energies) > 4 else "",
                "pose_file": r.get("pose_file", ""),
            }
            writer.writerow(row)

    logger.info(
        "Wrote %d successful results to %s", len(successful), output_path
    )

    if failed:
        failed_path = output_path.replace(".csv", "_failed.csv")
        write_failed_csv(failed, failed_path)


def write_failed_csv(failed: list, output_path: str) -> None:
    """Write failed docking attempts to a separate CSV for re-processing."""
    fieldnames = ["ligand", "error"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in failed:
            writer.writerow({
                "ligand": r["ligand_path"],
                "error": r.get("error", "unknown"),
            })

    logger.info("Wrote %d failed results to %s", len(failed), output_path)


def print_summary(results: list) -> None:
    """Print a summary of docking results to stdout."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        print("No successful docking results.")
        return

    energies = [r["best_energy"] for r in successful]
    best = min(energies)
    worst = max(energies)
    mean = sum(energies) / len(energies)

    print(f"\n{'='*60}")
    print(f"DOCKING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total ligands:    {len(results)}")
    print(f"  Successful:       {len(successful)}")
    print(f"  Failed:           {len(failed)}")
    print(f"  Best energy:      {best:.3f} kcal/mol")
    print(f"  Worst energy:     {worst:.3f} kcal/mol")
    print(f"  Mean energy:      {mean:.3f} kcal/mol")
    print(f"{'='*60}")

    # Top 10 hits
    sorted_results = sorted(successful, key=lambda r: r["best_energy"])
    print(f"\nTop 10 hits:")
    print(f"  {'Rank':<6} {'Ligand':<40} {'Energy (kcal/mol)':<20}")
    print(f"  {'-'*6} {'-'*40} {'-'*20}")
    for i, r in enumerate(sorted_results[:10], 1):
        name = os.path.basename(r["ligand_path"])
        print(f"  {i:<6} {name:<40} {r['best_energy']:<20.3f}")
    print()
