"""
Input handling: config parsing, ligand discovery, and validation.
"""

import glob
import logging
import os
import sys

import yaml

from docking_engine import DockingConfig

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> DockingConfig:
    """Load docking configuration from a YAML file."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    receptor = raw.get("receptor")
    if not receptor:
        raise ValueError("Config must specify 'receptor' (path to PDBQT file)")

    center = raw.get("center")
    if not center or len(center) != 3:
        raise ValueError("Config must specify 'center' as [x, y, z]")

    box_size = raw.get("box_size")
    if not box_size or len(box_size) != 3:
        raise ValueError("Config must specify 'box_size' as [sx, sy, sz]")

    return DockingConfig(
        receptor_pdbqt=receptor,
        center=center,
        box_size=box_size,
        spacing=raw.get("spacing", 0.375),
        scoring_function=raw.get("scoring_function", "vina"),
        exhaustiveness=raw.get("exhaustiveness", 8),
        n_poses=raw.get("n_poses", 9),
        min_rmsd=raw.get("min_rmsd", 1.0),
        max_evals=raw.get("max_evals", 0),
        energy_range=raw.get("energy_range", 3.0),
        seed=raw.get("seed", 0),
        write_poses=raw.get("write_poses", True),
        output_dir=raw.get("output_dir", "output"),
        maps_dir=raw.get("maps_dir", "maps"),
    )


def discover_ligands(ligand_dir: str, pattern: str = "*.pdbqt") -> list:
    """
    Find all ligand PDBQT files in a directory (non-recursive by default).
    Returns sorted list of absolute paths.
    """
    search = os.path.join(ligand_dir, pattern)
    paths = sorted(glob.glob(search))
    if not paths:
        raise FileNotFoundError(
            f"No ligand files matching '{pattern}' found in {ligand_dir}"
        )
    return paths


def discover_ligands_recursive(ligand_dir: str, pattern: str = "*.pdbqt") -> list:
    """
    Find all ligand PDBQT files recursively under a directory.
    Returns sorted list of absolute paths.
    """
    search = os.path.join(ligand_dir, "**", pattern)
    paths = sorted(glob.glob(search, recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"No ligand files matching '{pattern}' found under {ligand_dir}"
        )
    return paths


def validate_inputs(config: DockingConfig, ligand_paths: list) -> None:
    """Validate that receptor and ligand files exist and are readable."""
    if not os.path.isfile(config.receptor_pdbqt):
        raise FileNotFoundError(
            f"Receptor file not found: {config.receptor_pdbqt}"
        )

    missing = [p for p in ligand_paths[:10] if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Ligand files not found (showing first {len(missing)}): {missing}"
        )

    logger.info(
        "Validated inputs: receptor=%s, ligands=%d",
        config.receptor_pdbqt,
        len(ligand_paths),
    )
