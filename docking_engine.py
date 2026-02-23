"""
Core docking engine wrapping AutoDock Vina.

Each MPI worker instantiates a DockingEngine, loads pre-computed affinity
maps once, then docks many ligands sequentially. Using cpu=1 because
parallelism is at the MPI level (one Vina instance per core).
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

from vina import Vina

logger = logging.getLogger(__name__)


@dataclass
class DockingResult:
    """Result of a single ligand docking run."""
    ligand_path: str
    success: bool
    best_energy: Optional[float] = None
    energies: Optional[list] = None  # [total, inter, intra, torsions, intra_best]
    n_poses: int = 0
    pose_file: Optional[str] = None
    error: Optional[str] = None


@dataclass
class DockingConfig:
    """Configuration for a docking campaign."""
    receptor_pdbqt: str
    center: list  # [x, y, z]
    box_size: list  # [sx, sy, sz]
    spacing: float = 0.375
    scoring_function: str = "vina"
    exhaustiveness: int = 8
    n_poses: int = 9
    min_rmsd: float = 1.0
    max_evals: int = 0
    energy_range: float = 3.0
    seed: int = 0
    write_poses: bool = True
    output_dir: str = "output"
    maps_dir: str = "maps"


class DockingEngine:
    """
    Wraps AutoDock Vina for repeated docking of ligands against a single
    receptor. Affinity maps are computed once and reused.
    """

    def __init__(self, config: DockingConfig, rank: int = 0):
        self.config = config
        self.rank = rank
        self._vina: Optional[Vina] = None

    def prepare_maps(self) -> str:
        """
        Compute and write affinity maps. Should be called once by rank 0.
        Returns the map prefix path for workers to load.
        """
        os.makedirs(self.config.maps_dir, exist_ok=True)
        map_prefix = os.path.join(self.config.maps_dir, "receptor")

        v = Vina(
            sf_name=self.config.scoring_function,
            cpu=1,
            seed=self.config.seed,
            verbosity=0,
        )
        v.set_receptor(rigid_pdbqt_filename=self.config.receptor_pdbqt)
        v.compute_vina_maps(
            center=self.config.center,
            box_size=self.config.box_size,
            spacing=self.config.spacing,
        )
        v.write_maps(map_prefix_filename=map_prefix, overwrite=True)
        logger.info("Affinity maps written to %s", map_prefix)
        return map_prefix

    def initialize(self, map_prefix: str) -> None:
        """
        Initialize the Vina instance and load pre-computed maps.
        Called by every worker (including rank 0 after map generation).
        """
        self._vina = Vina(
            sf_name=self.config.scoring_function,
            cpu=1,
            seed=self.config.seed,
            verbosity=0,
        )
        self._vina.load_maps(map_prefix_filename=map_prefix)
        logger.debug("Rank %d: loaded maps from %s", self.rank, map_prefix)

    def dock_ligand(self, ligand_path: str) -> DockingResult:
        """
        Dock a single ligand file. Returns a DockingResult with energies
        and optionally writes pose files.
        """
        if self._vina is None:
            return DockingResult(
                ligand_path=ligand_path,
                success=False,
                error="Engine not initialized. Call initialize() first.",
            )

        try:
            self._vina.set_ligand_from_file(ligand_path)
            self._vina.dock(
                exhaustiveness=self.config.exhaustiveness,
                n_poses=self.config.n_poses,
                min_rmsd=self.config.min_rmsd,
                max_evals=self.config.max_evals,
            )

            energies = self._vina.energies(
                n_poses=self.config.n_poses,
                energy_range=self.config.energy_range,
            )

            n_poses_found = energies.shape[0]
            best_energy = float(energies[0][0])
            all_energies = energies[0].tolist()

            pose_file = None
            if self.config.write_poses:
                os.makedirs(self.config.output_dir, exist_ok=True)
                basename = os.path.splitext(os.path.basename(ligand_path))[0]
                pose_file = os.path.join(
                    self.config.output_dir, f"{basename}_out.pdbqt"
                )
                self._vina.write_poses(
                    pdbqt_filename=pose_file,
                    n_poses=self.config.n_poses,
                    energy_range=self.config.energy_range,
                    overwrite=True,
                )

            return DockingResult(
                ligand_path=ligand_path,
                success=True,
                best_energy=best_energy,
                energies=all_energies,
                n_poses=n_poses_found,
                pose_file=pose_file,
            )

        except Exception as e:
            logger.warning(
                "Rank %d: failed to dock %s: %s", self.rank, ligand_path, e
            )
            return DockingResult(
                ligand_path=ligand_path,
                success=False,
                error=str(e),
            )

    def dock_batch(self, ligand_paths: list) -> list:
        """
        Dock a list of ligands sequentially. Returns list of DockingResult.
        Logs progress every 100 ligands.
        """
        results = []
        total = len(ligand_paths)
        for i, path in enumerate(ligand_paths):
            result = self.dock_ligand(path)
            results.append(result)
            if (i + 1) % 100 == 0 or (i + 1) == total:
                logger.info(
                    "Rank %d: docked %d/%d ligands", self.rank, i + 1, total
                )
        return results
