"""
AutoDock Vina backend using the Python API.

Uses the same DockingEngine from the MPI pipeline for consistency.
Falls back to CLI if the Python bindings are unavailable.
"""

import logging
import os
import re
from typing import Optional

from .base import BackendConfig, BackendResult, DockingBackend

logger = logging.getLogger(__name__)


class VinaBackend(DockingBackend):
    """AutoDock Vina docking backend."""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._engine = None
        self._use_cli = False

    @property
    def name(self) -> str:
        return "vina"

    def is_available(self) -> bool:
        try:
            from vina import Vina
            return True
        except ImportError:
            return self._check_executable("vina")

    def prepare(self) -> None:
        out_dir = self._ensure_output_dir()
        try:
            from vina import Vina
            v = Vina(sf_name=self.config.scoring_function, cpu=1,
                     seed=self.config.seed, verbosity=0)
            v.set_receptor(rigid_pdbqt_filename=self.config.receptor_pdbqt)
            v.compute_vina_maps(
                center=self.config.center,
                box_size=self.config.box_size,
            )
            maps_dir = os.path.join(out_dir, "maps")
            os.makedirs(maps_dir, exist_ok=True)
            map_prefix = os.path.join(maps_dir, "receptor")
            v.write_maps(map_prefix_filename=map_prefix, overwrite=True)
            self._map_prefix = map_prefix
            self._use_cli = False
            logger.info("[vina] Prepared maps via Python API")
        except ImportError:
            self._use_cli = True
            logger.info("[vina] Python API unavailable, using CLI")
        self._prepared = True

    def dock_ligand(self, ligand_path: str) -> BackendResult:
        if not self._prepared:
            self.prepare()

        if self._use_cli:
            return self._dock_cli(ligand_path)
        return self._dock_api(ligand_path)

    def _dock_api(self, ligand_path: str) -> BackendResult:
        try:
            from vina import Vina
            v = Vina(sf_name=self.config.scoring_function, cpu=1,
                     seed=self.config.seed, verbosity=0)
            v.load_maps(map_prefix_filename=self._map_prefix)
            v.set_ligand_from_file(ligand_path)
            v.dock(exhaustiveness=self.config.exhaustiveness,
                   n_poses=self.config.n_poses)
            energies = v.energies(n_poses=self.config.n_poses)
            best = float(energies[0][0])

            pose_file = None
            out_dir = self._ensure_output_dir()
            basename = os.path.splitext(os.path.basename(ligand_path))[0]
            pose_file = os.path.join(out_dir, f"{basename}_vina.pdbqt")
            v.write_poses(pdbqt_filename=pose_file, n_poses=self.config.n_poses,
                          overwrite=True)

            return BackendResult(
                ligand_path=ligand_path, backend_name=self.name,
                success=True, score=best, pose_file=pose_file,
                extra_scores={"energies": energies[0].tolist()},
            )
        except Exception as e:
            return BackendResult(
                ligand_path=ligand_path, backend_name=self.name,
                success=False, error=str(e),
            )

    def _dock_cli(self, ligand_path: str) -> BackendResult:
        out_dir = self._ensure_output_dir()
        basename = os.path.splitext(os.path.basename(ligand_path))[0]
        pose_file = os.path.join(out_dir, f"{basename}_vina.pdbqt")

        cmd = [
            "vina",
            "--receptor", self.config.receptor_pdbqt,
            "--ligand", ligand_path,
            "--center_x", str(self.config.center[0]),
            "--center_y", str(self.config.center[1]),
            "--center_z", str(self.config.center[2]),
            "--size_x", str(self.config.box_size[0]),
            "--size_y", str(self.config.box_size[1]),
            "--size_z", str(self.config.box_size[2]),
            "--exhaustiveness", str(self.config.exhaustiveness),
            "--num_modes", str(self.config.n_poses),
            "--seed", str(self.config.seed),
            "--out", pose_file,
        ]

        try:
            result = self._run_command(cmd)
            score = self._parse_vina_output(result.stdout)
            if score is None:
                return BackendResult(
                    ligand_path=ligand_path, backend_name=self.name,
                    success=False, error=f"Could not parse score: {result.stderr}",
                )
            return BackendResult(
                ligand_path=ligand_path, backend_name=self.name,
                success=True, score=score, pose_file=pose_file,
            )
        except Exception as e:
            return BackendResult(
                ligand_path=ligand_path, backend_name=self.name,
                success=False, error=str(e),
            )

    @staticmethod
    def _parse_vina_output(stdout: str) -> Optional[float]:
        """Parse best score from Vina CLI output."""
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("1"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        continue
        return None
