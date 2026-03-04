"""
Smina backend — a Vina fork with custom scoring functions.

Smina accepts the same PDBQT format and box parameters as Vina but supports
additional scoring functions (e.g., vinardo, custom weights).

CLI: smina --receptor R --ligand L --center_x X ... --out O
"""

import logging
import os
import re
from typing import Optional

from .base import BackendConfig, BackendResult, DockingBackend

logger = logging.getLogger(__name__)


class SminaBackend(DockingBackend):
    """Smina docking backend (CLI-based)."""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._scoring = config.extra.get("smina_scoring", "vinardo")

    @property
    def name(self) -> str:
        return "smina"

    def is_available(self) -> bool:
        return self._check_executable("smina")

    def prepare(self) -> None:
        self._ensure_output_dir()
        logger.info("[smina] Ready (scoring: %s)", self._scoring)
        self._prepared = True

    def dock_ligand(self, ligand_path: str) -> BackendResult:
        if not self._prepared:
            self.prepare()

        out_dir = self._ensure_output_dir()
        basename = os.path.splitext(os.path.basename(ligand_path))[0]
        pose_file = os.path.join(out_dir, f"{basename}_smina.pdbqt")

        cmd = [
            "smina",
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
            "--scoring", self._scoring,
            "--out", pose_file,
        ]

        try:
            result = self._run_command(cmd)
            score = self._parse_output(result.stdout)
            if score is None:
                return BackendResult(
                    ligand_path=ligand_path, backend_name=self.name,
                    success=False,
                    error=f"Could not parse score: {result.stderr[:200]}",
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
    def _parse_output(stdout: str) -> Optional[float]:
        """Parse best affinity from Smina output (same format as Vina)."""
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
