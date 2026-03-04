"""
GNINA backend — CNN-scored molecular docking.

GNINA extends Vina with convolutional neural network scoring. It outputs
both a CNN affinity score and the traditional Vina score, making it
particularly valuable for consensus docking.

CLI: gnina --receptor R --ligand L --center_x X ... --out O
Output includes: CNNscore, CNNaffinity, and Vina affinity.
"""

import logging
import os
import re
from typing import Optional

from .base import BackendConfig, BackendResult, DockingBackend

logger = logging.getLogger(__name__)


class GninaBackend(DockingBackend):
    """GNINA CNN-scored docking backend (CLI-based)."""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._cnn_scoring = config.extra.get("gnina_cnn_scoring", "rescore")
        # rescore = dock with Vina, rescore with CNN (faster)
        # refinement = CNN-guided refinement (slower, more accurate)
        # none = Vina only (no CNN)

    @property
    def name(self) -> str:
        return "gnina"

    def is_available(self) -> bool:
        return self._check_executable("gnina")

    def prepare(self) -> None:
        self._ensure_output_dir()
        logger.info("[gnina] Ready (CNN scoring: %s)", self._cnn_scoring)
        self._prepared = True

    def dock_ligand(self, ligand_path: str) -> BackendResult:
        if not self._prepared:
            self.prepare()

        out_dir = self._ensure_output_dir()
        basename = os.path.splitext(os.path.basename(ligand_path))[0]
        pose_file = os.path.join(out_dir, f"{basename}_gnina.sdf")
        log_file = os.path.join(out_dir, f"{basename}_gnina.log")

        cmd = [
            "gnina",
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
            "--cnn_scoring", self._cnn_scoring,
            "--out", pose_file,
            "--log", log_file,
        ]

        try:
            result = self._run_command(cmd, timeout=600)
            scores = self._parse_output(result.stdout, log_file)
            if scores is None:
                return BackendResult(
                    ligand_path=ligand_path, backend_name=self.name,
                    success=False,
                    error=f"Could not parse scores: {result.stderr[:200]}",
                )

            return BackendResult(
                ligand_path=ligand_path, backend_name=self.name,
                success=True,
                score=scores["cnn_affinity"],
                extra_scores={
                    "cnn_score": scores.get("cnn_score"),
                    "cnn_affinity": scores["cnn_affinity"],
                    "vina_affinity": scores.get("vina_affinity"),
                },
                pose_file=pose_file,
            )
        except Exception as e:
            return BackendResult(
                ligand_path=ligand_path, backend_name=self.name,
                success=False, error=str(e),
            )

    @staticmethod
    def _parse_output(stdout: str, log_file: str = None) -> Optional[dict]:
        """
        Parse GNINA output for CNN and Vina scores.

        GNINA log format:
          mode |   affinity | CNN score | CNN affinity
          -----+------------+-----------+-------------
            1       -7.2       0.543        6.12
        """
        scores = {}

        # Try parsing stdout first, then log file
        text = stdout
        if log_file and os.path.exists(log_file):
            with open(log_file, "r") as f:
                text = text + "\n" + f.read()

        for line in text.splitlines():
            line = line.strip()
            # Look for the first mode line (mode 1 = best pose)
            if re.match(r"^\s*1\s+", line):
                parts = line.split()
                try:
                    if len(parts) >= 4:
                        scores["vina_affinity"] = float(parts[1])
                        scores["cnn_score"] = float(parts[2])
                        scores["cnn_affinity"] = float(parts[3])
                    elif len(parts) >= 2:
                        scores["vina_affinity"] = float(parts[1])
                        scores["cnn_affinity"] = float(parts[1])
                except (ValueError, IndexError):
                    continue
                if scores:
                    return scores

        return None if not scores else scores
