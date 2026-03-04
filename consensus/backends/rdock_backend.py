"""
rDock backend — cavity-based docking.

rDock uses a different approach from Vina-family tools: it defines the
binding site via a cavity mapping step and uses its own scoring function.
Input ligands must be in SD/MOL2 format (not PDBQT).

Workflow:
  1. Generate cavity with rbcavity using a reference ligand or sphere definition
  2. Dock with rbdock
  3. Parse scores from output SD file

CLI: rbcavity -r cavity.prm -was > cavity.as
     rbdock -r cavity.prm -p dock.prm -i ligand.sd -o output -n 1
"""

import logging
import os
import re
import tempfile
from typing import Optional

from .base import BackendConfig, BackendResult, DockingBackend

logger = logging.getLogger(__name__)


class RDockBackend(DockingBackend):
    """rDock cavity-based docking backend (CLI-based)."""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        # rDock-specific config
        self._receptor_mol2 = config.extra.get("rdock_receptor_mol2")
        self._reference_ligand = config.extra.get("rdock_reference_ligand")
        self._prm_file = config.extra.get("rdock_prm_file")
        self._cavity_radius = config.extra.get("rdock_cavity_radius", 6.0)
        self._docking_protocol = config.extra.get("rdock_protocol", "dock.prm")

    @property
    def name(self) -> str:
        return "rdock"

    def is_available(self) -> bool:
        return self._check_executable("rbdock") and self._check_executable("rbcavity")

    def prepare(self) -> None:
        out_dir = self._ensure_output_dir()

        if self._prm_file and os.path.exists(self._prm_file):
            logger.info("[rdock] Using existing PRM file: %s", self._prm_file)
        else:
            # Generate a PRM file from the box parameters
            self._prm_file = self._generate_prm(out_dir)

        # Run cavity generation
        cmd = ["rbcavity", "-r", self._prm_file, "-was"]
        try:
            result = self._run_command(cmd, timeout=120)
            if result.returncode != 0:
                logger.warning("[rdock] rbcavity stderr: %s", result.stderr[:300])
            logger.info("[rdock] Cavity generated")
        except Exception as e:
            logger.error("[rdock] Cavity generation failed: %s", e)
            raise

        self._prepared = True

    def _generate_prm(self, out_dir: str) -> str:
        """
        Generate an rDock .prm file from the BackendConfig parameters.
        Uses a sphere-based cavity definition centered on the box center.
        """
        prm_path = os.path.join(out_dir, "cavity.prm")

        # rDock needs receptor in MOL2 format
        receptor_path = self._receptor_mol2 or self.config.receptor_pdbqt
        ref_ligand = self._reference_ligand or ""

        cx, cy, cz = self.config.center
        radius = max(self.config.box_size) / 2.0

        prm_content = f"""RBT_PARAMETER_FILE_V1.00
TITLE rDock cavity definition (auto-generated)
RECEPTOR_FILE {receptor_path}
RECEPTOR_FLEX 3.0
"""
        if ref_ligand and os.path.exists(ref_ligand):
            prm_content += f"""
SECTION MAPPER
    SITE_MAPPER RbtLigandSiteMapper
    REF_MOL {ref_ligand}
    RADIUS {self._cavity_radius}
    SMALL_SPHERE 1.0
    MIN_VOLUME 100
    MAX_CAVITIES 1
    VOL_INCR 0.0
    GRIDSTEP 0.5
END_SECTION
"""
        else:
            # Sphere-based cavity (no reference ligand needed)
            prm_content += f"""
SECTION MAPPER
    SITE_MAPPER RbtSphereSiteMapper
    CENTER ({cx}, {cy}, {cz})
    RADIUS {radius}
    SMALL_SPHERE 1.0
    LARGE_SPHERE {radius}
    MAX_CAVITIES 1
END_SECTION
"""

        prm_content += """
SECTION CAVITY
    SCORING_FUNCTION RbtCavityGridSF
    WEIGHT 1.0
END_SECTION
"""

        with open(prm_path, "w") as f:
            f.write(prm_content)

        logger.info("[rdock] Generated PRM file: %s", prm_path)
        return prm_path

    def dock_ligand(self, ligand_path: str) -> BackendResult:
        if not self._prepared:
            self.prepare()

        out_dir = self._ensure_output_dir()
        basename = os.path.splitext(os.path.basename(ligand_path))[0]
        output_prefix = os.path.join(out_dir, f"{basename}_rdock")

        cmd = [
            "rbdock",
            "-r", self._prm_file,
            "-p", self._docking_protocol,
            "-i", ligand_path,
            "-o", output_prefix,
            "-n", str(self.config.n_poses),
            "-s", str(self.config.seed),
        ]

        try:
            result = self._run_command(cmd, timeout=300)
            output_sd = f"{output_prefix}.sd"
            score = self._parse_sd_score(output_sd)

            if score is None:
                return BackendResult(
                    ligand_path=ligand_path, backend_name=self.name,
                    success=False,
                    error=f"Could not parse score from {output_sd}: {result.stderr[:200]}",
                )

            return BackendResult(
                ligand_path=ligand_path, backend_name=self.name,
                success=True, score=score, pose_file=output_sd,
                extra_scores={"inter": score},
            )
        except Exception as e:
            return BackendResult(
                ligand_path=ligand_path, backend_name=self.name,
                success=False, error=str(e),
            )

    @staticmethod
    def _parse_sd_score(sd_path: str) -> Optional[float]:
        """
        Parse the best (lowest) SCORE from an rDock output SD file.
        rDock writes scores as SD properties: >  <SCORE>
        """
        if not os.path.exists(sd_path):
            return None

        scores = []
        in_score = False
        with open(sd_path, "r") as f:
            for line in f:
                if ">  <SCORE>" in line or "> <SCORE>" in line:
                    in_score = True
                    continue
                if in_score:
                    line = line.strip()
                    if line:
                        try:
                            scores.append(float(line))
                        except ValueError:
                            pass
                    in_score = False

        return min(scores) if scores else None
