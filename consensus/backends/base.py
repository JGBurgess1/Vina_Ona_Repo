"""
Abstract base class for docking tool backends.

Each backend wraps a specific docking program (Vina, Smina, GNINA, rDock)
and provides a uniform interface for the consensus docking pipeline.
"""

import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BackendResult:
    """Standardized result from any docking backend."""
    ligand_path: str
    backend_name: str
    success: bool
    score: Optional[float] = None       # primary score (kcal/mol or equivalent)
    extra_scores: Optional[dict] = None  # backend-specific additional scores
    pose_file: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BackendConfig:
    """Configuration shared across all backends."""
    receptor_pdbqt: str
    center: list          # [x, y, z]
    box_size: list        # [sx, sy, sz]
    exhaustiveness: int = 8
    n_poses: int = 1
    seed: int = 42
    scoring_function: str = "vina"
    output_dir: str = "consensus_output"
    extra: dict = field(default_factory=dict)  # backend-specific options


class DockingBackend(ABC):
    """
    Abstract interface for a docking tool.

    Subclasses must implement:
      - name: property returning the tool name
      - is_available(): check if the tool is installed
      - prepare(): any one-time setup (e.g., receptor conversion, map generation)
      - dock_ligand(): dock a single ligand and return a BackendResult
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self._prepared = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend (e.g., 'vina', 'smina', 'gnina')."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the docking tool is installed and accessible."""
        ...

    @abstractmethod
    def prepare(self) -> None:
        """
        One-time preparation (receptor conversion, map computation, etc.).
        Called once before docking begins.
        """
        ...

    @abstractmethod
    def dock_ligand(self, ligand_path: str) -> BackendResult:
        """Dock a single ligand and return a standardized result."""
        ...

    def dock_batch(self, ligand_paths: list) -> list:
        """
        Dock a list of ligands sequentially.
        Override in subclasses for batch-optimized execution.
        """
        if not self._prepared:
            self.prepare()
            self._prepared = True

        results = []
        total = len(ligand_paths)
        for i, path in enumerate(ligand_paths):
            result = self.dock_ligand(path)
            results.append(result)
            if (i + 1) % 50 == 0 or (i + 1) == total:
                logger.info(
                    "[%s] Docked %d/%d ligands", self.name, i + 1, total
                )
        return results

    def _run_command(self, cmd: list, timeout: int = 300) -> subprocess.CompletedProcess:
        """Run a shell command with timeout and error handling."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{self.name}: command timed out after {timeout}s: {' '.join(cmd)}")
        except FileNotFoundError:
            raise RuntimeError(f"{self.name}: executable not found: {cmd[0]}")

    def _check_executable(self, name: str) -> bool:
        """Check if an executable is on PATH."""
        return shutil.which(name) is not None

    def _ensure_output_dir(self) -> str:
        """Create and return the backend-specific output directory."""
        out_dir = os.path.join(self.config.output_dir, self.name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
