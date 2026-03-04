"""
Parameter optimization engine for docking validation.

Supports two search strategies:
  1. Grid search: exhaustive evaluation of all parameter combinations
  2. Iterative refinement: coarse grid → zoom into best region → fine grid

Optimizable parameters:
  - box_size: [sx, sy, sz] dimensions in Angstroms
  - center: [x, y, z] offsets from the base center
  - exhaustiveness: Monte Carlo sampling depth
"""

import itertools
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ParameterRange:
    """Range specification for a single parameter."""
    name: str
    values: list  # explicit list of values to try

    @classmethod
    def from_range(cls, name: str, start: float, stop: float, step: float):
        """Create from start/stop/step."""
        values = list(np.arange(start, stop + step / 2, step))
        return cls(name=name, values=[round(v, 4) for v in values])

    @classmethod
    def from_list(cls, name: str, values: list):
        return cls(name=name, values=values)


@dataclass
class ParameterSet:
    """A single parameter configuration to evaluate."""
    center: list          # [x, y, z]
    box_size: list        # [sx, sy, sz]
    exhaustiveness: int
    label: str = ""       # human-readable label for this config

    def to_dict(self) -> dict:
        return {
            "center": self.center,
            "box_size": self.box_size,
            "exhaustiveness": self.exhaustiveness,
            "label": self.label,
        }


@dataclass
class OptimizationConfig:
    """Configuration for the parameter optimization campaign."""
    # Base docking config (receptor, scoring function, etc.)
    receptor_pdbqt: str
    base_center: list       # [x, y, z] starting center
    base_box_size: list     # [sx, sy, sz] starting box size
    scoring_function: str = "vina"
    spacing: float = 0.375
    n_poses: int = 1
    seed: int = 42

    # Parameter ranges to search
    box_size_range: Optional[list] = None     # list of [sx, sy, sz] to try
    center_offsets: Optional[list] = None     # list of [dx, dy, dz] offsets
    exhaustiveness_values: Optional[list] = None

    # Optimization settings
    metric: str = "roc_auc"           # metric to optimize: roc_auc, bedroc, log_auc, ef_1pct
    n_refinement_rounds: int = 1      # 1 = single grid, >1 = iterative refinement
    refinement_zoom: float = 0.5      # shrink factor for refinement

    # Output
    output_dir: str = "optimization_results"
    write_poses: bool = False

    # Backend-specific options (passed through to consensus backends)
    backend_options: dict = field(default_factory=dict)


def load_optimization_config(config_path: str) -> OptimizationConfig:
    """Load optimization configuration from YAML."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    receptor = raw["receptor"]
    base_center = raw["base_center"]
    base_box_size = raw["base_box_size"]

    # Parse box size range
    box_size_range = None
    if "box_size_range" in raw:
        bsr = raw["box_size_range"]
        if isinstance(bsr, dict) and "min" in bsr:
            # Generate from min/max/step (uniform for all dimensions)
            sizes = list(np.arange(
                bsr["min"], bsr["max"] + bsr.get("step", 2) / 2, bsr.get("step", 2)
            ))
            box_size_range = [[round(s, 1)] * 3 for s in sizes]
        elif isinstance(bsr, list):
            box_size_range = bsr

    # Parse center offsets
    center_offsets = None
    if "center_offsets" in raw:
        co = raw["center_offsets"]
        if isinstance(co, dict) and "range" in co:
            r = co["range"]
            step = co.get("step", 1.0)
            offsets_1d = list(np.arange(-r, r + step / 2, step))
            offsets_1d = [round(v, 2) for v in offsets_1d]
            # Generate 3D grid of offsets
            center_offsets = [
                [dx, dy, dz]
                for dx in offsets_1d
                for dy in offsets_1d
                for dz in offsets_1d
            ]
        elif isinstance(co, list):
            center_offsets = co

    # Parse exhaustiveness
    exhaustiveness_values = raw.get("exhaustiveness_values", [8])
    if isinstance(exhaustiveness_values, (int, float)):
        exhaustiveness_values = [int(exhaustiveness_values)]

    return OptimizationConfig(
        receptor_pdbqt=receptor,
        base_center=base_center,
        base_box_size=base_box_size,
        scoring_function=raw.get("scoring_function", "vina"),
        spacing=raw.get("spacing", 0.375),
        n_poses=raw.get("n_poses", 1),
        seed=raw.get("seed", 42),
        box_size_range=box_size_range,
        center_offsets=center_offsets,
        exhaustiveness_values=exhaustiveness_values,
        metric=raw.get("metric", "roc_auc"),
        n_refinement_rounds=raw.get("n_refinement_rounds", 1),
        refinement_zoom=raw.get("refinement_zoom", 0.5),
        output_dir=raw.get("output_dir", "optimization_results"),
        write_poses=raw.get("write_poses", False),
        backend_options=raw.get("backend_options", {}),
    )


def generate_parameter_grid(config: OptimizationConfig) -> list:
    """
    Generate all parameter combinations from the config ranges.
    Returns a list of ParameterSet objects.
    """
    box_sizes = config.box_size_range or [config.base_box_size]
    center_offsets = config.center_offsets or [[0.0, 0.0, 0.0]]
    exhaustiveness_vals = config.exhaustiveness_values or [8]

    param_sets = []
    for bs, co, ex in itertools.product(box_sizes, center_offsets, exhaustiveness_vals):
        center = [
            round(config.base_center[0] + co[0], 3),
            round(config.base_center[1] + co[1], 3),
            round(config.base_center[2] + co[2], 3),
        ]
        label = f"box={bs[0]}x{bs[1]}x{bs[2]}_center=[{center[0]},{center[1]},{center[2]}]_exh={ex}"
        param_sets.append(ParameterSet(
            center=center,
            box_size=bs,
            exhaustiveness=int(ex),
            label=label,
        ))

    logger.info(
        "Generated %d parameter combinations: %d box sizes x %d center offsets x %d exhaustiveness",
        len(param_sets),
        len(box_sizes),
        len(center_offsets),
        len(exhaustiveness_vals),
    )

    return param_sets


def refine_around_best(
    config: OptimizationConfig,
    best_params: ParameterSet,
    zoom: float = 0.5,
) -> list:
    """
    Generate a refined parameter grid centered on the best parameters.
    Shrinks the search range by the zoom factor.
    """
    # Refine box size: +/- 2 Angstroms around best, with finer step
    bs = best_params.box_size
    step = 1.0
    box_sizes = []
    for delta in np.arange(-2, 2 + step / 2, step):
        new_bs = [round(bs[0] + delta, 1), round(bs[1] + delta, 1), round(bs[2] + delta, 1)]
        if all(s >= 6.0 for s in new_bs):  # minimum box size
            box_sizes.append(new_bs)

    # Refine center: +/- 1 Angstrom around best, step 0.5
    c = best_params.center
    step = 0.5
    offsets_1d = list(np.arange(-1, 1 + step / 2, step))
    center_offsets = [
        [round(dx, 2), round(dy, 2), round(dz, 2)]
        for dx in offsets_1d
        for dy in offsets_1d
        for dz in offsets_1d
    ]

    # Keep exhaustiveness at best value (or try +/- 4)
    ex = best_params.exhaustiveness
    exhaustiveness_vals = sorted(set([max(4, ex - 4), ex, ex + 4]))

    param_sets = []
    for bs_new, co, ex_new in itertools.product(box_sizes, center_offsets, exhaustiveness_vals):
        center = [
            round(c[0] + co[0], 3),
            round(c[1] + co[1], 3),
            round(c[2] + co[2], 3),
        ]
        label = f"refined_box={bs_new[0]}x{bs_new[1]}x{bs_new[2]}_center=[{center[0]},{center[1]},{center[2]}]_exh={ex_new}"
        param_sets.append(ParameterSet(
            center=center,
            box_size=bs_new,
            exhaustiveness=int(ex_new),
            label=label,
        ))

    logger.info("Refinement grid: %d combinations around best params", len(param_sets))
    return param_sets


def write_optimized_config(
    config: OptimizationConfig,
    best_params: ParameterSet,
    output_path: str,
) -> None:
    """
    Write the optimized docking configuration as a YAML file
    compatible with run_docking.py from the Vina MPI pipeline.
    """
    optimized = {
        "receptor": config.receptor_pdbqt,
        "center": best_params.center,
        "box_size": best_params.box_size,
        "spacing": config.spacing,
        "scoring_function": config.scoring_function,
        "exhaustiveness": best_params.exhaustiveness,
        "n_poses": 9,
        "min_rmsd": 1.0,
        "max_evals": 0,
        "energy_range": 3.0,
        "seed": config.seed,
        "write_poses": True,
        "output_dir": "output/poses",
        "maps_dir": "maps",
    }

    with open(output_path, "w") as f:
        yaml.dump(optimized, f, default_flow_style=False, sort_keys=False)

    logger.info("Wrote optimized config to %s", output_path)
