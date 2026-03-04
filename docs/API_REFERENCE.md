# Vina_Ona_Repo — API Reference

This document describes every module, class, and function in the Vina MPI Docking
and Parameter Optimization pipeline.

---

## Table of Contents

- [Entry Points](#entry-points)
  - [run_docking.py](#run_dockingpy)
  - [run_optimize.py](#run_optimizepy)
- [Core Docking Modules](#core-docking-modules)
  - [docking_engine.py](#docking_enginepy)
  - [mpi_orchestrator.py](#mpi_orchestratorpy)
  - [input_handler.py](#input_handlerpy)
  - [results_writer.py](#results_writerpy)
- [Optimizer Modules](#optimizer-modules)
  - [optimizer/roc_metrics.py](#optimizerroc_metricspy)
  - [optimizer/param_optimizer.py](#optimizerparam_optimizerpy)
  - [optimizer/validation_docker.py](#optimizervalidation_dockerpy)
  - [optimizer/optimization_plots.py](#optimizeroptimization_plotspy)
- [Configuration Files](#configuration-files)

---

## Entry Points

### `run_docking.py`

Main entry point for the large-scale MPI docking campaign. Parses CLI arguments,
loads configuration, discovers ligands, and delegates to `MPIOrchestrator`.

**Run with:**
```bash
mpiexec -n 600 python run_docking.py --config config/example.yaml --ligands /data/ligands/
```

#### Functions

| Function | Description |
|---|---|
| `setup_logging(rank, verbose)` | Configures logging. Rank 0 logs at INFO; workers at WARNING. |
| `parse_args()` | Parses CLI arguments: `--config`, `--ligands`, `--output`, `--recursive`, `--pattern`, `--verbose`. |
| `main()` | Orchestrates the pipeline: load config → broadcast → scatter ligands → dock → gather → write CSV. |

---

### `run_optimize.py`

Entry point for docking parameter optimization. Supports both serial and MPI-parallel
execution. Evaluates parameter combinations by docking known actives and decoys,
computing ROC/enrichment metrics, and selecting the best configuration.

**Run with:**
```bash
# Serial
python run_optimize.py -c config/optimize_example.yaml -a actives/ -d decoys/

# MPI parallel
mpiexec -n 32 python run_optimize.py -c config/optimize_example.yaml -a actives/ -d decoys/ --mpi
```

#### Functions

| Function | Signature | Description |
|---|---|---|
| `setup_logging` | `(rank=0, verbose=False)` | Configures per-rank logging. Non-zero ranks only log warnings. |
| `parse_args` | `() → Namespace` | Parses CLI: `--config`, `--actives`, `--decoys`, `--mpi`, `--refine`, `--metric`, `--output-dir`, `--pattern`, `--verbose`. |
| `write_results_csv` | `(results, output_path)` | Writes all parameter set evaluations to CSV with metrics and parameter values. |
| `write_docking_scores_csv` | `(results, output_path)` | Writes per-ligand scores from the best parameter set. Output is compatible with `Vina_ML_Pipeline`'s `load_vina_results()`. |
| `print_summary` | `(results, metric_name, n_ranks=1)` | Prints a ranked table of the top 10 configurations and the best configuration details. |
| `_run_round` | `(opt_config, param_sets, active_paths, decoy_paths, use_mpi, comm)` | Dispatches to `run_optimization()` (serial) or `run_optimization_mpi()` (parallel) based on the `--mpi` flag. |
| `main` | `() → int` | Four-phase pipeline: (1) load config + discover ligands, (2) grid search, (3) iterative refinement, (4) write outputs + plots. |

---

## Core Docking Modules

### `docking_engine.py`

Wraps AutoDock Vina's Python API for repeated docking of ligands against a single
receptor. Affinity maps are computed once and reused across all ligands.

#### Classes

##### `DockingResult`
Dataclass holding the result of a single ligand docking.

| Field | Type | Description |
|---|---|---|
| `ligand_path` | `str` | Path to the input PDBQT file. |
| `success` | `bool` | Whether docking completed without error. |
| `best_energy` | `float | None` | Best binding energy in kcal/mol (most negative = best). |
| `energies` | `list | None` | Full energy breakdown: `[total, inter, intra, torsions, intra_best_pose]`. |
| `n_poses` | `int` | Number of poses found. |
| `pose_file` | `str | None` | Path to output PDBQT with docked poses (if `write_poses=True`). |
| `error` | `str | None` | Error message if docking failed. |

##### `DockingConfig`
Dataclass holding all parameters for a docking campaign.

| Field | Type | Default | Description |
|---|---|---|---|
| `receptor_pdbqt` | `str` | required | Path to receptor PDBQT file. |
| `center` | `list[float]` | required | Search box center `[x, y, z]` in Angstroms. |
| `box_size` | `list[float]` | required | Search box dimensions `[sx, sy, sz]` in Angstroms. |
| `spacing` | `float` | `0.375` | Grid spacing in Angstroms. |
| `scoring_function` | `str` | `"vina"` | Scoring function: `vina`, `vinardo`, or `ad4`. |
| `exhaustiveness` | `int` | `8` | Number of Monte Carlo runs per ligand. Higher = more thorough but slower. |
| `n_poses` | `int` | `9` | Maximum poses to generate per ligand. |
| `min_rmsd` | `float` | `1.0` | Minimum RMSD between poses in Angstroms. |
| `max_evals` | `int` | `0` | Maximum evaluations (0 = use heuristics). |
| `energy_range` | `float` | `3.0` | Maximum energy difference from best pose in kcal/mol. |
| `seed` | `int` | `0` | Random seed for reproducibility. |
| `write_poses` | `bool` | `True` | Whether to write output PDBQT pose files. |
| `output_dir` | `str` | `"output"` | Directory for pose output files. |
| `maps_dir` | `str` | `"maps"` | Directory for affinity map files. |

##### `DockingEngine`
Main docking wrapper. One instance per MPI rank.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(config: DockingConfig, rank=0)` | Stores config and rank. Does not initialize Vina yet. |
| `prepare_maps` | `() → str` | Computes receptor affinity maps and writes them to disk. Returns the map prefix path. Should be called once (typically by rank 0). |
| `initialize` | `(map_prefix: str)` | Creates a Vina instance and loads pre-computed affinity maps. Called by every worker. |
| `dock_ligand` | `(ligand_path: str) → DockingResult` | Docks a single ligand. Sets the ligand, runs `dock()`, extracts energies, optionally writes poses. Catches exceptions per-ligand so one failure doesn't crash the batch. |
| `dock_batch` | `(ligand_paths: list) → list[DockingResult]` | Docks a list of ligands sequentially. Logs progress every 100 ligands. |

**Design decisions:**
- `cpu=1` per Vina instance because parallelism is at the MPI level, not within Vina.
- Maps are computed once and loaded by all workers to avoid redundant computation.
- Per-ligand exception handling prevents single failures from crashing the campaign.

---

### `mpi_orchestrator.py`

Coordinates parallel docking across MPI ranks using a scatter/gather pattern.

#### Functions

| Function | Signature | Description |
|---|---|---|
| `chunk_list` | `(lst, n_chunks) → list[list]` | Splits a list into `n_chunks` roughly equal parts using round-robin assignment. |

#### Classes

##### `MPIOrchestrator`
Manages the full MPI docking lifecycle.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(config: DockingConfig, ligand_paths=None)` | Initializes MPI communicator. `ligand_paths` only needed on rank 0. |
| `run` | `() → list[dict] | None` | Executes the four-phase pipeline: (1) rank 0 prepares maps, (2) broadcasts map prefix + scatters ligand chunks, (3) all ranks dock their chunk, (4) gathers results to rank 0. Returns flattened result dicts on rank 0, `None` on workers. |

**MPI communication pattern:**
1. `bcast(map_prefix)` — all ranks learn where to load maps from.
2. `scatter(chunks)` — each rank receives its ligand file paths.
3. Each rank calls `DockingEngine.dock_batch()` independently.
4. `gather(results)` — rank 0 collects all results.

---

### `input_handler.py`

Configuration parsing, ligand file discovery, and input validation.

| Function | Signature | Description |
|---|---|---|
| `load_config` | `(config_path: str) → DockingConfig` | Parses a YAML file into a `DockingConfig`. Validates that `receptor`, `center`, and `box_size` are present. |
| `discover_ligands` | `(ligand_dir, pattern="*.pdbqt") → list[str]` | Finds all matching ligand files in a directory (non-recursive). Returns sorted absolute paths. |
| `discover_ligands_recursive` | `(ligand_dir, pattern="*.pdbqt") → list[str]` | Same as above but searches subdirectories recursively. |
| `validate_inputs` | `(config: DockingConfig, ligand_paths: list)` | Checks that the receptor file exists and spot-checks the first 10 ligand paths. Raises `FileNotFoundError` on failure. |

---

### `results_writer.py`

Output writing and summary reporting for the docking campaign.

| Function | Signature | Description |
|---|---|---|
| `write_results_csv` | `(results, output_path, sort_by_energy=True)` | Writes successful docking results to CSV sorted by binding energy (best first). Columns: rank, ligand, best_energy_kcal, n_poses, energy breakdown, pose_file. Automatically writes a separate `_failed.csv` if any ligands failed. |
| `write_failed_csv` | `(failed, output_path)` | Writes failed docking attempts with error messages for re-processing. |
| `print_summary` | `(results: list)` | Prints statistics (total, success, failed, best/worst/mean energy) and a top-10 hits table to stdout. |

---

## Optimizer Modules

### `optimizer/roc_metrics.py`

Enrichment metrics for evaluating how well docking parameters discriminate
known actives from property-matched decoys.

All metrics treat docking scores as "lower is better" (more negative = stronger binding).
Internally, scores are negated where needed so that sklearn functions work correctly.

#### Classes

##### `EnrichmentMetrics`
Dataclass holding the full suite of metrics for one parameter configuration.

| Field | Type | Description |
|---|---|---|
| `roc_auc` | `float` | Area under the ROC curve. 1.0 = perfect, 0.5 = random. |
| `log_auc` | `float` | Semi-log AUC (log10 x-axis). Emphasizes early enrichment at low false positive rates. |
| `bedroc` | `float` | Boltzmann-Enhanced Discrimination of ROC (alpha=20). Weights early enrichment exponentially; alpha=20 means ~80% of the score comes from the top 8% of the ranked list. |
| `ef_1pct` | `float` | Enrichment factor at 1%: how many times more actives are found in the top 1% vs random expectation. |
| `ef_5pct` | `float` | Enrichment factor at 5%. |
| `ef_10pct` | `float` | Enrichment factor at 10%. |
| `n_actives` | `int` | Number of actives that docked successfully. |
| `n_decoys` | `int` | Number of decoys that docked successfully. |
| `fpr` | `ndarray | None` | False positive rate array for ROC curve plotting. |
| `tpr` | `ndarray | None` | True positive rate array for ROC curve plotting. |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `compute_roc_auc` | `(labels, scores) → (auc, fpr, tpr, thresholds)` | Computes ROC AUC. Negates scores internally (lower docking score = better binder = higher rank). |
| `compute_log_auc` | `(fpr, tpr, min_fpr=0.001) → float` | Integrates the ROC curve on a log10 x-axis from `min_fpr` to 1.0. Normalized by the log range. Useful for evaluating early enrichment. |
| `compute_bedroc` | `(labels, scores, alpha=20.0) → float` | BEDROC metric (Truchon & Bayly, 2007). Sorts compounds by score, computes exponentially-weighted sum of active ranks. Higher alpha = more emphasis on top-ranked compounds. |
| `compute_enrichment_factor` | `(labels, scores, fraction) → float` | Enrichment factor at a given fraction. Sorts by score, counts actives in the top `fraction` of the list, divides by random expectation. EF=1.0 means no better than random. |
| `compute_all_metrics` | `(labels, scores, store_curve=True) → EnrichmentMetrics` | Convenience function that computes all six metrics in one call. Optionally stores FPR/TPR arrays for plotting. |

---

### `optimizer/param_optimizer.py`

Parameter grid generation, iterative refinement, and configuration I/O.

#### Classes

##### `ParameterRange`
Dataclass for specifying a range of values for a single parameter.

| Method | Description |
|---|---|
| `from_range(name, start, stop, step)` | Creates from numeric range. |
| `from_list(name, values)` | Creates from explicit value list. |

##### `ParameterSet`
A single parameter configuration to evaluate.

| Field | Type | Description |
|---|---|---|
| `center` | `list[float]` | Box center `[x, y, z]` in Angstroms. |
| `box_size` | `list[float]` | Box dimensions `[sx, sy, sz]` in Angstroms. |
| `exhaustiveness` | `int` | Monte Carlo sampling depth. |
| `label` | `str` | Human-readable label (auto-generated). |

| Method | Description |
|---|---|
| `to_dict()` | Converts to a plain dict for serialization. |

##### `OptimizationConfig`
Full configuration for a parameter optimization campaign.

| Field | Type | Default | Description |
|---|---|---|---|
| `receptor_pdbqt` | `str` | required | Path to receptor PDBQT. |
| `base_center` | `list` | required | Starting box center. |
| `base_box_size` | `list` | required | Starting box dimensions. |
| `scoring_function` | `str` | `"vina"` | Scoring function. |
| `spacing` | `float` | `0.375` | Grid spacing. |
| `n_poses` | `int` | `1` | Poses per ligand (1 is sufficient for scoring). |
| `seed` | `int` | `42` | Random seed. |
| `box_size_range` | `list | None` | `None` | List of `[sx, sy, sz]` to try. |
| `center_offsets` | `list | None` | `None` | List of `[dx, dy, dz]` offsets from base center. |
| `exhaustiveness_values` | `list | None` | `None` | Exhaustiveness values to compare. |
| `metric` | `str` | `"roc_auc"` | Target metric to maximize. |
| `n_refinement_rounds` | `int` | `1` | 1 = single grid, >1 = iterative refinement. |
| `refinement_zoom` | `float` | `0.5` | Shrink factor per refinement round. |
| `output_dir` | `str` | `"optimization_results"` | Output directory. |
| `write_poses` | `bool` | `False` | Whether to save docked pose files. |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `load_optimization_config` | `(config_path) → OptimizationConfig` | Parses YAML config. Supports `box_size_range` as `{min, max, step}` dict or explicit list. Supports `center_offsets` as `{range, step}` dict (generates 3D grid) or explicit list. |
| `generate_parameter_grid` | `(config) → list[ParameterSet]` | Generates the Cartesian product of all box sizes × center offsets × exhaustiveness values. Each combination becomes a `ParameterSet`. |
| `refine_around_best` | `(config, best_params, zoom=0.5) → list[ParameterSet]` | Generates a finer grid centered on the best parameters: box size ±2 Å (step 1), center ±1 Å (step 0.5), exhaustiveness ±4. |
| `write_optimized_config` | `(config, best_params, output_path)` | Writes the winning parameters as a YAML file compatible with `run_docking.py`. |

---

### `optimizer/validation_docker.py`

Docks actives and decoys for parameter evaluation. Supports serial and MPI execution.

#### Classes

##### `ValidationResult`
Result of evaluating a single parameter set.

| Field | Type | Description |
|---|---|---|
| `params` | `ParameterSet` | The parameter configuration that was evaluated. |
| `metrics` | `EnrichmentMetrics` | ROC AUC, BEDROC, enrichment factors, etc. |
| `active_scores` | `ndarray` | Docking scores for successfully docked actives. |
| `decoy_scores` | `ndarray` | Docking scores for successfully docked decoys. |
| `all_scores` | `ndarray` | Combined scores (actives first, then decoys). |
| `all_labels` | `ndarray` | Binary labels (1=active, 0=decoy). |
| `n_active_failures` | `int` | Number of actives that failed to dock. |
| `n_decoy_failures` | `int` | Number of decoys that failed to dock. |
| `wall_time` | `float` | Wall clock time in seconds. |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `discover_ligands` | `(directory, pattern="*.pdbqt") → list[str]` | Finds ligand files in a directory. |
| `dock_validation_set` | `(opt_config, params, active_paths, decoy_paths, mpi_rank=0) → ValidationResult` | The core unit of work. For a single `ParameterSet`: builds a `DockingConfig`, computes affinity maps, docks all actives and decoys, computes enrichment metrics. Uses per-rank map directories to avoid file collisions in parallel execution. |
| `run_optimization` | `(opt_config, param_sets, active_paths, decoy_paths) → list[ValidationResult]` | **Serial execution.** Loops over all parameter sets, calls `dock_validation_set()` for each, returns results sorted by target metric. |
| `run_optimization_mpi` | `(opt_config, param_sets, active_paths, decoy_paths, comm=None) → list[ValidationResult] | None` | **MPI execution.** Rank 0 broadcasts config and ligand paths, scatters parameter sets round-robin. Each rank evaluates its chunk independently. Results are serialized, gathered to rank 0, deserialized, and sorted. Returns sorted results on rank 0, `None` on workers. |

**Internal helpers:**

| Function | Description |
|---|---|
| `_to_serializable(result)` | Converts `ValidationResult` to a pickle-safe dict (numpy arrays → lists). |
| `_from_serializable(d)` | Reconstructs `ValidationResult` from a serialized dict. |
| `_sort_results(results, metric_key)` | Sorts results by a metric field, descending. |
| `_chunk_round_robin(items, n_chunks)` | Distributes items across chunks using round-robin. |
| `_import_docking_engine()` | Lazy import of `DockingConfig` and `DockingEngine` to avoid loading Vina C++ bindings at module import time. |

---

### `optimizer/optimization_plots.py`

Visualization for parameter optimization results. All plots use the `Agg` backend
(headless-safe) and are saved as PNG files.

| Function | Signature | Description |
|---|---|---|
| `plot_roc_overlay` | `(results, output_dir, top_n=10)` | Overlays ROC curves for the top N parameter sets on a single plot with a random baseline. Color-coded by rank. |
| `plot_roc_semilog` | `(results, output_dir, top_n=5)` | Semi-log ROC plot (log10 x-axis) emphasizing early enrichment at low false positive rates. |
| `plot_score_distributions` | `(results, output_dir, top_n=4)` | Side-by-side histograms of active vs decoy docking scores for the top parameter sets. Good separation = good discrimination. |
| `plot_enrichment_bars` | `(results, output_dir, top_n=10)` | Grouped bar chart comparing EF1%, EF5%, EF10% across top parameter sets. |
| `plot_parameter_sensitivity` | `(results, output_dir)` | Scatter plots showing how each varying parameter (box size X/Y/Z, center X/Y/Z, exhaustiveness) affects ROC AUC and BEDROC. Only plots dimensions that actually vary. |
| `plot_metrics_heatmap` | `(results, output_dir, top_n=20)` | Heatmap of all six metrics across the top parameter configurations. |
| `generate_all_plots` | `(results, output_dir)` | Calls all six plot functions above. |

---

## Configuration Files

### `config/example.yaml`
Docking campaign configuration for `run_docking.py`. Defines receptor, search box,
scoring function, and docking parameters.

### `config/optimize_example.yaml`
Parameter optimization configuration for `run_optimize.py`. Defines the receptor,
base search box, parameter ranges to search (box size, center offsets, exhaustiveness),
target metric, and refinement settings.

Key YAML fields for the optimizer:

```yaml
# Box size search: cubic boxes from 16 to 28 Å
box_size_range:
  min: 16
  max: 28
  step: 4

# Center offset search: ±2 Å in each dimension
center_offsets:
  range: 2.0
  step: 2.0

# Exhaustiveness values to compare
exhaustiveness_values: [8, 16, 32]

# Metric to maximize
metric: roc_auc  # or bedroc, log_auc, ef_1pct, ef_5pct

# Iterative refinement
n_refinement_rounds: 2
```
