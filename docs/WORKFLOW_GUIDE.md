# End-to-End Workflow Guide

This guide describes the complete virtual screening pipeline spanning two repositories:

- **Vina_Ona_Repo** — Parameter optimization and large-scale MPI docking
- **Vina_ML_Pipeline** — ML-based scoring using molecular fingerprints

---

## Pipeline Overview

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                        PREPARATION                                      │
 │  Receptor PDBQT + Known actives + LUDe decoys                          │
 └────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 1: Parameter Optimization              (Vina_Ona_Repo)          │
 │                                                                         │
 │  run_optimize.py --mpi                                                  │
 │  • Generates grid of docking parameters                                 │
 │  • Docks actives + decoys for each combination                          │
 │  • Computes ROC AUC, BEDROC, enrichment factors                        │
 │  • Selects best parameters via iterative refinement                     │
 │                                                                         │
 │  Outputs:                                                               │
 │    optimized_config.yaml    ──→  Stage 2 input                         │
 │    best_docking_scores.csv  ──→  Stage 3 input (optional)              │
 │    plots/roc_overlay.png         (ROC curves, sensitivity analysis)     │
 └────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 2: Large-Scale Docking Campaign        (Vina_Ona_Repo)          │
 │                                                                         │
 │  mpiexec -n 600 python run_docking.py --config optimized_config.yaml   │
 │  • Distributes tens of thousands of ligands across 600 MPI ranks       │
 │  • Each rank docks its chunk using pre-computed affinity maps           │
 │  • Results gathered and sorted by binding energy                        │
 │                                                                         │
 │  Outputs:                                                               │
 │    results.csv              ──→  Stage 3 input                         │
 │    results_failed.csv            (for re-processing)                    │
 │    output/poses/*.pdbqt          (docked conformations)                 │
 └────────────────────────────┬────────────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 3: ML Scoring Pipeline                 (Vina_ML_Pipeline)       │
 │                                                                         │
 │  python run_ml_pipeline.py --scores results.csv --smiles ligands.smi   │
 │  • Generates 6 fingerprint types per molecule (RDKit)                   │
 │  • Exploratory data analysis (correlations, PCA, mutual information)    │
 │  • Trains 7 regression models × 6 fingerprint types (5-fold CV)        │
 │  • Compares performance: RMSE, R², Pearson r, MAE                      │
 │                                                                         │
 │  Outputs:                                                               │
 │    model_comparison.csv          (all 42 model+FP combinations)         │
 │    performance_heatmaps.png      (RMSE and R² heatmaps)                │
 │    prediction_scatter.png        (predicted vs actual)                  │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Parameter Optimization

### Purpose

Docking results are sensitive to the search box parameters. A box that is too small
may miss the binding site; too large wastes computation and reduces accuracy. The
exhaustiveness parameter controls sampling depth — too low gives inconsistent results,
too high wastes time.

This stage finds the parameter combination that best discriminates known active
compounds from property-matched decoys (e.g., LUDe decoys).

### Prerequisites

1. **Receptor PDBQT file** — prepared with `mk_prepare_receptor.py` (Meeko) or
   `prepare_receptor` (ADFR Suite).
2. **Active ligands** — known binders in PDBQT format. Typically 20-100 compounds
   with confirmed activity against the target.
3. **Decoy ligands** — property-matched non-binders in PDBQT format. Generated using
   LUDe (Ligand Unbiased Decoy Enrichment) or similar methods. Should match actives
   in MW, LogP, rotatable bonds, HBD/HBA but be topologically dissimilar. Typically
   10-50× more decoys than actives.

### Configuration

Edit `config/optimize_example.yaml`:

```yaml
receptor: /data/receptor.pdbqt
base_center: [15.190, 53.903, 16.917]   # active site coordinates
base_box_size: [20.0, 20.0, 20.0]       # starting box

box_size_range:
  min: 16
  max: 28
  step: 4          # → tries 16, 20, 24, 28 Å cubic boxes

center_offsets:
  range: 2.0
  step: 2.0        # → tries ±2 Å in each dimension (27 offsets)

exhaustiveness_values: [8, 16, 32]

metric: roc_auc    # what to maximize
n_refinement_rounds: 2
```

This generates 4 × 27 × 3 = 324 parameter combinations in the first round,
then refines around the best with a finer grid.

### Running

```bash
# MPI parallel (recommended for large grids)
mpiexec -n 32 python run_optimize.py \
    --config config/optimize_example.yaml \
    --actives /data/actives/ \
    --decoys /data/decoys/ \
    --mpi

# Serial (small grids or debugging)
python run_optimize.py \
    --config config/optimize_example.yaml \
    --actives /data/actives/ \
    --decoys /data/decoys/
```

### Interpreting Results

- **ROC AUC > 0.8**: Good discrimination. The docking setup reliably ranks actives
  above decoys.
- **ROC AUC 0.6-0.8**: Moderate. Consider adjusting the receptor preparation or
  trying flexible residues.
- **ROC AUC < 0.6**: Poor. The binding site definition or scoring function may need
  revision.
- **BEDROC**: More relevant than AUC for virtual screening because it emphasizes
  early enrichment (finding actives in the top few percent).
- **EF1%**: Practical metric — how many times better than random is the top 1%?
  EF1% > 10 is strong.

### Output Files

| File | Description | Used by |
|---|---|---|
| `optimized_config.yaml` | Best parameters as a docking config | Stage 2: `run_docking.py --config` |
| `best_docking_scores.csv` | Per-ligand scores from the best config | Stage 3: `run_ml_pipeline.py --scores` |
| `optimization_results.csv` | All parameter sets with metrics | Analysis |
| `plots/roc_overlay.png` | ROC curves for top configurations | Analysis |
| `plots/score_distributions.png` | Active vs decoy score histograms | Analysis |
| `plots/parameter_sensitivity.png` | How each parameter affects metrics | Analysis |

---

## Stage 2: Large-Scale Docking Campaign

### Purpose

Dock the full compound library (tens of thousands of ligands) using the optimized
parameters from Stage 1.

### Prerequisites

1. **Optimized config** from Stage 1 (`optimized_config.yaml`).
2. **Ligand library** — directory of PDBQT files. Use `mk_prepare_ligand.py` (Meeko)
   to convert from SDF.
3. **HPC cluster** with MPI (OpenMPI, MPICH, or Intel MPI).

### Running

```bash
# Standard MPI launch
mpiexec -n 600 python run_docking.py \
    --config optimization_results/optimized_config.yaml \
    --ligands /data/ligand_library/ \
    --output output/results.csv

# With SLURM
srun -n 600 python run_docking.py \
    --config optimization_results/optimized_config.yaml \
    --ligands /data/ligand_library/
```

### How It Works

1. **Rank 0** reads the config, computes receptor affinity maps, discovers all ligand files.
2. **Rank 0** broadcasts the map file path and scatters ligand paths evenly across all ranks.
3. **Each rank** loads the pre-computed maps, docks its assigned ligands sequentially.
4. **Rank 0** gathers all results, sorts by binding energy, writes CSV output.

Affinity maps are computed once and shared via the filesystem, avoiding redundant
computation across 600 ranks.

### Performance Expectations

| Exhaustiveness | Time per ligand | 50k ligands on 600 cores |
|---|---|---|
| 8 | ~10-15 sec | ~15 min |
| 16 | ~20-30 sec | ~30 min |
| 32 | ~40-60 sec | ~55 min |

Times vary with ligand flexibility (rotatable bonds) and box size.

### Output Files

| File | Description | Used by |
|---|---|---|
| `results.csv` | All dockings sorted by energy | Stage 3: `run_ml_pipeline.py --scores` |
| `results_failed.csv` | Failed ligands for re-processing | Re-run |
| `output/poses/*.pdbqt` | Docked conformations | Visualization (PyMOL, etc.) |

---

## Stage 3: ML Scoring Pipeline

### Purpose

Train ML models to predict docking scores from molecular fingerprints. This enables:
- Rapid pre-screening of new compounds without running Vina
- Understanding which structural features drive binding
- Comparing which fingerprint representation best captures binding information

### Prerequisites

1. **Docking results CSV** from Stage 2 (`results.csv`).
2. **Molecular structures** — SDF file or SMILES file matching the docked ligands.

### Running

```bash
# Full pipeline with real data
python run_ml_pipeline.py \
    --scores output/results.csv \
    --smiles /data/ligands.smi \
    --output-dir ml_results/

# EDA only (no model training)
python run_ml_pipeline.py \
    --scores output/results.csv \
    --smiles /data/ligands.smi \
    --skip-training

# Quick test with synthetic data
python run_ml_pipeline.py --synthetic --n-molecules 1000
```

### Pipeline Phases

**Phase 1: Data Loading**
- Parses the Vina results CSV
- Loads molecular structures from SDF or SMILES
- Matches ligand names between scores and structures

**Phase 2: Fingerprint Generation**
- Generates 6 fingerprint types per molecule using RDKit
- Each produces a binary feature matrix (n_molecules × n_bits)

**Phase 3: Exploratory Data Analysis**
- Score distribution (histogram + box plot)
- Per-fingerprint: bit variance, Pearson/Spearman correlations, mutual information
- PCA projections colored by docking score
- Inter-fingerprint similarity heatmap

**Phase 4: Model Training & Comparison**
- 7 models × 6 fingerprint types = 42 combinations
- 5-fold cross-validation with StandardScaler
- Metrics: RMSE, MAE, R², Pearson r, Spearman r
- Comparison bar charts, heatmaps, prediction scatter plots

### Interpreting Results

- **RMSE < 1.0 kcal/mol**: Good predictive accuracy for docking scores.
- **R² > 0.5**: The model explains more than half the variance in scores.
- **Pearson r > 0.7**: Strong linear correlation between predicted and actual.
- **Lasso performing well**: Suggests the relationship is approximately linear
  and only a few fingerprint bits are important.
- **Random Forest/GBM performing well**: Suggests non-linear relationships or
  feature interactions matter.
- **MACCS outperforming Morgan**: The 166 predefined substructure keys capture
  binding-relevant features better than hashed circular environments for this target.

### Output Files

| File | Description |
|---|---|
| `model_comparison.csv` | All 42 combinations with mean/std of each metric |
| `model_comparison_bars.png` | Grouped bar charts (RMSE, R², Pearson r, MAE) |
| `performance_heatmaps.png` | Fingerprint × model heatmaps |
| `prediction_scatter.png` | Predicted vs actual for each fingerprint |
| `plots/eda_*.png` | Per-fingerprint EDA plots |
| `plots/pca_comparison.png` | PCA projections |
| `plots/score_distribution.png` | Score histogram |

---

## Data Flow Between Stages

```
Stage 1 Output                    Stage 2 Input
─────────────────                 ─────────────────
optimized_config.yaml      ──→   run_docking.py --config optimized_config.yaml

Stage 2 Output                    Stage 3 Input
─────────────────                 ─────────────────
results.csv                ──→   run_ml_pipeline.py --scores results.csv

Stage 1 Output (alternative)      Stage 3 Input
─────────────────                 ─────────────────
best_docking_scores.csv    ──→   run_ml_pipeline.py --scores best_docking_scores.csv
```

The CSV formats are compatible across stages:
- `run_docking.py` outputs columns: `rank, ligand, best_energy_kcal, ...`
- `run_optimize.py` outputs columns: `rank, ligand, best_energy_kcal, is_active`
- `run_ml_pipeline.py` auto-detects columns containing "ligand" and "energy"/"score"

---

## Quick Start Example

```bash
# 1. Optimize parameters (32 MPI ranks)
cd Vina_Ona_Repo
mpiexec -n 32 python run_optimize.py \
    -c config/optimize_example.yaml \
    -a data/actives/ -d data/decoys/ --mpi

# 2. Run large-scale docking with optimized params (600 MPI ranks)
mpiexec -n 600 python run_docking.py \
    --config optimization_results/optimized_config.yaml \
    --ligands /data/50k_library/

# 3. Train ML models on the results
cd ../Vina_ML_Pipeline
python run_ml_pipeline.py \
    --scores ../Vina_Ona_Repo/output/results.csv \
    --smiles /data/50k_library.smi \
    -o ml_results/
```
