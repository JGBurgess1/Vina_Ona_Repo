# Vina MPI Docking

Parallel molecular docking of large compound libraries using AutoDock Vina and MPI.

## Architecture

Manager-worker pattern over MPI:

1. **Rank 0** computes receptor affinity maps once, writes them to shared filesystem
2. **All ranks** load pre-computed maps, receive an even chunk of ligands via `MPI.scatter`
3. Each rank docks its ligands sequentially (1 Vina instance per core, `cpu=1`)
4. **Rank 0** gathers results via `MPI.gather`, writes sorted CSV output

This avoids redundant map computation and minimizes inter-process communication.

## Prerequisites

- Python 3.8+
- MPI implementation (OpenMPI, MPICH, Intel MPI)
- AutoDock Vina Python bindings (`vina` package)
- `mpi4py`, `numpy`, `pyyaml`

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Dock 50,000 ligands across 600 cores
mpiexec -n 600 python run_docking.py \
    --config config/example.yaml \
    --ligands /data/ligand_library/ \
    --output output/results.csv

# Recursive ligand search
mpiexec -n 600 python run_docking.py \
    --config config/example.yaml \
    --ligands /data/ligand_library/ \
    --recursive

# With SLURM
srun -n 600 python run_docking.py \
    --config config/example.yaml \
    --ligands /data/ligand_library/
```

## Configuration

See `config/example.yaml`. Key parameters:

| Parameter | Description | Default |
|---|---|---|
| `receptor` | Path to receptor PDBQT file | required |
| `center` | Search box center [x, y, z] in Angstroms | required |
| `box_size` | Search box dimensions [sx, sy, sz] | required |
| `exhaustiveness` | Monte Carlo runs per ligand | 8 |
| `n_poses` | Max poses per ligand | 9 |
| `scoring_function` | `vina`, `vinardo`, or `ad4` | `vina` |
| `write_poses` | Write output PDBQT files | true |

## Input

- **Receptor**: Single PDBQT file (prepared with `prepare_receptor` from ADFR suite or similar)
- **Ligands**: Directory of PDBQT files (one per compound). Use `meeko` or `prepare_ligand` to convert from SDF/MOL2

## Output

- `results.csv` — All successful dockings sorted by binding energy (best first)
- `results_failed.csv` — Failed ligands with error messages (for re-processing)
- `output/poses/` — PDBQT files with docked poses (if `write_poses: true`)

## Performance

With `exhaustiveness=8` and typical drug-like molecules:
- ~10-30 seconds per ligand per core
- 50,000 ligands on 600 cores: ~15-45 minutes wall time
- Ligands are round-robin distributed for load balancing

## File Structure

```
vina-mpi-docking/
├── run_docking.py          # Entry point
├── docking_engine.py       # Vina wrapper (map prep, single-ligand docking)
├── mpi_orchestrator.py     # MPI scatter/gather coordination
├── input_handler.py        # Config parsing, ligand discovery, validation
├── results_writer.py       # CSV output and summary
├── config/
│   └── example.yaml        # Sample configuration
└── requirements.txt
```
