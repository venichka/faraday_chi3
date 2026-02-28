# Faraday Rotation Cavity (Meep)

This project simulates pump-probe nonlinear Faraday-like polarization rotation in a DBR-like cavity, optimizes cavity geometry, and runs pump-intensity sweeps.

## Main capabilities

- 1D and 3D field simulations (`faraday_meep_fp_circ.py`)
- Pump-intensity sweeps with aggregate diagnostics (`pump_intensity_sweep.py`)
- Geometry optimization with resonance/Q constraints (`optimize_cavity_geometry.py`)
- Optional Bayesian or Powell refinement
- Debug artifacts: epsilon profile, reflectance with mode markers, mode-overlap plots

## Core scripts

- `faraday_meep_fp_circ.py`: single simulation run and plots
- `optimize_cavity_geometry.py`: optimize geometry for rotation objective
- `pump_intensity_sweep.py`: sweep pump intensity and aggregate reports
- `fp_cavity_modes_spectrum.py`: cavity mode analysis helper
- `geometry_io.py`, `mode_targeting.py`, `material_fit.py`: IO/material/model utilities

## Environment setup

### Recommended (conda)

```bash
conda env create -f environment.yml
conda activate faraday-meep
```

### Optional pip extras

```bash
pip install -r requirements.txt
```

Note: `meep` is typically most reliable from `conda-forge`.

## Quick start

### 1) Single simulation (1D)

```bash
python faraday_meep_fp_circ.py \
  --dim 1 \
  --materials fit \
  --sin-fit si3n4.csv --sio2-fit sio2.csv \
  --geometry-file optimized_geometry_bayes_w6_q40_rerun.json \
  --cavity-modes-file cavity_modes_bayes_w6_q40_rerun.json \
  --resolution 100 \
  --decay-threshold 1e-3 \
  --pump-intensity 1e12 \
  --output-dir output_faraday_1d
```

### 2) Single simulation (3D)

```bash
python faraday_meep_fp_circ.py \
  --dim 3 \
  --materials fit \
  --sin-fit si3n4.csv --sio2-fit sio2.csv \
  --geometry-file optimized_geometry_bayes_w6_q40_rerun.json \
  --cavity-modes-file cavity_modes_bayes_w6_q40_rerun.json \
  --resolution 30 \
  --decay-threshold 1e-3 \
  --pump-intensity 1e12 \
  --output-dir output_faraday_3d
```

### 3) Optimize geometry

Use classic objective (`|theta|`):

```bash
python optimize_cavity_geometry.py \
  --optimizer bayes \
  --objective-metric abs_rotation \
  --workers 6 \
  --materials fit \
  --sin-fit si3n4.csv --sio2-fit sio2.csv \
  --pump-intensity 1e12 \
  --debug
```

Use quality-weighted objective:

```bash
python optimize_cavity_geometry.py \
  --optimizer bayes \
  --objective-metric quality_weighted_abs_rotation \
  --quality-std-ref-deg 15 \
  --workers 6 \
  --materials fit \
  --sin-fit si3n4.csv --sio2-fit sio2.csv \
  --pump-intensity 1e12 \
  --debug
```

Quality-weighted score is:

- `score = |theta_final| * quality_factor`
- `quality_factor = probe_quality_factor * source_quality_factor`
- `probe_quality_factor = DoLP_tail * sqrt(S0_tail_rel_max) * exp(-(theta_std/std_ref)^2)`
- `source_quality_factor = pump_dom_term * pump_purity_term * pump_balance_term`

### 4) Pump-intensity sweep

```bash
python pump_intensity_sweep.py \
  --dim 1 \
  --intensity-range 1e8 2e12 9 \
  --range-scale log \
  --workers 9 \
  --materials fit \
  --sin-fit si3n4.csv --sio2-fit sio2.csv \
  --geometry-file optimized_geometry_bayes_w6_q40_rerun.json \
  --cavity-modes-file cavity_modes_bayes_w6_q40_rerun.json \
  --resolution 30 \
  --decay-threshold 1e-6 \
  --output-root pump_sweep_outputs
```

## Output conventions

- Single-run simulation writes plots + `faraday_summary.json`
- Optimizer writes:
  - `optimized_geometry.json`
  - `cavity_modes.json`
  - `optimize_report.json`
- Sweep writes per-dimension folders and aggregate:
  - `pump_intensity_sweep_report.json`
  - `pump_intensity_sweep_summary.md`

## Parallelization notes

- Sweep parallelism: each intensity point is a separate process worker.
- Optimizer parallelism: candidate evaluations are parallelized with process workers.
- To improve throughput, sweep workers return lightweight metadata while heavy traces are stored in compressed per-run bundles (`sweep_trace_bundle.npz`).

## Cluster usage

Typical pattern (SLURM example):

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate faraday-meep

python pump_intensity_sweep.py --dim 1 --workers 16 ...
```

If you use MPI-enabled Meep, you can also wrap single-run commands with `mpirun`/`srun`; this repository’s sweep/optimizer process-level parallelism is independent from MPI.

## Tests

```bash
pytest -q
```

Tests cover:

- polarization/mapping math helpers
- objective-score parsing from summary data
- sweep trace-bundle serialization helpers

