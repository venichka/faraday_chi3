#!/bin/bash
#SBATCH -J sic_sweep1d
#SBATCH -p defq
#SBATCH -w cpu001
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o SiC_optimizations/sic_L3p2um/sweep_1d/slurm-%j.out
#SBATCH -e SiC_optimizations/sic_L3p2um/sweep_1d/slurm-%j.err

# SiC L=3.2um, 1D pump-intensity sweep. Same material/geometry/modes as the 1D runs.
# 12 intensity points (log 1e8..2e12 W/cm^2), one process worker per point.
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project

python pump_intensity_sweep.py \
  --dim 1 --mode full --materials fit \
  --sin-fit sic.csv --sio2-fit sio2.csv --fit-poles 3 --fit-window 600 2000 \
  --high-index-material sic \
  --geometry-file SiC_optimizations/sic_L3p2um/geometry.json \
  --cavity-modes-file SiC_optimizations/sic_L3p2um/cavity_modes.json \
  --resolution 100 --decay-threshold 1e-4 \
  --intensity-range 1e8 2e12 12 --range-scale log --workers 12 \
  --output-root SiC_optimizations/sic_L3p2um/sweep_1d
