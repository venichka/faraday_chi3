#!/bin/bash
#SBATCH -J chi5_refine_lowd
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH -t 03:00:00
#SBATCH -o chi5_optimization/refine_sin_lowd/slurm-%j.out
#SBATCH -e chi5_optimization/refine_sin_lowd/slurm-%j.err

# Low-Delta extension: bracket the turnover BELOW the railed 0.008, anchored on the refine best centers.
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
python chi5_optimization/refine_sin.py --workers 40 --res 80 --decay 1e-4 \
    --anchor refine --deltas=0.004,0.005,0.006,0.007 --center-offsets=-0.003,0.0,0.003 \
    --out chi5_optimization/refine_sin_lowd
