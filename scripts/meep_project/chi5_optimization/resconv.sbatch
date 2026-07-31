#!/bin/bash
#SBATCH -J chi5_resconv
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o chi5_optimization/cross3d/slurm-resconv-%j.out
#SBATCH -e chi5_optimization/cross3d/slurm-resconv-%j.err

# 1D resolution convergence of cand07 at the cross-check operating point (center 0.69013, Delta 0.006).
# Submit with:  sbatch chi5_optimization/resconv.sbatch
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
python chi5_optimization/resconv.py --workers 8 --decay 1e-4
