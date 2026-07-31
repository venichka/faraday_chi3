#!/bin/bash
#SBATCH -J chi5_lsweep
#SBATCH -p defq
#SBATCH -w node002
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH -o chi5_optimization/clean_lsweep/slurm-%j.out
#SBATCH -e chi5_optimization/clean_lsweep/slurm-%j.err

# Clean per-geometry FDTD operating-point sweep (10 L x 10 center x 3 Delta, decay 1e-5).
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
python chi5_optimization/clean_lsweep.py
