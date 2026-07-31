#!/bin/bash
#SBATCH -J chi5_refine_sin
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH -t 03:00:00
#SBATCH -o chi5_optimization/refine_sin/slurm-%j.out
#SBATCH -e chi5_optimization/refine_sin/slurm-%j.err

# Narrow-Delta refinement of the SiN hybrid winners (Delta brackets below the railed 0.015).
# Submit with:  sbatch -w node002 chi5_optimization/refine_sin.sbatch
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
python chi5_optimization/refine_sin.py --workers 75 --res 80 --decay 1e-4
