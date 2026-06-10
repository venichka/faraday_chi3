#!/bin/bash
#SBATCH -J chi5_iscan
#SBATCH -p defq
#SBATCH -w cpu008
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH -o chi5_optimization/intensity_scaling/slurm-%j.out
#SBATCH -e chi5_optimization/intensity_scaling/slurm-%j.err

# Intensity-scaling confirmation (chi5 vs chi3) at small vs native Delta.
# 10 serial 1D runs, beat-resolved, all in parallel on one idle cpu node.
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
python chi5_optimization/run_intensity_scaling.py
