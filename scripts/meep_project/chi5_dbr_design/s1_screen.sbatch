#!/bin/bash
#SBATCH --job-name=s1screen
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=chi5_dbr_design/runs/s1_screen/slurm-s1-%j.out
#SBATCH --error=chi5_dbr_design/runs/s1_screen/slurm-s1-%j.err

# Stage 1: analytic (TMM) feasibility screen + coarse pre-rank. No FDTD, ~0.6 s/candidate.

set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi

cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

python chi5_dbr_design/s1_screen.py \
    --n-samples 32768 \
    --workers 40 \
    --topk 20
