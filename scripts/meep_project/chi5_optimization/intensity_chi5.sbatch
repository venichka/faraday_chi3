#!/bin/bash
#SBATCH --job-name=ichi5
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=chi5_optimization/intensity_chi5/slurm-%x-%j.out
#SBATCH --error=chi5_optimization/intensity_chi5/slurm-%x-%j.err
set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
echo "host=$(hostname) slice=${SLICE} start=$(date -Is)"
python chi5_optimization/intensity_chi5.py --workers "${WORKERS:-26}" --slice "${SLICE}"
echo "end=$(date -Is)"
