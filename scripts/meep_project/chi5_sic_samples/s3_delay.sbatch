#!/bin/bash
#SBATCH --job-name=sic_dly
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=chi5_sic_samples/runs/s3_finalists/slurm-dly-%j.out
#SBATCH --error=chi5_sic_samples/runs/s3_finalists/slurm-dly-%j.err

# Predicted lab delay trace for the SiC finalists: effect and fringe vs envelope delay.
set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
echo "host=$(hostname)"
python chi5_sic_samples/s3_finalists.py --what delay --workers 45
