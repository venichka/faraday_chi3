#!/bin/bash
#SBATCH --job-name=sin51flt
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=sim/slurm-flt-%j.out
#SBATCH --error=sim/slurm-flt-%j.err
set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
cd /Users/phi/nikita/code/faraday_chi3/scripts/lab_analysis/sin51_lockin
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
python sim_filter.py --workers 10
