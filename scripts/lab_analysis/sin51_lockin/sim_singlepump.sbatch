#!/bin/bash
#SBATCH --job-name=sin51one
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=sim/slurm-one-%j.out
#SBATCH --error=sim/slurm-one-%j.err
set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
cd /Users/phi/nikita/code/faraday_chi3/scripts/lab_analysis/sin51_lockin
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
python sim_singlepump.py --sample sic --workers 2
