#!/bin/bash
#SBATCH --job-name=sic_s0
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=chi5_sic_samples/runs/s0_intensity/slurm-s0-%j.out
#SBATCH --error=chi5_sic_samples/runs/s0_intensity/slurm-s0-%j.err

# Stage 0 pre-flight: intensity ladder to pick the reference pump intensity for the SiC
# samples. SiC's ~10x n2 means 1e12 W/cm^2 (fine for SiN) is already large-signal here.
set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
echo "host=$(hostname)"
python chi5_sic_samples/s0_intensity.py --workers 40
