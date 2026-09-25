#!/bin/bash
#SBATCH --job-name=sic_s4
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=chi5_sic_samples/runs/s4_n2/slurm-s4-%j.out
#SBATCH --error=chi5_sic_samples/runs/s4_n2/slurm-s4-%j.err

# Stage 4: n2 sensitivity of effect / fringe / contrast (60 1D sims, ~21 min per slice).
#   sbatch --export=ALL,PART=0/2 chi5_sic_samples/s4_n2.sbatch
#   sbatch --export=ALL,PART=1/2 chi5_sic_samples/s4_n2.sbatch
set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
echo "host=$(hostname) part=${PART}"
python chi5_sic_samples/s4_n2.py --part "${PART}" --workers 30
