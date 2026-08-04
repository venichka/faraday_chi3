#!/bin/bash
#SBATCH --job-name=s4delay
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=chi5_dbr_design/runs/s4_delay/slurm-s4-%A_%a.out
#SBATCH --error=chi5_dbr_design/runs/s4_delay/slurm-s4-%A_%a.err

# Stage 4: predicted lab trace (V-H vs pump1 delay) for the new designs vs the fabricated one.
# 300 sims = 3 designs x 25 delays x 4 carrier sub-samples, fanned across six 40-core nodes.
# The pad is FIXED at 350 fs across the whole scan so pump1 is the only source whose timing
# changes for both signs of tau -- the delay-convention bug fixed in the earlier campaign.

set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi

cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

python chi5_dbr_design/s4_delay.py --workers 26 --slice "${SLURM_ARRAY_TASK_ID}/6"
