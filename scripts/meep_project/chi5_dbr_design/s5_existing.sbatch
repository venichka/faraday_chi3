#!/bin/bash
#SBATCH --job-name=s5exist
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --array=0-7
#SBATCH --output=chi5_dbr_design/runs/s5_existing/slurm-s5-%A_%a.out
#SBATCH --error=chi5_dbr_design/runs/s5_existing/slurm-s5-%A_%a.err

# Stage 5: exhaustive operating-point map of the EXISTING fabricated sample (no refabrication).
# Geometry frozen at best_absolute; scans probe mode x pump center x Delta, carrier-averaged.
# PHASE is set by --export (A = coarse over all probe modes, B = fine around the winner).
: "${PHASE:=A}"
# optional: PROBE_ARG="--probe-nm 800.1,867.3" to refine specific probe modes
PROBE_ARG="${PROBE_ARG:-}"

set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi

cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

python chi5_dbr_design/s5_existing.py --phase "${PHASE}" --workers 26 ${PROBE_ARG} \
    --slice "${SLURM_ARRAY_TASK_ID}/8"
