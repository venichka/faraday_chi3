#!/bin/bash
#SBATCH --job-name=s2fdtd
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --array=0-11
#SBATCH --output=chi5_dbr_design/runs/s2_fdtd/slurm-s2-%A_%a.out
#SBATCH --error=chi5_dbr_design/runs/s2_fdtd/slurm-s2-%A_%a.err

# Stage 2: 1D FDTD operating-point sweep + carrier-averaged ranking.
# Twelve array tasks fan the work across twelve 40-core cpu nodes at 26 workers each
# (0.6 sims/core -- packing a node is slower, memory bandwidth dominates).
# The runner is idempotent, so slices may overlap and the job may be re-run to fill gaps.
#
# SCREEN_SUBSAMPLES is set from the Stage-0 correlation result:
#   1 -> a single carrier phase ranks operating points like the carrier average
#   4 -> it does not, and the screen must be averaged too
: "${SCREEN_SUBSAMPLES:=4}"
: "${CONFIRM_TOP:=2}"

set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi

cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

python chi5_dbr_design/s2_fdtd.py \
    --workers 26 \
    --screen-subsamples "${SCREEN_SUBSAMPLES}" \
    --confirm-top "${CONFIRM_TOP}" \
    --phase screen \
    --slice "${SLURM_ARRAY_TASK_ID}/12"
