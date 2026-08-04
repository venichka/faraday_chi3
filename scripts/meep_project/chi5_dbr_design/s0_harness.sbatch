#!/bin/bash
#SBATCH --job-name=s0harness
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=chi5_dbr_design/runs/s0_harness/slurm-s0-%A_%a.out
#SBATCH --error=chi5_dbr_design/runs/s0_harness/slurm-s0-%A_%a.err

# Stage 0: harness validation + estimator-correlation study (72 serial 1D sims).
# Three array tasks, each on its own 40-core cpu node, 24 workers = 0.6 sims/core.
# Packing a node to its core count is SLOWER (memory-bandwidth contention): 26 sims on a
# 40-core node ran at 0.016 s/step vs 0.040 s/step for 61 sims on a 96-core node.
# The runner is idempotent (skips any point whose faraday_summary.json exists), so slices
# may overlap and be re-run safely.

set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi

cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

python chi5_dbr_design/s0_harness.py \
    --workers 24 \
    --slice "${SLURM_ARRAY_TASK_ID}/3"
