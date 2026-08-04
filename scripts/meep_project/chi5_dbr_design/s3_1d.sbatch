#!/bin/bash
#SBATCH --job-name=s3val1d
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=chi5_dbr_design/runs/s3_validate/slurm-s3-%j.out
#SBATCH --error=chi5_dbr_design/runs/s3_validate/slurm-s3-%j.err

# Stage 3, the 1D half: intensity scaling (does it obey the chi5 I^2 law?) and fabrication
# tolerance (does it survive +-3/+-5 nm layer error?).  Pass WHAT=intensity or WHAT=tolerance.
#   sbatch --export=ALL,WHAT=intensity chi5_dbr_design/s3_1d.sbatch
# Run the intensity sweep FIRST -- the tolerance report normalises against its I=1e12 point.
: "${WHAT:=intensity}"

set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi

cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

# Optional fan-out: submit with --array=0-N and each task takes its own slice.  The tolerance
# study is 480 sims (5 finalists x 2 sigmas x 12 draws x 4 sub-samples), which is ~9 h on one
# node but ~1.5 h across six.  The runner is idempotent, so slices may overlap safely.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    SLICE_ARG=(--slice "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}")
else
    SLICE_ARG=()
fi

python chi5_dbr_design/s3_validate.py --what "${WHAT}" --workers 26 "${SLICE_ARG[@]}"
