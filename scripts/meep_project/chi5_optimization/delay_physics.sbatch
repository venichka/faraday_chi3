#!/bin/bash
#SBATCH --job-name=dphys
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=chi5_optimization/delay_physics/slurm-%x-%j.out
#SBATCH --error=chi5_optimization/delay_physics/slurm-%x-%j.err

# One slice of the delay_physics study per node. Slices are interleaved (k % n == i) so
# every node gets a mix of cheap (small |tau|) and expensive (large |tau|) runs, and the
# runner is idempotent, so slices may be re-run or overlapped safely.
#
#   sbatch --export=ALL,SLICE=0/8 chi5_optimization/delay_physics.sbatch
#
# 26 workers on a 40-core node: ~0.65 sims/core beat ~0.9 sims/core in earlier campaigns
# (memory-bandwidth contention dominates, not core count).

set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

echo "host=$(hostname)  slice=${SLICE}  start=$(date -Is)"
python chi5_optimization/delay_physics.py --workers "${WORKERS:-26}" --slice "${SLICE}"
echo "end=$(date -Is)"
