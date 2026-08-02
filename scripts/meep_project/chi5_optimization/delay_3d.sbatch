#!/bin/bash
#SBATCH --job-name=d3d
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-31%20
#SBATCH --output=chi5_optimization/delay_3d/slurm-%A_%a.out
#SBATCH --error=chi5_optimization/delay_3d/slurm-%A_%a.err

# One 3D delay sim per array task, 24 MPI ranks each -- the same rank count as the reference
# 3D run (pipeline_cluster_20260325_114410, 7604 s, theta = 1.991 deg). %20 caps concurrency
# at 20 tasks = 480 cores, i.e. 4 tasks per 96-core node00x. Array tasks are idempotent
# (a completed sim is skipped), so the array can be resubmitted after a partial failure.
#
#   sbatch chi5_optimization/delay_3d.sbatch
#   python chi5_optimization/delay_3d.py --aggregate

set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

# IMPORTANT: request the cores as --ntasks=1 --cpus-per-task=24, NOT --ntasks=24. With
# --ntasks=24 the MPICH Hydra launcher tries to bootstrap through SLURM's PMI and dies with
# "PMI_Get_appnum returned -1" -- every rank then runs the script standalone and aborts.
# -launcher fork keeps Hydra spawning the ranks locally, which is what we want on one node.
echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID}  start=$(date -Is)"
# Build the argument list in a SERIAL call, then run the simulator itself under mpirun.
# Empty output means this delay/sub-sample is already done.
ARGS=$(python chi5_optimization/delay_3d.py --index "${SLURM_ARRAY_TASK_ID}" \
        --print-args --res "${RES:-30}" --decay "${DECAY:-1e-4}")
if [ -z "$ARGS" ]; then echo "task ${SLURM_ARRAY_TASK_ID} already complete"; exit 0; fi
echo "args: $ARGS"
mpirun -launcher fork -np 24 python faraday_meep_fp_circ.py $ARGS
echo "end=$(date -Is)"
