#!/bin/bash
#SBATCH --job-name=sic_3d
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-23
#SBATCH --output=chi5_sic_samples/runs/s3_finalists/slurm-3d-%A_%a.out
#SBATCH --error=chi5_sic_samples/runs/s3_finalists/slurm-3d-%A_%a.err

# 3D validation of the SiC finalists. One MPI simulation per array task.
# See the header of s3_finalists.py for the two MPI rules; --ntasks=1 --cpus-per-task=24 and a
# SERIAL --print-args call are both load-bearing.
set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

ARGS=$(python chi5_sic_samples/s3_finalists.py --what 3d --print-args \
        --index "${SLURM_ARRAY_TASK_ID}")
if [ -z "${ARGS}" ]; then
    echo "job ${SLURM_ARRAY_TASK_ID}: nothing to do (already complete or out of range)"
    exit 0
fi
echo "host=$(hostname)"
mpirun -np 24 python faraday_meep_fp_circ.py ${ARGS}
