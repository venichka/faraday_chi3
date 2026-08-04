#!/bin/bash
#SBATCH --job-name=s3val3d
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-19
#SBATCH --output=chi5_dbr_design/runs/s3_validate/slurm-s3d-%A_%a.out
#SBATCH --error=chi5_dbr_design/runs/s3_validate/slurm-s3d-%A_%a.err

# Stage 3, the 3D half.  One MPI simulation per array task.
#
# THE TWO RULES THAT MAKE THIS WORK (both learned from failed submissions, 2026-08-01):
#   1. --ntasks=1 --cpus-per-task=24, NOT --ntasks=24.  With --ntasks=N the MPICH Hydra
#      launcher bootstraps through SLURM's PMI and dies instantly with
#      "PMI_Get_appnum returned -1", and every rank then runs the script standalone.
#   2. Only ONE program may run under mpirun.  A driver that spawned the simulation via
#      subprocess would have each of the 24 ranks fork its own non-MPI child, whose
#      `import meep` calls MPI_Init outside the job and aborts the same way.  So a SERIAL
#      --print-args call builds the argument list and mpirun runs the simulator directly.
# Four concurrent 24-rank tasks on one 96-core node00x are fine; contention was not the issue.

set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi

cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

# serial, rank-free: emits the argument list, or nothing if this job is already done
ARGS=$(python chi5_dbr_design/s3_validate.py --what 3d --print-args --index "${SLURM_ARRAY_TASK_ID}")

if [ -z "${ARGS}" ]; then
    echo "job ${SLURM_ARRAY_TASK_ID} already complete -- skipping"
    exit 0
fi

echo "mpirun -np 24 python faraday_meep_fp_circ.py ${ARGS}"
mpirun -np 24 python faraday_meep_fp_circ.py ${ARGS}
