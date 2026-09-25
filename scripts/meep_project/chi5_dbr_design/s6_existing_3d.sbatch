#!/bin/bash
#SBATCH --job-name=s6ex3d
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=chi5_dbr_design/runs/s6_existing_3d/slurm-s6-%A_%a.out
#SBATCH --error=chi5_dbr_design/runs/s6_existing_3d/slurm-s6-%A_%a.err

# Stage 6: 3D check of the existing sample's two accessible optima. Same MPI recipe as
# s3_3d.sbatch: --ntasks=1 --cpus-per-task=24, and a SERIAL --print-args call builds the
# argument list so only the simulator itself runs under mpirun.

set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi

cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

ARGS=$(python chi5_dbr_design/s6_existing_3d.py --print-args --index "${SLURM_ARRAY_TASK_ID}")

if [ -z "${ARGS}" ]; then
    echo "job ${SLURM_ARRAY_TASK_ID} already complete -- skipping"
    exit 0
fi

echo "mpirun -np 24 python faraday_meep_fp_circ.py ${ARGS}"
mpirun -np 24 python faraday_meep_fp_circ.py ${ARGS}
