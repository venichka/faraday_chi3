#!/bin/bash
#SBATCH --job-name=sin51s3d
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=sim/slurm-3d-%A_%a.out
#SBATCH --error=sim/slurm-3d-%A_%a.err

# 3D, tau = 0, as-built SIN51 at the lab's operating point: one carrier sub-sample per task.
# MPI recipe as in chi5_dbr_design/s3_3d.sbatch: --ntasks=1 --cpus-per-task=24, and a SERIAL
# --print-args call builds the argument list so only the simulator runs under mpirun.
set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
cd /Users/phi/nikita/code/faraday_chi3/scripts/lab_analysis/sin51_lockin
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
ARGS=$(python sim_labpoint.py --print-args-3d --index "${SLURM_ARRAY_TASK_ID}")
if [ -z "${ARGS}" ]; then echo "job ${SLURM_ARRAY_TASK_ID} already complete"; exit 0; fi
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
echo "mpirun -np 24 python faraday_meep_fp_circ.py ${ARGS}"
mpirun -np 24 python faraday_meep_fp_circ.py ${ARGS}
