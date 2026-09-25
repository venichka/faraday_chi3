#!/bin/bash
#SBATCH --job-name=sin51sim
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=70
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=sim/slurm-sim-%j.out
#SBATCH --error=sim/slurm-sim-%j.err

# 1D runs at the lab's operating point on the as-built stacks (see sim_labpoint.py), 124 sims.
set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
cd /Users/phi/nikita/code/faraday_chi3/scripts/lab_analysis/sin51_lockin
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
echo "host=$(hostname)"
python sim_labpoint.py --workers 70
