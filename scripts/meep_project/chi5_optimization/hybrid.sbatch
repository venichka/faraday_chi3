#!/bin/bash
#SBATCH -J chi5_hybrid
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH -t 04:00:00
#SBATCH -o chi5_optimization/hybrid/slurm-%j.out
#SBATCH -e chi5_optimization/hybrid/slurm-%j.err

# Hybrid chi5 optimizer (Stage A analytic v2 pre-rank + Stage B 1D-FDTD op-point rank).
# Submit with:  sbatch -w nodeNNN --export=ALL,MAT=sin,NS=256,TOPK=10 chi5_optimization/hybrid.sbatch
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
python chi5_optimization/hybrid.py --material "${MAT:-sin}" --n-samples "${NS:-256}" \
    --topk "${TOPK:-10}" --workers 90 --res 80 --decay 1e-4
