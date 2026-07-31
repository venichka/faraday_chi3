#!/bin/bash
#SBATCH -J dscan_main
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=88
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o chi5_optimization/delay_scan/slurm-main-%j.out
#SBATCH -e chi5_optimization/delay_scan/slurm-main-%j.err

# Stage 1 (61, carrier-locked delay scan) + fine carrier check (24) = 85 sims, one full batch.
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
python chi5_optimization/delay_scan.py --stage1 --fine --workers 85 --res 80 --decay 1e-4
