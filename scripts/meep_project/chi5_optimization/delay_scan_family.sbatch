#!/bin/bash
#SBATCH -J dscan_fam
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o chi5_optimization/delay_scan/slurm-fam-%j.out
#SBATCH -e chi5_optimization/delay_scan/slurm-fam-%j.err

# One stage-2 systematics family (26 delay points). Families write to disjoint output dirs,
# so all seven run concurrently on separate nodes. Family tag comes in as $FAMILY.
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
python chi5_optimization/delay_scan.py --stage2 --families "$FAMILY" --workers 26 --res 80 --decay 1e-4
