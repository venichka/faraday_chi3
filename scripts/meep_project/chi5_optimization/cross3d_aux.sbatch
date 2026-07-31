#!/bin/bash
#SBATCH -J chi5_cross3d_aux
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o chi5_optimization/cross3d/slurm-aux-%j.out
#SBATCH -e chi5_optimization/cross3d/slurm-aux-%j.err

# Stage C aux: 1D res-30 companion (matched-resolution baseline for the 3D/1D ratio) + TCMT
# extraction (FaradayJL params). Both are 1D/serial -> cheap; runs in parallel with the 3D job.
# Submit with:  sbatch chi5_optimization/cross3d_aux.sbatch
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
python chi5_optimization/cross3d.py --run-aux --res1d 30 --decay 1e-4
