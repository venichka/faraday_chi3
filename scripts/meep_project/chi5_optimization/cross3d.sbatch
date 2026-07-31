#!/bin/bash
#SBATCH -J chi5_cross3d
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH -t 2-00:00:00
#SBATCH -o chi5_optimization/cross3d/slurm-3d-%j.out
#SBATCH -e chi5_optimization/cross3d/slurm-3d-%j.err

# Stage C heavy 3D run of cand07 at center 0.69013 / Delta 0.006 (res 30, courant 0.25, I=1e12).
# Submit with:  sbatch -w node002 chi5_optimization/cross3d.sbatch
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project

# Prep inputs once (serial; json only) BEFORE the MPI launch.
python chi5_optimization/cross3d.py --prep

mpirun -np 96 python faraday_meep_fp_circ.py --dim 3 --mode full \
  --materials fit --sin-fit si3n4.csv --sio2-fit sio2.csv --fit-poles 2 --fit-window 600 2000 \
  --high-index-material sin \
  --geometry-file     chi5_optimization/cross3d/cand07/geometry.json \
  --cavity-modes-file chi5_optimization/cross3d/cand07/cavity_modes.json \
  --resolution 30 --courant 0.25 --decay-threshold 1e-4 \
  --pump-intensity 1e12 \
  --pump1-frequency 0.693130000 --pump2-frequency 0.687130000 \
  --output-dir chi5_optimization/cross3d/cand07/run3d
