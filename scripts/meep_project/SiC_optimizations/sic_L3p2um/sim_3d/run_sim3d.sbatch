#!/bin/bash
#SBATCH -J sic3d_L3p2
#SBATCH -p defq
#SBATCH -w node005
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH -o SiC_optimizations/sic_L3p2um/sim_3d/slurm-%j.out
#SBATCH -e SiC_optimizations/sic_L3p2um/sim_3d/slurm-%j.err

# SiC L=3.2um, 3D Faraday rotation. Same material/geometry/modes as the 1D runs.
# Courant 0.25 keeps the dispersive SiC/SiO2 fit (pole f0~23 /um) FDTD-stable at res 30.
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project

mpirun -np 96 python faraday_meep_fp_circ.py \
  --dim 3 --mode full --materials fit \
  --sin-fit sic.csv --sio2-fit sio2.csv --fit-poles 3 --fit-window 600 2000 \
  --high-index-material sic --courant 0.25 \
  --geometry-file SiC_optimizations/sic_L3p2um/geometry.json \
  --cavity-modes-file SiC_optimizations/sic_L3p2um/cavity_modes.json \
  --resolution 30 --decay-threshold 1e-4 --pump-intensity 1e12 \
  --output-dir SiC_optimizations/sic_L3p2um/sim_3d
