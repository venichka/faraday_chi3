#!/bin/bash
#SBATCH -J sic_sweep_res
#SBATCH -p defq
#SBATCH -w node004
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o SiC_optimizations/sic_L3p2um/sweep_1d_resonant/slurm-%j.out
#SBATCH -e SiC_optimizations/sic_L3p2um/sweep_1d_resonant/slurm-%j.err

# Resonant-pump 1D sweep: pumps at f1=0.535/f2=0.5116 (~1.87um stopband), balanced, to isolate the chi^(5)
# I^2 scaling in a clean perturbative range. 20 points = 20 workers.
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi
export MPLBACKEND=Agg OMP_NUM_THREADS=1
cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project

python pump_intensity_sweep.py \
  --dim 1 --mode full --materials fit \
  --sin-fit sic.csv --sio2-fit sio2.csv --fit-poles 3 --fit-window 600 2000 \
  --high-index-material sic \
  --geometry-file SiC_optimizations/sic_L3p2um/geometry.json \
  --cavity-modes-file SiC_optimizations/sic_L3p2um/cavity_modes_resonant.json \
  --resolution 100 --decay-threshold 1e-4 \
  --intensity-range 1e8 1e11 20 --range-scale log --workers 20 \
  --output-root SiC_optimizations/sic_L3p2um/sweep_1d_resonant
