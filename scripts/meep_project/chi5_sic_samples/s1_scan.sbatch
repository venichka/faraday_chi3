#!/bin/bash
#SBATCH --job-name=sic_s1
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=chi5_sic_samples/runs/s1_scan/slurm-s1-%A_%a.out
#SBATCH --error=chi5_sic_samples/runs/s1_scan/slurm-s1-%A_%a.err

# Stage 1: operating-point map of the two fabricated SiC-cavity samples (2860 x 1D FDTD).
# Scans the full 600-1000 nm probe band; the lab's "now" ({~800} u [850,950]) and "future"
# ([600,900]) probe windows are applied as filters at analysis time, so one set of sims answers
# both questions and also exposes anything good that falls outside both.
#
# 90 cpus-per-task targets the 96-core node00x machines. Workers are set to ~0.65 x cores, which
# beat 0.9 in past sweeps -- these runs are memory-bandwidth bound, not CPU bound.
# SAMPLE and PHASE are set by --export (SAMPLE = L3p2|L4p8|both, PHASE = A|B).
#   A  every probe mode x 3 centre offsets x 3 Delta   (868 sims)
#   B  the top 2 probe modes per sample, fine grid     (672 sims)
# ⚠️ --export splits on commas, so never put a comma inside one of these values.
: "${SAMPLE:=both}"
: "${PHASE:=A}"
# Reference pump intensity, chosen by s0_intensity.py: the highest intensity at which BOTH
# cavities are still clean chi5 -- local log-log slope 2.02 / 1.92 and DoLP 0.9993 / 0.9979.
# At 1e12 (the SiN campaign's reference) L4p8 is at DoLP 0.81 and slope < 1.5, i.e. large-signal.
: "${INTENSITY:=1e11}"

set -euo pipefail
export MAMBA_ROOT_PREFIX=/data/phi/nikita/envs/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi

cd /Users/phi/nikita/code/faraday_chi3/scripts/meep_project
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

echo "SAMPLE=${SAMPLE} PHASE=${PHASE} I=${INTENSITY} slice=${SLURM_ARRAY_TASK_ID}/4 host=$(hostname)"
python chi5_sic_samples/s1_scan.py --sample "${SAMPLE}" --phase "${PHASE}" --workers 45 \
    --intensity "${INTENSITY}" --slice "${SLURM_ARRAY_TASK_ID}/4"
