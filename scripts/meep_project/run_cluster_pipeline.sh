#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_cluster_pipeline.sh debug [extra cluster_pipeline_slurm.py args...]
#   ./run_cluster_pipeline.sh release [extra cluster_pipeline_slurm.py args...]

PROFILE="${1:-debug}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${PROFILE}" in
  debug|release) ;;
  *)
    echo "Usage: $0 [debug|release] [extra args...]"
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$(micromamba shell hook --shell bash)"
micromamba activate meep-mpi

python "${SCRIPT_DIR}/cluster_pipeline_slurm.py" \
  --submit \
  --cluster-profile "${PROFILE}" \
  "$@"
