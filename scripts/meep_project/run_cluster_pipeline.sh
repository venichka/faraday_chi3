#!/usr/bin/env bash
set -euo pipefail

print_help() {
  cat <<'EOF'
Usage:
  ./run_cluster_pipeline.sh [debug|release|custom] [options]
  ./run_cluster_pipeline.sh --help

Purpose:
  Thin wrapper around cluster_pipeline_slurm.py with:
  1) profile selection (debug/release/custom),
  2) micromamba activation (meep-mpi),
  3) convenient stage shortcuts,
  4) full pass-through of advanced knobs.

Profiles:
  debug    -> 2 nodes, 20 tasks/node, 1 cpu/task (40 CPUs total)
  release  -> 5 nodes, 20 tasks/node, 1 cpu/task (100 CPUs total)
  custom   -> no profile defaults; set Slurm resources explicitly

Wrapper-specific options:
  -h, --help                 Show this help
  --profile <p>              Override profile (debug|release|custom)
  --submit / --no-submit     Submit to Slurm (default: --submit)
  --env-name <name>          Micromamba environment name (default: meep-mpi)
  --skip-activate            Do not run micromamba activation in wrapper
  --show-default-parameters  Show resolved defaults from python launcher and exit
  --only-opt                 Equivalent to: --stages opt
  --only-sim                 Equivalent to: --stages sim
  --only-sweep               Equivalent to: --stages sweep
  --only-opt-sim             Equivalent to: --stages opt,sim
  --only-sim-sweep           Equivalent to: --stages sim,sweep

Common Slurm knobs (passed through):
  --nodes, --ntasks-per-node, --cpus-per-task, --mem, --time-limit
  --partition, --nodelist, --exclude-nodes, --account, --qos, --constraint
  --job-name, --slurm-output, --slurm-error, --sbatch-extra

Common optimizer knobs (passed through):
  --optimizers new|mf|new,mf
  --optimizer-workers        (default: total allocated CPUs)
  --parallel-optimizers / --no-parallel-optimizers
  --objective-mode quick|full
  --objective-resolution
  --objective-decay-threshold
  --optimizer-pump-intensity
  --bayes-init, --bayes-iters, --bayes-batch-size, --bayes-candidates, --bayes-xi
  --mf-probe-epsilon, --mf-stage1-per-n, --mf-stage2-topk, --mf-stage3-topk
  --optimizer-debug

Common simulation knobs (passed through):
  --sim-dims 1|3|1,3
  --sim-fidelity low|high|both
  --sim-cutoff
  --sim-1d-res-low, --sim-1d-res-high
  --sim-3d-res-low, --sim-3d-res-high
  --sim-3d-mpi-ranks         (default: total CPUs / 2)

Common sweep knobs (passed through):
  --sweep-dims 1|3|1,3
  --sweep-fidelity low|high|both
  --sweep-cutoff
  --sweep-i-min, --sweep-i-max, --sweep-range-scale
  --sweep-points             (global override for both dims)
  --sweep-1d-points          (default: total CPUs)
  --sweep-3d-points          (default: total CPUs / 2)
  --sweep-1d-workers         (default: total CPUs)
  --sweep-3d-workers
  --sweep-1d-res-low, --sweep-1d-res-high
  --sweep-3d-res-low, --sweep-3d-res-high
  --sweep-3d-mpi-ranks       (default: total CPUs / 2)
  --parallel-sweep-dims

Stage control:
  --stages all|opt|sim|sweep|opt,sim|sim,sweep|opt,sweep|opt,sim,sweep
  --source-run-root /path/to/previous_run   (for sim/sweep-only reuse)
  --skip-optimizers, --skip-sims, --skip-sweeps

Examples:
  1) Smoke test full pipeline on debug profile:
     ./run_cluster_pipeline.sh debug --preset smoke

  2) Run only optimizers:
     ./run_cluster_pipeline.sh release --only-opt --optimizers new,mf --preset full

  3) Reuse previous optimizer outputs and run only sim+sweep:
     ./run_cluster_pipeline.sh release --only-sim-sweep \
       --source-run-root /data/prev/pipeline_cluster_20260303_120000 --optimizers new,mf

  4) Pin to specific CPU nodes:
     ./run_cluster_pipeline.sh custom --submit --partition defq --nodelist cpu[001-004] \
       --nodes 1 --ntasks-per-node 20 --cpus-per-task 1

All additional arguments are forwarded directly to cluster_pipeline_slurm.py.
EOF
}

PROFILE="debug"
SUBMIT=1
ENV_NAME="meep-mpi"
SKIP_ACTIVATE=0
STAGE_SHORTCUT_ARGS=()
PASSTHROUGH_ARGS=()

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

if [[ "${1:-}" != "" && "${1:0:1}" != "-" ]]; then
  PROFILE="$1"
  shift
fi

case "${PROFILE}" in
  debug|release|custom) ;;
  *)
    echo "Invalid profile '${PROFILE}'. Use debug|release|custom."
    echo
    print_help
    exit 2
    ;;
esac

while (($#)); do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    --profile)
      if [[ $# -lt 2 ]]; then
        echo "--profile requires a value"
        exit 2
      fi
      PROFILE="$2"
      shift 2
      ;;
    --submit)
      SUBMIT=1
      shift
      ;;
    --no-submit)
      SUBMIT=0
      shift
      ;;
    --env-name)
      if [[ $# -lt 2 ]]; then
        echo "--env-name requires a value"
        exit 2
      fi
      ENV_NAME="$2"
      shift 2
      ;;
    --skip-activate)
      SKIP_ACTIVATE=1
      shift
      ;;
    --show-default-parameters)
      PASSTHROUGH_ARGS+=(--show-default-parameters)
      shift
      ;;
    --only-opt)
      STAGE_SHORTCUT_ARGS=(--stages opt)
      shift
      ;;
    --only-sim)
      STAGE_SHORTCUT_ARGS=(--stages sim)
      shift
      ;;
    --only-sweep)
      STAGE_SHORTCUT_ARGS=(--stages sweep)
      shift
      ;;
    --only-opt-sim)
      STAGE_SHORTCUT_ARGS=(--stages opt,sim)
      shift
      ;;
    --only-sim-sweep)
      STAGE_SHORTCUT_ARGS=(--stages sim,sweep)
      shift
      ;;
    --)
      shift
      while (($#)); do
        PASSTHROUGH_ARGS+=("$1")
        shift
      done
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${PROFILE}" in
  debug|release|custom) ;;
  *)
    echo "Invalid profile '${PROFILE}'. Use debug|release|custom."
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ${SKIP_ACTIVATE} -eq 0 ]]; then
  if ! command -v micromamba >/dev/null 2>&1; then
    echo "micromamba is not available in PATH. Use --skip-activate or load micromamba first."
    exit 127
  fi
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate "${ENV_NAME}"
fi

CMD=(python "${SCRIPT_DIR}/cluster_pipeline_slurm.py" --cluster-profile "${PROFILE}")
if [[ ${SUBMIT} -eq 1 ]]; then
  CMD+=(--submit)
fi
CMD+=(--python-exe "$(command -v python)")
if [[ ${#STAGE_SHORTCUT_ARGS[@]} -gt 0 ]]; then
  CMD+=("${STAGE_SHORTCUT_ARGS[@]}")
fi
if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
  CMD+=("${PASSTHROUGH_ARGS[@]}")
fi

echo "[cmd] ${CMD[*]}"
"${CMD[@]}"
