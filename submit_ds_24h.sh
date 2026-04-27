#!/usr/bin/env bash
set -euo pipefail

ROOT_FALLBACK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DEFAULT="/n/holylfs06/LABS/mzitnik_lab/Users/yektefaie/PhylaFlow"
if [[ -d "$ROOT_DEFAULT" ]]; then
  ROOT="$ROOT_DEFAULT"
else
  ROOT="$ROOT_FALLBACK"
fi

SCRATCH_ROOT="${SCRATCH_ROOT:-/n/netscratch/mzitnik_lab/Lab/yektefaie/phylaflow}"
LOG_ROOT="${LOG_ROOT:-$SCRATCH_ROOT/slurm_logs/ds_24h}"
BATCH_SCRIPT="$ROOT/slurm/run_ds_24h.sbatch"
PARTITION="${PARTITION:-kempner_h100}"
ACCOUNT="${ACCOUNT:-kempner_mzitnik_lab}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pgt}"

case "$PARTITION" in
  kempner_h100)
    GPU_REQUEST_DEFAULT="gpu:nvidia_h100_80gb_hbm3:1"
    ;;
  *)
    GPU_REQUEST_DEFAULT="gpu:1"
    ;;
esac
GPU_REQUEST="${GPU_REQUEST:-$GPU_REQUEST_DEFAULT}"

usage() {
  cat <<'EOF'
Usage:
  ./submit_ds_24h.sh <target>
  ./submit_ds_24h.sh <target> --dry-run
  ./submit_ds_24h.sh <target> --test-only
  ./submit_ds_24h.sh list

Targets:
  ds1 ds2 ds3 ds4 ds5 ds6 ds7 ds8

Defaults:
  PARTITION=kempner_h100
  ACCOUNT=kempner_mzitnik_lab
  TIME_LIMIT=24:00:00
  CPUS_PER_TASK=4
  MEMORY=32G
  CONDA_ENV_NAME=pgt
EOF
}

print_targets() {
  printf '%s\n' ds1 ds2 ds3 ds4 ds5 ds6 ds7 ds8
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 1
fi

TARGET="${1,,}"
case "$TARGET" in
  list|--list|-l)
    print_targets
    exit 0
    ;;
  help|--help|-h)
    usage
    exit 0
    ;;
esac

if [[ ! -f "$BATCH_SCRIPT" ]]; then
  echo "Batch script not found: $BATCH_SCRIPT" >&2
  exit 1
fi

case "$TARGET" in
  ds1|ds2|ds3|ds4|ds5|ds6|ds7|ds8)
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    usage >&2
    exit 1
    ;;
esac

DRY_RUN=0
TEST_ONLY=0
for arg in "${@:2}"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    --test-only)
      TEST_ONLY=1
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_ROOT"

JOB_NAME="phylaflow-${TARGET}-24h"
OUT_LOG="$LOG_ROOT/${JOB_NAME}-%j.out"
ERR_LOG="$LOG_ROOT/${JOB_NAME}-%j.err"

SBATCH_CMD=(
  sbatch
  --job-name "$JOB_NAME"
  --partition "$PARTITION"
  --account "$ACCOUNT"
  --ntasks 1
  --cpus-per-task "$CPUS_PER_TASK"
  --mem "$MEMORY"
  --time "$TIME_LIMIT"
  --gres "$GPU_REQUEST"
  --output "$OUT_LOG"
  --error "$ERR_LOG"
  --export "ALL,SCRATCH_ROOT=$SCRATCH_ROOT,CONDA_ENV_NAME=$CONDA_ENV_NAME"
)

if [[ "$TEST_ONLY" -eq 1 ]]; then
  SBATCH_CMD+=(--test-only)
fi

SBATCH_CMD+=("$BATCH_SCRIPT" "$TARGET")

if [[ "$DRY_RUN" -eq 1 || "$TEST_ONLY" -eq 1 ]]; then
  printf '%q ' "${SBATCH_CMD[@]}"
  printf '\n'
  if [[ "$TEST_ONLY" -eq 1 ]]; then
    "${SBATCH_CMD[@]}"
  fi
  exit 0
fi

"${SBATCH_CMD[@]}"
