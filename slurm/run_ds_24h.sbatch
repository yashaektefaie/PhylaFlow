#!/usr/bin/env bash
set -euo pipefail

ROOT_FALLBACK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DEFAULT="/n/holylfs06/LABS/mzitnik_lab/Users/yektefaie/PhylaFlow"
if [[ -d "$ROOT_DEFAULT" ]]; then
  ROOT="$ROOT_DEFAULT"
else
  ROOT="$ROOT_FALLBACK"
fi

SCRATCH_ROOT="${SCRATCH_ROOT:-/n/netscratch/mzitnik_lab/Lab/yektefaie/phylaflow}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pgt}"
CONDA_SH_DEFAULT="/n/home03/yektefaie/miniconda3/etc/profile.d/conda.sh"
TARGET="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  sbatch slurm/run_ds_24h.sbatch <target>
  sbatch slurm/run_ds_24h.sbatch </abs/path/to/config.yaml>

Targets:
  ds1 ds2 ds3 ds4 ds5 ds6 ds7 ds8
EOF
}

config_for_target() {
  case "$1" in
    ds1)
      echo "$ROOT/configs/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptboth_6000_currentrecipe_20260425.yaml"
      ;;
    ds2)
      echo "$ROOT/configs/ds2_short_multipair42_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
      ;;
    ds3)
      echo "$ROOT/configs/ds3_short_multipair243_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
      ;;
    ds4)
      echo "$ROOT/configs/ds4_short_multipair573_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
      ;;
    ds5)
      echo "$ROOT/configs/ds5_short_multipair525_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
      ;;
    ds6)
      echo "$ROOT/configs/ds6_short_multipair219_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
      ;;
    ds7)
      echo "$ROOT/configs/ds7_short_multipair1344_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
      ;;
    ds8)
      echo "$ROOT/configs/ds8_short_multipair1122_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ -z "$TARGET" ]]; then
  usage >&2
  exit 1
fi

if [[ -f "$TARGET" ]]; then
  CONFIG_PATH="$TARGET"
else
  if ! CONFIG_PATH="$(config_for_target "$TARGET")"; then
    echo "Unknown target or config path: $TARGET" >&2
    usage >&2
    exit 1
  fi
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p \
  "$SCRATCH_ROOT/checkpoints/full_sanity_fixedpair_20260401" \
  "$SCRATCH_ROOT/metrics/full_sanity_fixedpair_20260401" \
  "$SCRATCH_ROOT/wandb" \
  "$SCRATCH_ROOT/wandb_cache" \
  "$SCRATCH_ROOT/triton/${SLURM_JOB_ID:-manual}"

export WANDB_DIR="$SCRATCH_ROOT/wandb"
export WANDB_CACHE_DIR="$SCRATCH_ROOT/wandb_cache"
export TRITON_CACHE_DIR="$SCRATCH_ROOT/triton/${SLURM_JOB_ID:-manual}"
export PYTHONUNBUFFERED=1

if command -v module >/dev/null 2>&1; then
  module load cuda/12.4.1-fasrc01
fi

if [[ -f "$CONDA_SH_DEFAULT" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_SH_DEFAULT"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "Could not find conda.sh to activate $CONDA_ENV_NAME" >&2
  exit 1
fi

conda activate "$CONDA_ENV_NAME"

echo "Host: $(hostname)"
echo "Target: $TARGET"
echo "Config: $CONFIG_PATH"
echo "SLURM job: ${SLURM_JOB_ID:-none}"
echo "WANDB_DIR: $WANDB_DIR"
echo "TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
echo "Conda env: $CONDA_ENV_NAME"

srun --unbuffered python -m run.run "$CONFIG_PATH"
