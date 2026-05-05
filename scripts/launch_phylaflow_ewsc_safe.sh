#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <config.yaml> [cuda_visible_devices]" >&2
  exit 2
fi

CONFIG_PATH="$1"
CUDA_DEVICES="${2:-${CUDA_VISIBLE_DEVICES:-}}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

WORKDIR="${PHYLA_WORKDIR:-/home/unix/yektefai/PhylaFlow}"
PYTHON_BIN="${PYTHON_BIN:-/ewsc/yektefai/envs/envs/pgt/bin/python}"
EWSC_ROOT="${PHYLA_EWSC_ROOT:-/ewsc/yektefai/phylaflow}"
LOG_ROOT="${PHYLA_LOG_ROOT:-$EWSC_ROOT/logs/full_sanity_fixedpair_20260401}"

RUN_NAME="$(
  awk '
    $1 == "wandb_name:" {
      sub(/^[[:space:]]*wandb_name:[[:space:]]*/, "")
      print
      exit
    }
  ' "$CONFIG_PATH"
)"

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(basename "$CONFIG_PATH" .yaml)"
fi

mkdir -p \
  "$EWSC_ROOT/tmp" \
  "$EWSC_ROOT/wandb" \
  "$EWSC_ROOT/wandb_cache" \
  "$EWSC_ROOT/wandb_data" \
  "$EWSC_ROOT/wandb_artifacts" \
  "$EWSC_ROOT/wandb_config" \
  "$EWSC_ROOT/cache" \
  "$EWSC_ROOT/torch_home" \
  "$EWSC_ROOT/torch_extensions" \
  "$EWSC_ROOT/triton/manual" \
  "$EWSC_ROOT/mpl_config" \
  "$LOG_ROOT"

export TMPDIR="$EWSC_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export WANDB_DIR="$EWSC_ROOT/wandb"
export WANDB_CACHE_DIR="$EWSC_ROOT/wandb_cache"
export WANDB_DATA_DIR="$EWSC_ROOT/wandb_data"
export WANDB_ARTIFACT_DIR="$EWSC_ROOT/wandb_artifacts"
export WANDB_CONFIG_DIR="$EWSC_ROOT/wandb_config"
export XDG_CACHE_HOME="$EWSC_ROOT/cache"
export TORCH_HOME="$EWSC_ROOT/torch_home"
export TORCH_EXTENSIONS_DIR="$EWSC_ROOT/torch_extensions"
export TRITON_CACHE_DIR="$EWSC_ROOT/triton/manual"
export MPLCONFIGDIR="$EWSC_ROOT/mpl_config"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

if [[ -n "$CUDA_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi

RUN_LOG="${RUN_LOG:-$LOG_ROOT/${RUN_NAME}.log}"

cd "$WORKDIR"

setsid "$PYTHON_BIN" -u -m run.run "$CONFIG_PATH" > "$RUN_LOG" 2>&1 < /dev/null &
PID="$!"

echo "pid=$PID"
echo "run_name=$RUN_NAME"
echo "config=$CONFIG_PATH"
echo "log=$RUN_LOG"
echo "tmpdir=$TMPDIR"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
fi
