#!/usr/bin/env bash
set -euo pipefail

ROOT_FALLBACK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DEFAULT="/n/holylfs06/LABS/mzitnik_lab/Users/yektefaie/PhylaFlow"
if [[ -d "$ROOT_DEFAULT" ]]; then
  ROOT="$ROOT_DEFAULT"
else
  ROOT="$ROOT_FALLBACK"
fi
SCRATCH_ROOT="/n/netscratch/mzitnik_lab/Lab/yektefaie/phylaflow"

declare -A CONFIGS=(
  [ds1]="$ROOT/configs/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptboth_6000_currentrecipe_20260425.yaml"
  [ds2]="$ROOT/configs/ds2_short_multipair42_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
  [ds3]="$ROOT/configs/ds3_short_multipair243_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
  [ds4]="$ROOT/configs/ds4_short_multipair573_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
  [ds5]="$ROOT/configs/ds5_short_multipair525_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
  [ds6]="$ROOT/configs/ds6_short_multipair219_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
  [ds7]="$ROOT/configs/ds7_short_multipair1344_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
  [ds8]="$ROOT/configs/ds8_short_multipair1122_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260425.yaml"
  [ds1-good]="$ROOT/configs/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptboth_6000_20260421.yaml"
  [ds5-20260424]="$ROOT/configs/ds5_short_multipair525_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260424.yaml"
)

print_usage() {
  cat <<'EOF'
Usage:
  ./launch_ds_local.sh list
  ./launch_ds_local.sh <target>
  ./launch_ds_local.sh <target> --print-only

Targets:
  ds1 ds2 ds3 ds4 ds5 ds6 ds7 ds8
  ds1-good
  ds5-20260424
EOF
}

print_targets() {
  local key
  for key in ds1 ds2 ds3 ds4 ds5 ds6 ds7 ds8 ds1-good ds5-20260424; do
    printf '%-12s %s\n' "$key" "${CONFIGS[$key]}"
  done
}

if [[ $# -eq 0 ]]; then
  print_usage
  exit 1
fi

target="${1,,}"
case "$target" in
  list|--list|-l)
    print_targets
    exit 0
    ;;
  help|--help|-h)
    print_usage
    exit 0
    ;;
esac

config="${CONFIGS[$target]:-}"
if [[ -z "$config" ]]; then
  echo "Unknown target: $1" >&2
  print_usage >&2
  exit 1
fi

if [[ ! -f "$config" ]]; then
  echo "Config not found: $config" >&2
  exit 1
fi

if [[ "${2:-}" == "--print-only" ]]; then
  printf 'python -m run.run %q\n' "$config"
  exit 0
fi

echo "Launching $target"
echo "Config: $config"
echo "Scratch outputs: $SCRATCH_ROOT"

mkdir -p \
  "$SCRATCH_ROOT/checkpoints/full_sanity_fixedpair_20260401" \
  "$SCRATCH_ROOT/metrics/full_sanity_fixedpair_20260401" \
  "$SCRATCH_ROOT/wandb"
export WANDB_DIR="$SCRATCH_ROOT/wandb"

if command -v module >/dev/null 2>&1; then
  module load cuda/12.4
fi

exec python -m run.run "$config"
