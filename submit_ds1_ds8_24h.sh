#!/usr/bin/env bash
set -euo pipefail

ROOT_FALLBACK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DEFAULT="/n/holylfs06/LABS/mzitnik_lab/Users/yektefaie/PhylaFlow"
if [[ -d "$ROOT_DEFAULT" ]]; then
  ROOT="$ROOT_DEFAULT"
else
  ROOT="$ROOT_FALLBACK"
fi

SUBMITTER="$ROOT/submit_ds_24h.sh"
TARGETS=(ds1 ds2 ds3 ds4 ds5 ds6 ds7 ds8)

if [[ ! -x "$SUBMITTER" ]]; then
  echo "Submitter not executable: $SUBMITTER" >&2
  exit 1
fi

for target in "${TARGETS[@]}"; do
  "$SUBMITTER" "$target" "$@"
done
