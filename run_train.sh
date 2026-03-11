#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-configs/sanity_train.yaml}"

if command -v module >/dev/null 2>&1; then
    module load cuda/12.4
fi

python -m run.run "${CONFIG_PATH}"
