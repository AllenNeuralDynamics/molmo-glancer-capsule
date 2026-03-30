#!/usr/bin/env bash
# run.sh — v2 pipeline entry point
#
# Usage:
#   bash /code/run.sh
#   bash /code/run.sh --some-future-flag

set -euo pipefail

export PLAYWRIGHT_BROWSERS_PATH=/scratch/ms-playwright
export HF_HOME=/scratch/hf-cache

python3 -u /code/run_capsule.py "$@" 2>&1 | tee /results/output.log

# ./cleanup.sh && clear && ./run.sh 