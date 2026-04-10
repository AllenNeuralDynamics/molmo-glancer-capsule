#!/usr/bin/env bash
# _download_weights.sh — Download model weights to /scratch
#
# Two models:
#   1. Molmo2-O-7B: vision model (SigLIP2 + OLMo3-7B backbone), ~14-16 GB on disk, ~14.5 GB fp16
#   2. OLMo 3.1 32B Think: text reasoning model, ~64 GB on disk, ~34 GB INT8 at runtime
#
# HF_TOKEN is required to avoid anonymous rate limits.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash /code/_download_weights.sh

set -euo pipefail

# Redirect HF cache to /scratch — keeps root overlay from filling up
export HF_HOME=/scratch/hf-cache
mkdir -p /scratch/checkpoints /scratch/hf-cache

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "  export HF_TOKEN=hf_...  then re-run."
    exit 1
fi

# ---------------------------------------------------------------------------
# Molmo2-O-7B (~14-16 GB on disk)
# Vision + image interpretation model (fp16, device_map=auto)
# ---------------------------------------------------------------------------
MOLMO2_DEST=/scratch/checkpoints/Molmo2-O-7B

echo "--- Molmo2-O-7B (~14-16 GB, resumes if partial) ---"
huggingface-cli download allenai/Molmo2-O-7B \
    --local-dir "$MOLMO2_DEST"
echo "  Done: $MOLMO2_DEST"

# ---------------------------------------------------------------------------
# OLMo 3.1 32B Think (~64 GB on disk)
# Text reasoning model (loaded as INT8 via bitsandbytes, ~34 GB VRAM)
# ---------------------------------------------------------------------------
OLMO_DEST=/scratch/checkpoints/Olmo-3.1-32B-Think

echo ""
echo "--- OLMo 3.1 32B Think (~64 GB, resumes if partial) ---"
huggingface-cli download allenai/OLMo-3.1-32B-Think \
    --local-dir "$OLMO_DEST"
echo "  Done: $OLMO_DEST"

# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Weights ready."
echo ""
echo " Molmo2-O-7B: $MOLMO2_DEST"
echo " OLMo 3.1 32B Think: $OLMO_DEST"
echo "=========================================="
