#!/usr/bin/env bash
# _download_weights.sh — Download Molmo2-O-7B model weights to /scratch
#
# Molmo2-O-7B: single model serving both visual interpretation and orchestration.
# Backbone: OLMo3-7B-Instruct + SigLIP2 vision encoder (~14-16 GB on disk).
# Runtime: ~3.6 GB in 4-bit (T4) or ~14.5 GB in fp16 (L40S).
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
# Model ID: allenai/Molmo2-O-7B
# Loaded at runtime via: AutoModelForImageTextToText (4-bit or fp16, auto-detected)
# ---------------------------------------------------------------------------
MOLMO2_DEST=/scratch/checkpoints/Molmo2-O-7B

echo "--- Molmo2-O-7B (~14-16 GB, resumes if partial) ---"
huggingface-cli download allenai/Molmo2-O-7B \
    --local-dir "$MOLMO2_DEST"
echo "  Done: $MOLMO2_DEST"

# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Weights ready."
echo ""
echo " Molmo2-O-7B: $MOLMO2_DEST"
echo ""
echo " Load in Python:"
echo "   from transformers import AutoProcessor, AutoModelForImageTextToText"
echo "   model = AutoModelForImageTextToText.from_pretrained("
echo "       '$MOLMO2_DEST',"
echo "       trust_remote_code=True,"
echo "       device_map='auto',"
echo "   )"
echo "=========================================="
