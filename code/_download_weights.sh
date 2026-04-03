#!/usr/bin/env bash
# _download_weights_v2.sh — Download Molmo2-O-7B model weights to /scratch
#
# Molmo2-O-7B: single model serving both visual interpretation and orchestration.
# Backbone: OLMo3-7B-Instruct + SigLIP2 vision encoder (~14-16 GB on disk).
# Runtime: ~8 GB in 8-bit quantization — fits on T4 (15 GB VRAM).
#
# HF_TOKEN is required to avoid anonymous rate limits.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash /code/_download_weights_v2.sh

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
# Molmo2-O-7B (~14-16 GB on disk, ~8 GB loaded in 8-bit)
# Model ID: allenai/Molmo2-O-7B
# Loaded at runtime via: AutoModelForImageTextToText with load_in_8bit=True
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
echo "       load_in_8bit=True,"
echo "       device_map='auto',"
echo "   )"
echo "=========================================="
