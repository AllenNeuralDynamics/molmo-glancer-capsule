#!/usr/bin/env bash
# _download_weights.sh — Download model weights to /scratch
#
# Downloads MolmoWeb-4B and OLMo-3-7B-Instruct directly to /scratch/checkpoints/.
# HF cache is also redirected to /scratch to avoid filling the 5 GB root overlay.
#
# Usage:
#   bash /code/_download_weights.sh
#   bash /code/_download_weights.sh --skip-olmo    # skip OLMo (14 GB)
#   bash /code/_download_weights.sh --skip-molmo   # skip MolmoWeb-4B (8 GB)

set -euo pipefail

SKIP_MOLMO=false
SKIP_OLMO=false

for arg in "$@"; do
    case "$arg" in
        --skip-molmo) SKIP_MOLMO=true ;;
        --skip-olmo)  SKIP_OLMO=true ;;
    esac
done

# Redirect HF hub cache to /scratch — prevents ~/.cache/huggingface/ filling root
export HF_HOME=/scratch/hf-cache
mkdir -p /scratch/checkpoints /scratch/hf-cache

# HF_TOKEN must be set to avoid anonymous rate limits (429 errors).
# Get a token at: huggingface.co → Settings → Access Tokens (read access)
# Usage: export HF_TOKEN=hf_... && bash /code/_download_weights.sh
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Anonymous downloads are rate-limited."
    echo "  export HF_TOKEN=hf_...  then re-run."
    exit 1
fi

# ---------------------------------------------------------------------------
# MolmoWeb-4B (~8 GB) — visual agent, fits on T4 (15 GB VRAM)
# Do NOT download MolmoWeb-8B — it exceeds T4 VRAM
# ---------------------------------------------------------------------------
MOLMO_DEST=/scratch/checkpoints/MolmoWeb-4B

if [ "$SKIP_MOLMO" = true ]; then
    echo "--- MolmoWeb-4B skipped (--skip-molmo) ---"
else
    echo "--- MolmoWeb-4B (~8 GB, resumes if partial) ---"
    huggingface-cli download allenai/MolmoWeb-4B \
        --local-dir "$MOLMO_DEST"
    echo "  Done: $MOLMO_DEST"
fi

# ---------------------------------------------------------------------------
# OLMo-3-7B-Instruct (~14 GB) — text LLM for tool calling
# ---------------------------------------------------------------------------
OLMO_DEST=/scratch/checkpoints/OLMo-3-7B-Instruct

if [ "$SKIP_OLMO" = true ]; then
    echo "--- OLMo-3-7B-Instruct skipped (--skip-olmo) ---"
else
    echo "--- OLMo-3-7B-Instruct (~14 GB, resumes if partial) ---"
    huggingface-cli download allenai/OLMo-3-7B-Instruct \
        --local-dir "$OLMO_DEST"
    echo "  Done: $OLMO_DEST"
fi

# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Weights ready."
echo ""
echo " MolmoWeb-4B:         $MOLMO_DEST"
echo " OLMo-3-7B-Instruct:  $OLMO_DEST"
echo ""
echo " Next: see CLAUDE.md 'Startup sequence' to launch services."
echo "=========================================="
