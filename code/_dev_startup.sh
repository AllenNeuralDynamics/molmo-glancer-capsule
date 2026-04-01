#!/usr/bin/env bash
# _dev_startup2.sh — Install v2 project dependencies
#
# Run once at capsule startup (after /code is mounted).
# The Docker image provides: Python 3.12, PyTorch+CUDA, boto3, ai2-olmo-core.
# This script installs everything else needed for the v2 pipeline.
#
# v2 stack:
#   - Molmo2-O-7B via HuggingFace transformers (VQA + orchestration, single model)
#   - NeuroglancerState (neuroglancer-chat, URL builder only)
#   - Playwright (headless screenshots)
#
# Usage:
#   bash /code/_dev_startup2.sh

set -euo pipefail

PIP=/opt/conda/bin/pip

# Redirect pip tmp/build dirs to /scratch to avoid filling the 5 GB root overlay.
mkdir -p /scratch/pip-tmp
export TMPDIR=/scratch/pip-tmp

echo "--- pip upgrade ---"
"$PIP" install -q --no-cache-dir --upgrade pip

# ---------------------------------------------------------------------------
# Molmo2-O-7B inference stack
# transformers>=4.57.1 required for Molmo2 model code
# bitsandbytes: 8-bit quantization (~8 GB on T4 vs ~16 GB fp32)
# decord2: video frame decoding (required by Molmo2 even for image-only use)
# ---------------------------------------------------------------------------
echo ""
echo "--- Molmo2 inference stack ---"
"$PIP" install -q --no-cache-dir \
    "transformers>=4.57.1,<5.0.0" \
    bitsandbytes \
    accelerate \
    einops \
    "molmo-utils>=0.0.1" \
    decord2 \
    pillow

# ---------------------------------------------------------------------------
# Playwright + Chromium browser (~130 MB) — headless screenshot capture
# Redirected to /scratch to avoid filling root overlay
# ---------------------------------------------------------------------------
echo ""
echo "--- Playwright ---"
"$PIP" install -q --no-cache-dir "playwright==1.44"

echo ""
echo "--- Playwright Chromium browser ---"
export PLAYWRIGHT_BROWSERS_PATH=/scratch/ms-playwright
/opt/conda/bin/playwright install --with-deps chromium

# ---------------------------------------------------------------------------
# torchvision — upgrade to match torch version in image
# Dockerfile upgrades torch but leaves torchvision at 0.19.0 (for torch 2.4),
# which breaks AutoProcessor. Upgrade to matching +cu130 build.
# ---------------------------------------------------------------------------
echo ""
echo "--- torchvision (upgrade to match torch) ---"
"$PIP" install -q --no-cache-dir torchvision --upgrade \
    --extra-index-url https://download.pytorch.org/whl/cu130

# ---------------------------------------------------------------------------
# Data / HTTP utilities
# httpx: fetch NG state URLs during testing
# zarr + s3fs: read zarr volumes from S3 if needed
# ---------------------------------------------------------------------------
echo ""
echo "--- utility deps ---"
"$PIP" install -q --no-cache-dir \
    httpx \
    zarr \
    s3fs \
    scikit-image

# ---------------------------------------------------------------------------
# neuroglancer-chat — editable install, no deps
# We only import NeuroglancerState (pure stdlib internally).
# All heavy deps (FastAPI, Panel, OpenAI, Polars) are intentionally excluded.
# ---------------------------------------------------------------------------
echo ""
echo "--- neuroglancer-chat (editable, --no-deps) ---"
"$PIP" install -q --no-cache-dir --no-deps -e /code/lib/neuroglancer-chat

# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Setup complete."
echo ""
echo " Next: download model weights:"
echo "   export HF_TOKEN=hf_..."
echo "   bash /code/_download_weights_v2.sh"
echo "=========================================="
