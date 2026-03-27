#!/usr/bin/env bash
# _dev_startup2.sh — Install project dependencies using pip (no uv)
#
# Run this once at capsule startup (after /code is mounted).
# The Docker image provides: Python 3.12.4, PyTorch+CUDA, boto3, ai2-olmo-core.
# This script installs everything else.
#
# Usage:
#   bash /code/_dev_startup2.sh
#   bash /code/_dev_startup2.sh --skip-vllm   # skip the heavy vllm install

set -euo pipefail

PIP=/opt/conda/bin/pip
SKIP_VLLM=false

# Redirect pip tmp/build dirs to /scratch to avoid filling the 5 GB root overlay.
# pip extracts wheels here before installing; vllm's wheel alone is ~500 MB.
mkdir -p /scratch/pip-tmp
export TMPDIR=/scratch/pip-tmp

echo "--- pip upgrade ---"
"$PIP" install -q --no-cache-dir --upgrade pip

for arg in "$@"; do
    case "$arg" in
        --skip-vllm) SKIP_VLLM=true ;;
    esac
done

# ---------------------------------------------------------------------------
# neuroglancer-chat runtime dependencies
# (from code/lib/neuroglancer-chat/pyproject.toml [project.dependencies])
# boto3 already in image; others likely not
# ---------------------------------------------------------------------------
echo ""
echo "--- neuroglancer-chat deps ---"
"$PIP" install -q --no-cache-dir \
    "cloud-volume>=12.5.0" \
    "fastapi>=0.116.1" \
    "httpx>=0.28.1" \
    "hvplot>=0.11.3" \
    "numpy>=2.2.6" \
    "pillow>=11.3.0" \
    "polars>=1.33.0" \
    "pyarrow>=21.0.0" \
    "pydantic>=2.11.7" \
    "python-dotenv>=1.1.1" \
    "python-multipart>=0.0.20" \
    "uvicorn[standard]>=0.35.0"

# Panel UI group (neuroglancer-chat panel dependency-group)
echo ""
echo "--- Panel UI deps ---"
"$PIP" install -q --no-cache-dir \
    "panel>=1.7.5" \
    "panel-neuroglancer>=0.1.0"

# ---------------------------------------------------------------------------
# molmoweb runtime dependencies
# (from code/lib/molmoweb/pyproject.toml [project.dependencies])
# torch/fastapi/jinja2/fire already covered above or in image
# ---------------------------------------------------------------------------
echo ""
echo "--- molmoweb deps ---"
"$PIP" install -q --no-cache-dir \
    "playwright==1.44" \
    "transformers>=4.51,<5" \
    accelerate \
    einops \
    "fire>=0.7.0" \
    "torchmetrics>=1.0.0" \
    "backoff>=2.0.0" \
    "tenacity==9.1.2" \
    "google-genai>=1.49.0" \
    "molmo-utils>=0.0.1" \
    "browserbase==1.4.0" \
    "python-fasthtml>=0.12.4" \
    "openai>=2.8.1"

echo ""
echo "--- ai2-molmo2 (from git — slow, ~2-3 min) ---"
"$PIP" install -q --no-cache-dir \
    "ai2-molmo2 @ git+https://github.com/allenai/molmo2.git"

# ---------------------------------------------------------------------------
# Playwright Chromium browser (~130 MB)
# Redirected to /scratch to avoid filling /root (~5 GB limit)
# (not in image because postInstall is empty — install here instead)
# ---------------------------------------------------------------------------
echo ""
echo "--- Playwright Chromium (browser + system deps) ---"
export PLAYWRIGHT_BROWSERS_PATH=/scratch/ms-playwright
/opt/conda/bin/playwright install --with-deps chromium

# ---------------------------------------------------------------------------
# torchvision — upgrade to match torch version
# The Dockerfile's `pip3 install -U ai2-olmo-core` upgrades torch (2.4→2.11)
# but leaves torchvision at 0.19.0 (built for 2.4), breaking AutoProcessor
# and the native MolmoWeb backend. Upgrade to the matching +cu130 build.
# ---------------------------------------------------------------------------
echo ""
echo "--- torchvision (upgrade to match torch) ---"
"$PIP" install -q --no-cache-dir torchvision --upgrade \
    --extra-index-url https://download.pytorch.org/whl/cu130

# ---------------------------------------------------------------------------
# Data pipeline tools (zarr, s3fs, scikit-image)
# ---------------------------------------------------------------------------
echo ""
echo "--- data pipeline tools ---"
"$PIP" install -q --no-cache-dir \
    zarr \
    s3fs \
    scikit-image

# ---------------------------------------------------------------------------
# vllm — serves OLMo-3-7B-Instruct locally (heavy, ~5 min, optional)
# Skip with: bash _dev_startup2.sh --skip-vllm
# ---------------------------------------------------------------------------
if [ "$SKIP_VLLM" = true ]; then
    echo ""
    echo "--- vllm skipped (--skip-vllm) ---"
else
    echo ""
    echo "--- vllm (heavy — installed to /scratch/vllm-venv to avoid filling root) ---"
    # vllm + deps are ~1-2 GB; root overlay is only 5 GB.
    # Use a system-site-packages venv in /scratch so torch/cuda from conda are reused.
    # To use vllm: source /scratch/vllm-venv/bin/activate
    if [ -d /scratch/vllm-venv ]; then
        echo "  /scratch/vllm-venv already exists — skipping (delete to reinstall)"
    else
        python3 -m venv /scratch/vllm-venv --system-site-packages
        TMPDIR=/scratch/pip-tmp /scratch/vllm-venv/bin/pip install -q --no-cache-dir vllm
        echo ""
        echo "  vllm installed at /scratch/vllm-venv"
        echo "  Usage: source /scratch/vllm-venv/bin/activate"
    fi
fi

# ---------------------------------------------------------------------------
# Editable installs from /code/lib/
# --no-deps: all deps already installed above; avoids redundant resolution
# ---------------------------------------------------------------------------
echo ""
echo "--- molmo-glancer (editable) ---"
"$PIP" install -q --no-cache-dir --no-deps -e /code/lib/molmo-glancer

echo ""
echo "--- neuroglancer-chat (editable) ---"
"$PIP" install -q --no-cache-dir --no-deps -e /code/lib/neuroglancer-chat

echo ""
echo "--- molmoweb (editable, local fork) ---"
"$PIP" install -q --no-cache-dir --no-deps -e /code/lib/molmoweb
# agent/ and utils/ have no __init__.py upstream; add them so the packages
# are discoverable both via sys.path and the editable install finder.
touch /code/lib/molmoweb/agent/__init__.py
touch /code/lib/molmoweb/utils/__init__.py

# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Setup complete."
echo ""
echo " Next steps:"
echo "   1. Download model weights:"
echo "      export HF_TOKEN=hf_..."
echo "      bash /code/_download_weights.sh"
echo "   2. See CLAUDE.md 'Startup sequence' for how to launch each service."
echo "=========================================="
