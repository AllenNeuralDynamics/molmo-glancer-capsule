#!/usr/bin/env bash
# _dev_startup.sh — Install all project dependencies for molmo-glancer dev environment
#
# Environment strategy:
#   - uv manages Python 3.12 independently of the capsule's system Python (3.8.5).
#   - A single shared venv at /code/.venv holds all directly-imported packages:
#       molmo-glancer (editable), neuroglancer-chat (editable), vllm, olmo-core, data tools.
#     Activate with: source /code/.venv/bin/activate
#   - neuroglancer-chat and molmoweb also get project-local uv venvs so they can
#     be launched with `uv run uvicorn` / `uv run python` as documented.
#
# Usage: bash /code/_dev_startup.sh

set -euo pipefail

CONDA_PIP=/opt/conda/bin/pip
PYTHON_VERSION=3.12
VENV=/scratch/.venv

# ---------------------------------------------------------------------------
# uv — install via conda pip (only thing conda Python is used for)
# ---------------------------------------------------------------------------
echo ""
echo "--- uv ---"
"$CONDA_PIP" install uv --quiet

UV=/opt/conda/bin/uv
export UV_CACHE_DIR=/scratch/.uv-cache
export UV_LINK_MODE=copy

# ---------------------------------------------------------------------------
# C compiler — required to build C extensions (e.g. posix-ipc via cloud-volume)
# ---------------------------------------------------------------------------
echo ""
echo "--- build tools (gcc) ---"
apt-get update -qq && apt-get install -y gcc --quiet

# ---------------------------------------------------------------------------
# Python 3.12 — downloaded and managed by uv, independent of system Python
# ---------------------------------------------------------------------------
echo ""
echo "--- Python $PYTHON_VERSION ---"
"$UV" python install "$PYTHON_VERSION"

# ---------------------------------------------------------------------------
# Shared venv at /code/.venv — used by task scripts, notebooks, conda shell
# ---------------------------------------------------------------------------
echo ""
echo "--- shared venv ($VENV) ---"
"$UV" venv "$VENV" --python "$PYTHON_VERSION" --clear

# ---------------------------------------------------------------------------
# molmo-glancer  (dev branch — editable)
# ---------------------------------------------------------------------------
echo ""
echo "--- molmo-glancer (editable) ---"
"$UV" pip install --python "$VENV" -e /code/lib/molmo-glancer

# ---------------------------------------------------------------------------
# neuroglancer-chat  (olmo-local branch — editable)
# panel deps installed separately (dependency-groups are not pip extras)
# ---------------------------------------------------------------------------
echo ""
echo "--- neuroglancer-chat (editable) ---"
"$UV" pip install --python "$VENV" -e /code/lib/neuroglancer-chat
"$UV" pip install --python "$VENV" "panel>=1.7.5" "panel-neuroglancer>=0.1.0"

# Also sync the project-local uv venv so `uv run uvicorn` works as documented
cd /code/lib/neuroglancer-chat
"$UV" sync --python "$PYTHON_VERSION"
"$UV" sync --python "$PYTHON_VERSION" --group panel

# ---------------------------------------------------------------------------
# OLMo-core (from PyPI — not editable, local clone is for reference only)
# ---------------------------------------------------------------------------
echo ""
echo "--- ai2-olmo-core (PyPI) ---"
"$UV" pip install --python "$VENV" ai2-olmo-core

# ---------------------------------------------------------------------------
# vllm — serves OLMo-3-7B-Instruct locally on the GPU (OpenAI-compatible API)
# neuroglancer-chat's OLMo adapter POSTs to it instead of OpenAI's servers
# ---------------------------------------------------------------------------
echo ""
echo "--- vllm ---"
"$UV" pip install --python "$VENV" vllm

# ---------------------------------------------------------------------------
# Data pipeline tools (used by molmo-glancer analysis + capsule scripts)
# ---------------------------------------------------------------------------
echo ""
echo "--- data tools ---"
"$UV" pip install --python "$VENV" zarr s3fs scikit-image

# ---------------------------------------------------------------------------
# molmoweb  (main branch — uv-managed service, installed from local clone)
# ---------------------------------------------------------------------------
echo ""
echo "--- molmoweb (uv) ---"
cd /code/lib/molmoweb
"$UV" sync --python "$PYTHON_VERSION"
"$UV" run playwright install --with-deps chromium

echo ""
echo "=========================================="
echo " Setup complete."
echo ""
echo " Activate the shared venv before running task scripts or notebooks:"
echo "   source /scratch/.venv/bin/activate"
echo ""
echo " Next steps:"
echo "   1. Download model weights (can run in a separate terminal now):"
echo "      cd /code/lib/molmoweb && bash scripts/download_weights.sh allenai/MolmoWeb-4B"
echo "      mv /code/lib/molmoweb/checkpoints/MolmoWeb-4B /scratch/checkpoints/MolmoWeb-4B"
echo "      huggingface-cli download allenai/OLMo-3-7B-Instruct \\"
echo "        --local-dir /scratch/checkpoints/OLMo-3-7B-Instruct"
echo "   2. See CLAUDE.md 'Startup sequence' for how to launch each service."
echo "=========================================="
