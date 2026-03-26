# Environment Setup Notes

## Strategy

- **Dockerfile + postInstall** → pip-install all stable, heavy packages (baked into image)
- **`/code/_dev_startup.sh`** → editable installs from `/code/lib/` (run at capsule startup)

Key constraint: `/code/` is mounted at **run time**, not at Docker build time, so editable installs cannot happen in the Dockerfile or postInstall.

---

## `environment/Dockerfile`

```dockerfile
# hash:sha256:5eb57a7fd0bb514f2a1d9250a00760d5a1e8b488551eca0b72b594c23f65b7fb
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

# VS Code server
ADD "https://github.com/coder/code-server/releases/download/v4.100.3/code-server-4.100.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.100.3-linux-amd64/bin/code-server /usr/bin/code-server \
    && mkdir -p /.vscode/extensions

# System packages: gcc (cloud-volume C extensions) + GL libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# neuroglancer-chat dependencies (from pyproject.toml)
RUN pip install -U --no-cache-dir \
    boto3 \
    cloud-volume \
    "fastapi>=0.116.1" \
    httpx \
    hvplot \
    numpy \
    "openai>=1.104.2" \
    pillow \
    polars \
    pyarrow \
    pydantic \
    python-dotenv \
    python-multipart \
    "uvicorn[standard]"

# Panel UI (neuroglancer-chat panel group)
RUN pip install -U --no-cache-dir \
    "panel>=1.7.5" \
    "panel-neuroglancer>=0.1.0"

# molmoweb dependencies
RUN pip install -U --no-cache-dir \
    "playwright==1.44" \
    "transformers>=4.51,<5" \
    accelerate \
    einops \
    jinja2 \
    fire \
    torchmetrics \
    backoff \
    "tenacity==9.1.2" \
    google-genai \
    molmo-utils \
    "ai2-molmo2 @ git+https://github.com/allenai/molmo2.git"

# Data pipeline tools
RUN pip install -U --no-cache-dir \
    zarr \
    s3fs \
    scikit-image

# OLMo / vLLM for local LLM serving
RUN pip install -U --no-cache-dir \
    ai2-olmo-core \
    vllm

COPY postInstall /
RUN /postInstall
```

---

## `environment/postInstall`

Runs during Docker build (`/code/` not yet mounted). Handles only image-level setup:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Install Playwright's Chromium browser + system dependencies into the image
# (avoids downloading ~150 MB on every capsule start)
playwright install --with-deps chromium
```

Make it executable: `chmod +x environment/postInstall`

---

## `/code/_dev_startup.sh`

Since Python 3.12 is now the base image, `uv` is no longer needed. Use `pip` directly:

```bash
#!/usr/bin/env bash
# dev_startup.sh — editable installs from /code/lib/ (runs at capsule start, /code is mounted)
set -euo pipefail

PIP=/opt/conda/bin/pip

echo "--- molmo-glancer (editable) ---"
"$PIP" install -q --no-cache-dir -e /code/lib/molmo-glancer

echo "--- neuroglancer-chat (editable) ---"
"$PIP" install -q --no-cache-dir -e /code/lib/neuroglancer-chat

echo "--- molmoweb (editable) ---"
"$PIP" install -q --no-cache-dir -e /code/lib/molmoweb --no-deps
# ^ --no-deps because heavy deps (transformers, playwright, ai2-molmo2) are already in the image

echo ""
echo "Setup complete. Activate conda env or use pip-installed packages directly."
echo "Next: download model weights to /scratch/checkpoints/"
```

---

## Key Notes

- **`vllm` + torch compatibility**: The base image has torch 2.4.0 with CUDA 12.4; vllm 0.9.x should be compatible. If there's a conflict at build time, pin it: `vllm==0.9.0`.
- **`ai2-molmo2` from git**: Including this in the Dockerfile bakes it in at build time (good for reproducibility), but adds ~2–3 min to image build. If it changes frequently, move it to `dev_startup.sh` instead.
- **`--no-deps` on molmoweb editable install**: Safe because all molmoweb deps are pre-installed in the image. If new deps are added to `molmoweb/pyproject.toml` later, add them to the Dockerfile too.
- **Playwright Chromium in postInstall**: Baking it into the image saves ~30s on each startup. Installs to `/root/.cache/ms-playwright/` inside the image layer.
