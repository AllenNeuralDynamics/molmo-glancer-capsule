# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session onboarding — read these first

| File | What it contains |
|------|-----------------|
| `CLAUDE.md` | This file — workspace layout, hardware, commands, architecture |
| `MEMORY_project.md` | Full project context: repos, architecture decisions, key design choices |
| `MEMORY_user_profile.md` | Who the user is and how to work with them effectively |
| `MEMORY_feedback.md` | Working style preferences — pace, checkpoints, level of detail |
| `IMPLEMENTATION_PLAN.md` | Detailed build plan: module structure, phases, GPU scheduling |
| `SETUP_NOTES.md` | Known env issues and fixes already applied to the code |
| `LOCAL_STACK_PLAN.md` | Earlier system design doc — component diagram, port map, VRAM analysis |
| `OLMO_INTEGRATION.md` | OLMo-core investigation — why OLMo-3-7B-Instruct, format details |

## Workspace Layout

This git repo (`molmo-glancer-capsule`) lives at `/root/capsule/`. The working paths are:

| Path | Purpose |
|------|---------|
| `/root/capsule/code/` | Scripts and library repos (git-tracked) |
| `/root/capsule/code/run_capsule.py` | **Main pipeline** — the full 6-step grid-search run |
| `/root/capsule/code/run` | Shell entry point: `bash /code/run` invokes `run_capsule.py` |
| `/root/capsule/code/_dev_startup2.sh` | Dep install script (pip-based; run once at capsule startup) |
| `/root/capsule/code/_download_weights.sh` | Weight download script (requires `HF_TOKEN`) |
| `/root/capsule/code/lib/molmo-glancer/` | molmo-glancer library (this project — orchestration, CV, browser) |
| `/root/capsule/code/lib/molmoweb/` | Local MolmoWeb fork (modified: bfloat16 fix, `__init__.py` additions) |
| `/root/capsule/code/lib/neuroglancer-chat/` | neuroglancer-chat fork (branch `olmo-local`) |
| `/root/capsule/code/lib/OLMo-core/` | OLMo-core library |
| `/root/capsule/data/` | Immutable input data |
| `/root/capsule/scratch/` → `/scratch/` | Unlimited scratch — model weights, caches |
| `/root/capsule/results/` → `/results/` | Curated outputs to export |

**Hardware:** 16 CPU cores, 64 GB RAM, **Tesla T4 GPU (15 GB VRAM, CUDA 13.0)**

---

## Project Goal

**molmo-glancer** — autonomous Neuroglancer question answering using a fully local AI stack.

The capsule runs MolmoWeb as a visual web-agent to navigate Neuroglancer (a WebGL-based 3D viewer for volumetric neuroscience data). The current pipeline (`run_capsule.py`) performs a **z-axis grid search**: screenshots Neuroglancer at N z-positions, CV-ranks by sharpness, MolmoWeb describes the top images visually, and OLMo synthesizes a recommendation.

---

## Key Commands

### Setup (run once after mounting `/code`)

```bash
# Install all dependencies (pip-based; ~10 min; add --skip-vllm to skip heavy vllm install)
bash /code/_dev_startup2.sh

# Download model weights (~22 GB total; requires HF_TOKEN)
export HF_TOKEN=hf_...
bash /code/_download_weights.sh
# Options: --skip-olmo (skip 14 GB OLMo), --skip-molmo (skip 8 GB MolmoWeb-4B)
```

### Run the pipeline

```bash
# Full run (CodeOcean "Reproducible Run" button calls this)
bash /code/run

# Direct invocation
PLAYWRIGHT_BROWSERS_PATH=/scratch/ms-playwright python3 -u /code/run_capsule.py
```

### Tests

```bash
# neuroglancer-chat tests (run from the lib dir, uses its own .venv)
cd /root/capsule/code/lib/neuroglancer-chat
python -m pytest tests/ -v

# Single test
python -m pytest tests/test_state_mutators.py -v -s

# With coverage
python -m coverage run -m pytest && python -m coverage report

# molmo-glancer tests
cd /root/capsule/code/lib/molmo-glancer
python -m pytest tests/ -v
```

---

## Architecture

### run_capsule.py — 6-step pipeline

```
Step 1: Read zarr volume shape (s3fs + zarr) → generate N_STEPS z-position URLs
Step 2: Screenshot all URLs with Playwright (SimpleEnv, reuse browser)
Step 3: CV-rank screenshots by sharpness (Laplace variance), contrast, entropy
Step 4: Start MolmoWeb server (FastAPI, port 8001, loads MolmoWeb-4B in bfloat16)
Step 5: Send top-N screenshots to MolmoWeb → get visual descriptions
Step 6: Stop MolmoWeb → start OLMo server (vllm, port 8002) → synthesize recommendation
```

Outputs to `/results/grid_search/`: `rank01_z*.png` … `rank10_z*.png`, `molmo_interpretations.json`, `olmo_recommendation.txt`, `results.json`.

### GPU memory scheduling

Models run **sequentially** on the T4 (15 GB VRAM) — never simultaneously:
- MolmoWeb-4B: ~8.9 GB bfloat16 (float32 = 17.8 GB → OOM; fix applied in fork)
- OLMo-3-7B-Instruct: ~14.4 GB float16 via vllm

vllm is installed to `/scratch/vllm-venv` (not conda) and invoked as `/scratch/vllm-venv/bin/python`. Flags required on T4: `--enforce-eager` (skip torch.compile autotuning), `--max-model-len 512`.

### MolmoWeb architecture

```
[Browser (Playwright/Chromium)]
        ↓ screenshot (base64 numpy array)
[FastApiActionPredictor]  →  POST /predict  →  [FastAPI server (port 8001)]
                                                        ↓
                                            [NativeActionPredictor]
                                            (ai2-molmo2, cuda:0, bfloat16)
```

Model output: JSON `{"thought": "...", "action": {"name": "click", "x": 45.2, "y": 12.8}, "action_description": "..."}`. Coordinates are **percentages (0–100)**, converted to pixels by the agent using viewport size (default 1280×720).

Key files in `code/lib/molmoweb/`:
- [agent/fastapi_model_server.py](code/lib/molmoweb/agent/fastapi_model_server.py) — FastAPI `/predict` endpoint
- [agent/model_backends.py](code/lib/molmoweb/agent/model_backends.py) — `NativeActionPredictor` (bfloat16 fix here)
- [agent/actions.py](code/lib/molmoweb/agent/actions.py) — action parsing & Playwright execution
- [inference/client.py](code/lib/molmoweb/inference/client.py) — high-level `MolmoWeb` client

**Note:** `HFActionPredictor` does not work with MolmoWeb-4B (checkpoint has no `processor_config.json`). Use `PREDICTOR_TYPE=native` only.

### neuroglancer-chat

Takes a different approach to Neuroglancer control — manipulates the **JSON state directly** via URL encoding instead of visual understanding.

```
User message → FastAPI backend (port 8000) → OpenAI function-calling loop
    ↓ tool calls → NeuroglancerState (parse/mutate/serialize NG URL hash)
    ↓ new URL → Panel UI with embedded Neuroglancer viewer (port 8006)
```

Requires `OPENAI_API_KEY`. Key files in `code/lib/neuroglancer-chat/src/neuroglancer_chat/`:
- [backend/tools/neuroglancer_state.py](code/lib/neuroglancer-chat/src/neuroglancer_chat/backend/tools/neuroglancer_state.py) — `NeuroglancerState` (URL parse/mutate/serialize)
- [backend/main.py](code/lib/neuroglancer-chat/src/neuroglancer_chat/backend/main.py) — FastAPI app + agent loop
- [panel/panel_app.py](code/lib/neuroglancer-chat/src/neuroglancer_chat/panel/panel_app.py) — Panel UI (~1500 lines)
- [backend/tools/io.py](code/lib/neuroglancer-chat/src/neuroglancer_chat/backend/tools/io.py) — CSV ingest + data tools

**Critical serialization rule:** Do NOT use `sort_keys=True` on NG state JSON — NG depends on dimension order (x,y,z,t).

---

## Model Choices (fixed for T4)

| Role | Model | VRAM | Notes |
|---|---|---|---|
| Visual agent | `allenai/MolmoWeb-4B` | ~8.9 GB bfloat16 | **4B only** — 8B exceeds T4 (15 GB) |
| Text LLM | `allenai/OLMo-3-7B-Instruct` | ~14.4 GB | Via vllm at `/scratch/vllm-venv` |

---

## Manual Service Startup (interactive dev)

```bash
# neuroglancer-chat backend (port 8000)
cd /root/capsule/code/lib/neuroglancer-chat/src/neuroglancer_chat
export OPENAI_API_KEY="sk-..."
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# neuroglancer-chat Panel UI (port 8006)
export BACKEND="http://127.0.0.1:8000"
python -m panel serve panel/panel_app.py --port 8006 --address 127.0.0.1 \
  --allow-websocket-origin=127.0.0.1:8006

# MolmoWeb model server (port 8001)
CKPT=/scratch/checkpoints/MolmoWeb-4B \
PREDICTOR_TYPE=native \
PYTHONPATH=/root/capsule/code/lib/molmoweb \
/opt/conda/bin/uvicorn agent.fastapi_model_server:app --host 0.0.0.0 --port 8001

# OLMo vllm server (port 8002) — kill MolmoWeb first
/scratch/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /scratch/checkpoints/OLMo-3-7B-Instruct \
  --served-model-name OLMo-3-7B-Instruct \
  --port 8002 --enforce-eager --max-model-len 512 \
  --gpu-memory-utilization 0.99 \
  VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((32*1024*1024))
```

---

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | (required for download) | HuggingFace access token |
| `PLAYWRIGHT_BROWSERS_PATH` | `/scratch/ms-playwright` | Chromium install location |
| `HF_HOME` | `/scratch/huggingface` | HF cache (keep off root overlay) |
| `OPENAI_API_KEY` | (required for ng-chat) | neuroglancer-chat LLM |
| `OPENAI_MODEL` | `gpt-5-nano` | Override neuroglancer-chat model |
| `NEUROGLANCER_BASE` | `https://neuroglancer-demo.appspot.com` | NG viewer base URL |

---

## Neuroglancer State Notes

Neuroglancer stores its entire state in the URL hash as JSON:
- Jump to coordinates: update `position` in state JSON → regenerate URL → `goto(url)`
- Annotation layer schema: `annotations` array at **layer level** (not inside `source`); each entry needs `"type": "point"/"box"/"ellipsoid"`; `source` must be `{"url": "local://annotations"}`

The cleanest MolmoWeb/Neuroglancer action pattern:
1. neuroglancer-chat `ng_set_view` generates target URL
2. MolmoWeb `goto(url=<new_ng_url>)` navigates
3. MolmoWeb screenshots to verify

---

## Known Issues (see SETUP_NOTES.md for full details)

- **torchvision/torch mismatch**: Dockerfile upgrades torch to 2.11+cu130 but leaves torchvision at 0.19.0. Fixed in `_dev_startup2.sh` (upgrades torchvision from pytorch cu130 index).
- **molmoweb agent/utils import**: Packages had no `__init__.py`; fixed by `_dev_startup2.sh` (creates them) and `PYTHONPATH=/code/lib/molmoweb` in subprocess env.
- **NativeActionPredictor float32 OOM**: Fixed in fork (`model_backends.py`) — sets `torch.bfloat16` before `build_model()`.
- **vllm OOM on T4**: Fixed via `--enforce-eager --max-model-len 512 --gpu-memory-utilization 0.99` in `run_capsule.py`.
- **MolmoWeb-4B download interruption**: `huggingface-cli download` is idempotent — re-run `_download_weights.sh` to resume.
