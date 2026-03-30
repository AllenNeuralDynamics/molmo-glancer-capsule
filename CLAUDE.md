# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session onboarding — read these first

| File | What it contains |
|------|-----------------|
| `CLAUDE.md` | This file — workspace layout, hardware, commands, architecture |
| `v2_plan.md` | Full v2 design: architecture, loop steps, milestones |
| `MEMORY_project.md` | Project context: repos, architecture decisions, key design choices |
| `MEMORY_user_profile.md` | Who the user is and how to work with them effectively |
| `MEMORY_feedback.md` | Working style preferences — pace, checkpoints, level of detail |
| `SETUP_NOTES.md` | Known env issues and fixes already applied to the code |

## Workspace Layout

This git repo (`molmo-glancer-capsule`) lives at `/root/capsule/`. The working paths are:

| Path | Purpose |
|------|---------|
| `/root/capsule/code/` | Scripts and library repos (git-tracked) |
| `/root/capsule/code/molmo_glancer.py` | **Main v2 pipeline** (to be created) |
| `/root/capsule/code/run_v2` | Shell entry point (to be created) |
| `/root/capsule/code/_dev_startup.sh` | Dep install script (pip-based; run once at capsule startup) |
| `/root/capsule/code/_download_weights.sh` | Weight download script (requires `HF_TOKEN`) |
| `/root/capsule/code/lib/neuroglancer-chat/` | neuroglancer-chat fork — used as library for `NeuroglancerState` |
| `/root/capsule/code/lib/molmoweb/` | Local MolmoWeb fork (not used in v2, kept as separate repo) |
| `/root/capsule/code/lib/OLMo-core/` | OLMo-core library (not used in v2, kept as separate repo) |
| `/root/capsule/code/lib/molmo-glancer/` | molmo-glancer library |
| `/root/capsule/data/` | Immutable input data |
| `/root/capsule/scratch/` → `/scratch/` | Unlimited scratch — model weights, caches |
| `/root/capsule/results/` → `/results/` | Curated outputs to export |

**Hardware:** 16 CPU cores, 64 GB RAM, **Tesla T4 GPU (15 GB VRAM, CUDA 13.0)**

---

## Project Goal

**molmo-glancer v2** — autonomous Neuroglancer question answering powered by a single unified model (Molmo2-O-7B).

Given an open-ended question about 3D data in Neuroglancer, the system iteratively plans views, screenshots them via Playwright, visually interprets them with Molmo2, and synthesizes a confident answer. All orchestration, visual understanding, and text reasoning use a single model loaded once.

---

## Architecture

### Orchestration Loop

```
User Question
    |
    v
+---------------------------+
|  ORCHESTRATION LOOP       |
|  (Molmo2-O-7B, text-only) |
+---------------------------+
    |               ^
    | view specs    | visual findings
    v               |
+------------------+  +-------------------+
| NeuroglancerState|  | Molmo2-O-7B       |
| (URL builder)    |  | (image + text)    |
+------------------+  +-------------------+
    |                       ^
    | NG URLs               | screenshots
    v                       |
+---------------------------+
|  Playwright (headless)    |
|  navigate + screenshot    |
+---------------------------+
```

### Loop Steps

1. **Plan** (text-only) — given question + findings so far, output view specs (position, zoom, layout, layers)
2. **Generate URLs** (NeuroglancerState) — deterministic URL construction, no LLM
3. **Screenshot** (Playwright) — navigate to each URL, wait for data load, capture PNG
4. **Interpret** (image+text) — pass screenshot + domain-specific prompt to Molmo2
5. **Decide** (text-only) — enough info? If not, loop to step 1. If yes, synthesize answer.

### Single Model: Molmo2-O-7B

- OLMo3-7B-Instruct backbone + SigLIP 2 vision encoder
- Handles BOTH text-only orchestration AND visual interpretation
- Loaded once, stays resident for the entire run
- 8-bit quantization (~8 GB) fits comfortably on T4 (15 GB VRAM)
- HuggingFace transformers inference (`AutoModelForImageTextToText`)

### NeuroglancerState (from neuroglancer-chat)

Used as a **library import only** — we import `NeuroglancerState` directly, not the FastAPI backend or Panel UI.

Key file: `code/lib/neuroglancer-chat/src/neuroglancer_chat/backend/tools/neuroglancer_state.py`

**Critical serialization rule:** Do NOT use `sort_keys=True` on NG state JSON — NG depends on dimension order (x,y,z,t).

---

## Key Commands

### Setup (run once after mounting `/code`)

```bash
# Install all dependencies
bash /code/_dev_startup.sh

# Download model weights (requires HF_TOKEN)
export HF_TOKEN=hf_...
bash /code/_download_weights.sh
```

### Run the pipeline

```bash
# Shell entry point (to be created)
bash /code/run_v2

# Direct invocation (to be created)
PLAYWRIGHT_BROWSERS_PATH=/scratch/ms-playwright python3 -u /code/molmo_glancer.py
```

### Tests

```bash
# neuroglancer-chat tests
cd /root/capsule/code/lib/neuroglancer-chat
python -m pytest tests/ -v

# molmo-glancer tests
cd /root/capsule/code/lib/molmo-glancer
python -m pytest tests/ -v
```

---

## Model Choice (fixed for T4)

| Role | Model | VRAM | Checkpoint |
|---|---|---|---|
| Orchestration + Vision | `allenai/Molmo2-O-7B` | ~8 GB (8-bit) | `/scratch/checkpoints/Molmo2-O-7B` |

v1 used two models (MolmoWeb-4B + OLMo-3-7B-Instruct) loaded sequentially. v2 uses one model loaded once.

---

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | (required for download) | HuggingFace access token |
| `PLAYWRIGHT_BROWSERS_PATH` | `/scratch/ms-playwright` | Chromium install location |
| `HF_HOME` | `/scratch/huggingface` | HF cache (keep off root overlay) |
| `NEUROGLANCER_BASE` | `https://neuroglancer-demo.appspot.com` | NG viewer base URL |

---

## Dependencies (v2)

| Package | Purpose | Notes |
|---------|---------|-------|
| `transformers>=4.57` | Molmo2-O-7B inference | Core |
| `bitsandbytes` | 8-bit quantization | Fits model on T4 |
| `accelerate` | device_map="auto" | HF standard |
| `molmo_utils` | Molmo2 processor helpers | Required per HF card |
| `decord2` | Video frame decoding | Required by Molmo2 (even if unused) |
| `torch` | Already installed | |
| `playwright` | Headless browser screenshots | Already installed |
| `neuroglancer-chat` | NeuroglancerState import | Already in lib/ |

vllm is **not needed** in v2 — inference is direct HuggingFace, not OpenAI-compatible server.

---

## Neuroglancer State Notes

Neuroglancer stores its entire state in the URL hash as JSON:
- Jump to coordinates: update `position` in state JSON → regenerate URL → `goto(url)`
- Annotation layer schema: `annotations` array at **layer level** (not inside `source`); each entry needs `"type": "point"/"box"/"ellipsoid"`; `source` must be `{"url": "local://annotations"}`

---

## Known Issues (see SETUP_NOTES.md for full details)

- **torchvision/torch mismatch**: Dockerfile upgrades torch to 2.11+cu130 but leaves torchvision at 0.19.0. Fixed in `_dev_startup.sh`.
- **molmoweb agent/utils import**: Packages had no `__init__.py`; fixed by `_dev_startup.sh`. (v1 issue, may not apply to v2)
