# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session onboarding — read these first

All context needed to resume work is in this directory. Read in this order:

| File | What it contains |
|------|-----------------|
| `CLAUDE.md` | This file — workspace layout, hardware, model choices, startup commands |
| `MEMORY_project.md` | Full project context: repos, architecture decisions, key design choices |
| `MEMORY_user_profile.md` | Who the user is and how to work with them effectively |
| `MEMORY_feedback.md` | Working style preferences — pace, checkpoints, level of detail |
| `IMPLEMENTATION_PLAN.md` | Detailed build plan: module structure, phases, GPU scheduling, startup sequence |
| `LOCAL_STACK_PLAN.md` | Earlier system design doc — component diagram, port map, VRAM analysis |
| `OLMO_INTEGRATION.md` | OLMo-core investigation — why OLMo-3-7B-Instruct, format details, adapter plan |
| `DEMO_PLAN.md` | The OUTDATED neuron-counting demo task that motivated the project |

## Workspace Layout

| Path | Purpose |
|------|---------|
| `/code/` | Source code — git-tracked, editable, part of molmo-glancer-capsule repo.|
| `/code/molmo-glancer/` | molmo-glancer library (this project — orchestration, CV, browser) |
| `/code/molmo-glancer-capsule/` | CodeOcean capsule / dev workspace / user entry point |
| `/code/molmoweb/` | Cloned MolmoWeb repo (AllenAI, unmodified) |
| `/code/neuroglancer-chat/` | Fork: github.com/seanmcculloch/neuroglancer-chat — branch `olmo-local` adds OLMo adapter |
| `/data/` | Immutable input data |
| `/scratch/` | Unlimited scratch space — use for model weights, caches, temp files |
| `/results/` | Curated outputs to export after session |

**Hardware:** 16 CPU cores, 64 GB RAM, **Tesla T4 GPU (15 GB VRAM, CUDA 13.0)**

**Project:** molmo-glancer — autonomous Neuroglancer question answering using a fully local AI stack.

---

## Project Goal

Run MolmoWeb as a visual web-agent to autonomously navigate **Neuroglancer** links. Neuroglancer is a WebGL-based 3D viewer for volumetric neuroscience data; navigating it requires interpreting visual state (camera position, layer visibility, segment selections) and issuing precise browser interactions (clicks at coordinates, scroll, URL navigation). MolmoWeb's screenshot-based action loop is well-suited to this.

---

## What is MolmoWeb

**MolmoWeb** (AllenAI, March 2026) is an open multimodal web agent built on Molmo 2 (OLMo-based). Given a natural-language task it autonomously controls a browser — clicking, typing, scrolling, navigating — by repeatedly:
1. Taking a screenshot of the current page
2. Predicting the next action (click at coordinates, type, scroll, navigate URL, etc.)
3. Executing that action via Playwright

### Models

| Model | Params | HF ID | Fits on T4? |
|-------|--------|-------|------------|
| MolmoWeb-4B | 4B | `allenai/MolmoWeb-4B` | Yes (~8 GB bfloat16) |
| MolmoWeb-4B-Native | 4B | `allenai/MolmoWeb-4B-Native` | Yes (~8 GB bfloat16) |
| MolmoWeb-8B | 8B | `allenai/MolmoWeb-8B` | **No** (~16 GB > 15 GB VRAM) |
| MolmoWeb-8B-Native | 8B | `allenai/MolmoWeb-8B-Native` | **No** (~16 GB > 15 GB VRAM) |

**Use 4B models only on this machine.** 8B requires quantization or a larger GPU.

**Benchmark results (8B):** WebVoyager 78.2%, DeepShop 42.3%, WebTailBench 49.5%.

### Architecture

```
[Browser (Playwright/Chromium)]
        ↓ screenshot (base64)
[MolmoWeb Inference Client]  ←→  POST /predict  →  [FastAPI Model Server]
        ↓ action string                                      ↓
[action executor (actions.py)]                    [HF or Native predictor]
```

The **model server** and **inference client** are fully decoupled via HTTP.

### Inference Backends

| Backend class | Notes |
|---|---|
| `HFActionPredictor` | HuggingFace Transformers; auto-detects CUDA |
| `NativeActionPredictor` | OLMo-native; hardcodes `cuda:0` — works fine now that GPU is present |
| `FastApiActionPredictor` | Remote HTTP endpoint |
| `ModalActionPredictor` | Serverless Modal endpoint |

Both `hf` and `native` backends work on this machine. The default `PREDICTOR_TYPE=native` in `start_server.sh` is correct.

---

## GPU-Accelerated Setup

With a Tesla T4 (15 GB VRAM), inference is fast: **~1–5 seconds per prediction step** (vs 30–120s on CPU). A 10–20 step Neuroglancer task completes in under 2 minutes.

### Fixed model choices for molmo-glancer

| Role | Model | VRAM |
|---|---|---|
| Visual agent | `allenai/MolmoWeb-4B` | ~8 GB |
| Text LLM (tool calling) | `allenai/OLMo-3-7B-Instruct` | ~14 GB |

These run **sequentially** on the T4 — 22 GB combined does not fit simultaneously. The pipeline is structured so they occupy separate phases. Do not use 8B MolmoWeb models (exceed T4 VRAM).

### Recommended Approach

1. Use `PREDICTOR_TYPE=native` (default) — works with CUDA
2. Download weights to `/scratch/checkpoints/` (not `/code/`)
3. Start FastAPI server; inference client connects to `http://127.0.0.1:8001`
4. Kill vLLM (OLMo) before starting MolmoWeb, and vice versa

---

## neuroglancer-chat: What It Is

**neuroglancer-chat** (`/code/neuroglancer-chat/`) takes a completely different approach to Neuroglancer control. Instead of visual understanding, it manipulates the Neuroglancer **JSON state directly** via URL encoding.

### How it works

```
User chat message
      ↓
FastAPI backend (main.py) → OpenAI function-calling loop (max 10 iters)
      ↓ tool calls
NeuroglancerState class — parse/mutate/serialize NG JSON state
      ↓ new URL
Panel UI updates embedded Neuroglancer viewer (panel-neuroglancer)
```

- **LLM**: OpenAI API (`gpt-5-nano` default, configurable via `OPENAI_MODEL`). **Requires `OPENAI_API_KEY`** — no local model.
- **Frontend**: Panel + panel-neuroglancer embedding a live Neuroglancer viewer; chat UI with streaming.
- **State**: `NeuroglancerState` class parses any NG URL (including `s3://`, `gs://`, `https://` pointer expansion), mutates it, and regenerates a canonical URL.
- **Data tools**: Upload CSVs → query with Polars → generate per-row NG navigation links → add annotations directly into viewer.

### neuroglancer-chat Tool Surface

| Category | Tools |
|----------|-------|
| State mutation | `ng_set_view`, `ng_set_lut`, `ng_add_layer`, `ng_set_layer_visibility`, `ng_annotations_add` |
| State read | `ng_state_summary`, `ng_state_link` |
| Data | `data_query_polars`, `data_ng_views_table`, `data_ng_annotations_from_data`, `data_plot`, `data_sample`, `data_describe`, `data_info` |
| Cloud | Pointer expansion: `s3://`, `gs://`, `https://` state JSON |

### Key Files

| File | Role |
|------|------|
| [src/.../backend/tools/neuroglancer_state.py](code/neuroglancer-chat/src/neuroglancer_chat/backend/tools/neuroglancer_state.py) | `NeuroglancerState` — full URL parse/mutate/serialize |
| [src/.../backend/adapters/llm.py](code/neuroglancer-chat/src/neuroglancer_chat/backend/adapters/llm.py) | OpenAI adapter + system prompt |
| [src/.../backend/main.py](code/neuroglancer-chat/src/neuroglancer_chat/backend/main.py) | FastAPI app + agent loop |
| [src/.../panel/panel_app.py](code/neuroglancer-chat/src/neuroglancer_chat/panel/panel_app.py) | Panel UI (~1500 lines) |
| [src/.../backend/tools/io.py](code/neuroglancer-chat/src/neuroglancer_chat/backend/tools/io.py) | CSV ingest + data tools |

---

## How neuroglancer-chat Helps With MolmoWeb

### The Core Problem MolmoWeb Faces with Neuroglancer

Neuroglancer is a **WebGL canvas app** — there is essentially no DOM to click, no accessible text, and the accessibility tree is largely empty. MolmoWeb's screenshot → coordinate click loop will work for some things (toolbar buttons, layer panel toggles), but:
- Navigating to a specific 3D coordinate requires typing into a position field precisely
- Changing layer contrast requires slider interactions — hard to target from coordinates
- Understanding what you're looking at requires reading the URL hash (the full state is encoded there)

### Proposed Hybrid Architecture

```
┌─────────────────────────────────────────────────────────┐
│  MolmoWeb (visual agent — screenshot → action)           │
│  • Handles: visual verification, ambiguous tasks,        │
│    exploratory navigation, "what do I see here?"        │
│  • Action: click, type, scroll, navigate_url            │
└──────────────────┬──────────────────────────────────────┘
                   │ screenshot loop
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Playwright / Chromium (headless browser)                │
│  • Opens neuroglancer-chat Panel app (localhost:8006)    │
│  • Or: opens raw Neuroglancer URL directly              │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  neuroglancer-chat FastAPI backend (localhost:8000)      │
│  • Handles: structured state changes, URL generation,   │
│    data-driven navigation, coordinate jumps             │
│  • MolmoWeb can POST to /tools/* directly as actions    │
└─────────────────────────────────────────────────────────┘
```

**The key insight**: MolmoWeb's action set could be extended with a `call_neuroglancer_api(tool, args)` action that POSTs to the neuroglancer-chat backend. MolmoWeb decides *what* to do from the screenshot; neuroglancer-chat executes *how* to do it without needing to visually interpret the page.

### Division of Labor

| Task | Best handled by |
|------|----------------|
| Navigate to a specific XYZ coordinate | neuroglancer-chat `ng_set_view` |
| Toggle layer visibility | neuroglancer-chat `ng_set_layer_visibility` |
| Adjust contrast/LUT | neuroglancer-chat `ng_set_lut` |
| Add annotation at coordinate | neuroglancer-chat `ng_annotations_add` |
| Generate per-cell navigation links from CSV | neuroglancer-chat `data_ng_views_table` |
| "What synapse structure do I see here?" | MolmoWeb (visual reasoning) |
| "Is this the right location?" (verify) | MolmoWeb (screenshot check) |
| Handle unexpected UI, popups, errors | MolmoWeb (flexible visual agent) |

---

## Key Files — MolmoWeb

| File | Role |
|------|------|
| [agent/fastapi_model_server.py](code/molmoweb/agent/fastapi_model_server.py) | FastAPI server exposing `/predict` |
| [agent/model_backends.py](code/molmoweb/agent/model_backends.py) | HF, Native, FastAPI, Modal backends |
| [agent/multimodal_agent.py](code/molmoweb/agent/multimodal_agent.py) | Main agent loop |
| [agent/actions.py](code/molmoweb/agent/actions.py) | Action parsing & execution |
| [inference/client.py](code/molmoweb/inference/client.py) | High-level `MolmoWeb` client |
| [inference/web_episode.py](code/molmoweb/inference/web_episode.py) | Episode/trajectory management |
| [scripts/start_server.sh](code/molmoweb/scripts/start_server.sh) | Server startup helper |
| [scripts/download_weights.sh](code/molmoweb/scripts/download_weights.sh) | Weight download helper |

---

## Environment Setup — Step by Step

### What's already installed

| Tool | Location | Notes |
|------|----------|-------|
| Python 3.12.4 | `/opt/conda/bin/python3` | Compatible with both repos (both require >=3.10) |
| `uv` | `/opt/conda/bin/uv` | Both repos use uv for deps — ready to use |
| Nothing else | — | torch, playwright, transformers, fastapi, panel — all need installing |

### Install neuroglancer-chat

```bash
cd /code/neuroglancer-chat
uv sync                        # core backend + data tools
uv sync --group panel          # adds panel + panel-neuroglancer for the UI
```

### Install MolmoWeb

```bash
cd /code/molmoweb
uv sync                        # installs transformers, playwright, fastapi, ai2-molmo2, etc.
uv run playwright install chromium        # install Chromium browser
uv run playwright install --with-deps chromium  # install system deps too (may be needed on Linux)
```

### Download model weights

```bash
cd /code/molmoweb
# 4B only — 8B exceeds T4 VRAM (15 GB)
bash scripts/download_weights.sh allenai/MolmoWeb-4B
# Downloads to ./checkpoints/MolmoWeb-4B — move to scratch to save /code/ space:
mv checkpoints/MolmoWeb-4B /scratch/checkpoints/MolmoWeb-4B
```

### Environment variables

```bash
# For neuroglancer-chat (required)
export OPENAI_API_KEY="sk-..."

# Optional neuroglancer-chat config
export OPENAI_MODEL="gpt-4o-mini"          # override default gpt-5-nano
export NEUROGLANCER_BASE="https://neuroglancer-demo.appspot.com"  # default NG viewer
export NEUROGLANCER_CHAT_DEBUG="1"         # verbose logging
export USE_STREAMING="true"                # SSE streaming mode

# For MolmoWeb model server (GPU)
export CKPT=/scratch/checkpoints/MolmoWeb-4B
export PORT=8001
# PREDICTOR_TYPE defaults to "native" in start_server.sh — works fine with cuda:0
```

### Startup sequence

```bash
# Terminal 1: neuroglancer-chat backend
cd /code/neuroglancer-chat/src/neuroglancer_chat
export OPENAI_API_KEY="sk-..."
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: neuroglancer-chat frontend (Panel UI)
cd /code/neuroglancer-chat/src/neuroglancer_chat
export BACKEND="http://127.0.0.1:8000"
uv run python -m panel serve panel/panel_app.py --port 8006 --address 127.0.0.1 \
  --allow-websocket-origin=127.0.0.1:8006

# Terminal 3: MolmoWeb model server (GPU, native backend)
cd /code/molmoweb
export CKPT=/scratch/checkpoints/MolmoWeb-4B
bash scripts/start_server.sh   # defaults: PREDICTOR_TYPE=native, PORT=8001

# Terminal 4: MolmoWeb client (Python)
cd /code/molmoweb
uv run python - <<'EOF'
from inference import MolmoWeb
client = MolmoWeb(endpoint="http://127.0.0.1:8001", local=True, headless=False)
traj = client.run("Go to http://localhost:8006 and ...", max_steps=15)
traj.save_html(query="neuroglancer task")
EOF
```

### Tests (neuroglancer-chat)

```bash
cd /code/neuroglancer-chat
uv run -m coverage run -m pytest
uv run python -m pytest tests/test_integration_query_with_links.py -v -s  # integration test
```

---

## MolmoWeb Internal Details

### Model input/output format

The model receives a Jinja2-rendered prompt string prepended with a system token:

```
"molmo_web_think: \n# GOAL\n<task>\n\n# PREVIOUS STEPS\n## Step 1\nTHOUGHT: ...\nACTION: ...\n\n# CURRENTLY ACTIVE PAGE\nPage 0: <title> | <url>\n\n# NEXT STEP\n"
```

The model outputs a **JSON string** (no markdown wrapper):

```json
{
  "thought": "I need to click the position box to enter coordinates",
  "action": {"name": "click", "x": 45.2, "y": 12.8},
  "action_description": "click on position input field"
}
```

**Coordinates are percentages (0–100)**, not pixels. The agent code converts them to pixels using viewport dimensions (default 1280×720).

### Full action vocabulary

| Action | Key params | Notes |
|--------|-----------|-------|
| `click` / `mouse_click` | `x`, `y`, `button`, `click_type` | Coords in % |
| `dblclick` | `x`, `y` | Double-click |
| `hover_at` | `x`, `y`, `duration` | Hover for duration seconds |
| `drag_and_drop` | `from_x`, `from_y`, `to_x`, `to_y` | Mouse drag |
| `scroll` | `delta_x`, `delta_y` | Page scroll (px) |
| `scroll_at` | `x`, `y`, `delta_x`, `delta_y` | Scroll at specific position |
| `type` / `keyboard_type` | `text` | Type text |
| `keypress` / `keyboard_press` | `key` | Enter, Escape, Backspace, Tab, Arrows, Ctrl+a/c/v, F5 |
| `goto` | `url` | Navigate to URL |
| `browser_nav` | `nav_type` (go_back/new_tab/tab_focus) | Tab management |
| `noop` | `noop_reason` | Wait (loading/captcha/retrying) |
| `send_msg_to_user` | `msg` | `[EXIT]` or `[ANSWER]` prefix terminates loop |
| `report_infeasible` | `infeasibility_reason` | Abort task |

### Browser environment (SimpleEnv)

- Playwright Chromium with `--disable-blink-features=AutomationControlled` (stealth)
- Screenshots via CDP `Page.captureScreenshot`
- After each action: 0.5s sleep + wait for `domcontentloaded` + 0.5s buffer
- axtree extraction disabled by default (`extract_axtree=False`); call `client.get_axtree()` on demand

### Trajectory output

`traj.save_html()` → `inference/htmls/<slug>.html` (self-contained with screenshots + thoughts).
`save_trajectory_screenshots_png(traj, output_dir)` dumps individual PNGs.

### Alternative agents (non-visual baselines)

- [agent/gpt_axtree_agent.py](code/molmoweb/agent/gpt_axtree_agent.py) — GPT-5 with accessibility tree
- [agent/gemini_axtree_agent.py](code/molmoweb/agent/gemini_axtree_agent.py) — Gemini with axtree
- [agent/gemini_cua.py](code/molmoweb/agent/gemini_cua.py) — Gemini Computer Use

For Neuroglancer these will have sparse axtrees (WebGL canvas). Configured via `GPT_AXTREE_MODEL` / `OPENAI_API_KEY`.

---

## neuroglancer-chat Internal Details

### Neuroglancer annotation JSON schema (verified)

```json
{
  "type": "annotation",
  "source": {"url": "local://annotations"},
  "tool": "annotatePoint",
  "tab": "annotations",
  "annotationColor": "#cecd11",
  "annotations": [
    {
      "point": [4998.87, 6216.5, 1175.07],
      "type": "point",
      "id": "bb79d5acc705a03fad2cc116a192df2c8a41e249"
    }
  ],
  "name": "MyLayer"
}
```

**Critical rules:**
- `annotations` array is at the **layer level**, NOT inside `source`
- Each annotation needs a `type` field (`"point"`, `"box"`, or `"ellipsoid"`)
- `source` must be `{"url": "local://annotations"}` — not a points array
- `tool`, `tab`, `annotationColor` are required for proper NG UI integration

### State serialization

`NeuroglancerState.to_url()` produces compact, sorted-key JSON. `from_url()` accepts full URLs, `#{...}` fragments, raw JSON dicts, or `s3://`/`gs://` pointers.

**Dimension order**: do NOT use `sort_keys=True` — NG depends on dimension order (x,y,z,t).

### Summary ID chaining pattern

`data_query_polars` returns a `summary_id` (e.g. `"query_456"`). Follow-up tools like `data_ng_annotations_from_data` accept `summary_id=` instead of `file_id=`. Use `summary_id="last"` for the most recent query.

### neuroglancer-chat known limitations

- `SEND_DATA_TO_LLM = False` hardcoded in `main.py` — LLM sees only metadata, not actual rows
- Global in-process `CURRENT_STATE` — no multi-user isolation
- Streaming mode (SSE) lacks feature parity (no Tabulator, plots, multi-view tables) — Phase 3 TODO
- `io.py` `load_csv` for S3 paths is stub code (incomplete)
- PowerShell launch scripts in `scripts/` — use manual uvicorn commands on Linux

### NEUROGLANCER_BASE config

Set via `NEUROGLANCER_BASE` env var (default: `https://neuroglancer-demo.appspot.com`). For AIND data, set to the AIND NG instance URL.

---

## Known Limitations — MolmoWeb

- Struggles with text recognition from screenshots
- Timing issues (scrolling before page loads)
- Limited drag-and-drop (relevant for Neuroglancer canvas panning)
- Not trained on login-protected tasks or WebGL-heavy apps
- The NG UI is almost entirely WebGL canvas — clicking on it lands on an opaque canvas

---

## Neuroglancer-Specific Notes

Neuroglancer stores its **entire state in the URL hash** — layers, camera position, shader code, annotations, layout — all as a JSON object:

- To jump to coordinates: update `position` in state JSON, regenerate URL, navigate browser
- To change layer visibility: update `visible` field in layer object
- To add annotations: add annotation layer with `annotations` array (see schema above)

**For Neuroglancer navigation with MolmoWeb**, the cleanest approach is:
1. neuroglancer-chat generates the target URL
2. MolmoWeb executes `goto(url=<new_ng_url>)` — the single most reliable NG action
3. MolmoWeb screenshots to verify visual result
4. Repeat

This avoids MolmoWeb needing to "click" inside the NG canvas at all.
