# molmo-glancer — Implementation Plan

## Repository Roles

```
/code/molmo-glancer/              ← Python library (pip-installable)
  Orchestration logic, CV analysis, browser automation, pipeline abstractions.
  Imported by capsule scripts. Published to PyPI eventually.

/code/molmo-glancer-capsule/      ← CodeOcean capsule (dev workspace + user entry point)
  Task scripts, notebooks, config, example links, results.
  Installs molmo-glancer as a dependency. Mirrors /capsule at runtime.
  NOT a library — never imported by molmo-glancer.

/code/neuroglancer-chat/          ← Fork: github.com/seanmcculloch/neuroglancer-chat
  Branch: olmo-local
  Modified only in adapters/llm.py + new adapters/olmo_adapter.py.
  Runs as a background service; molmo-glancer talks to it over HTTP.

/code/molmoweb/                   ← Unmodified upstream
  Runs as a background service; molmo-glancer talks to it over HTTP.
```

The dependency direction is strictly one-way:

```
molmo-glancer-capsule
  └── imports molmo_glancer (library)
        └── HTTP calls → neuroglancer-chat backend (port 8000)
        └── HTTP calls → MolmoWeb model server (port 8001)
        └── Playwright (in-process, no HTTP)
```

---

## System Architecture (updated with two-pass loop)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  USER  (molmo-glancer-capsule)                                               ║
║  run_capsule.py  /  notebooks  /  task scripts                               ║
╚══════════════════════════╦═══════════════════════════════════════════════════╝
                           ║ imports
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  molmo_glancer  (library)                                                    ║
║                                                                              ║
║  pipeline.AlignmentSweep.run(ng_url, task_description)                      ║
║  │                                                                           ║
║  ├─ Phase 1: Broad sweep                                                     ║
║  │   ng.grid.make_position_grid(bounds, step)                               ║
║  │       → list of (x,y,z)                                                  ║
║  │   ng_chat.generate_urls(positions)                                        ║
║  │       → HTTP POST neuroglancer-chat :8000  [OLMo-3-7B generates URLs]    ║
║  │   browser.screenshot_sweep.capture(urls)                                  ║
║  │       → Playwright (no model) → /scratch/pass1/*.png                     ║
║  │   analysis.alignment.score_all(pass1_dir)                                ║
║  │       → ranked DataFrame (position, alignment_score, misalignment_score)  ║
║  │                                                                           ║
║  ├─ Phase 2: Targeted refinement                                             ║
║  │   (top-N worst positions from Phase 1)                                   ║
║  │   ng_chat.generate_urls(worst_positions, zoom=closer, orientations=all)   ║
║  │   browser.screenshot_sweep.capture(urls, render_wait=diff)               ║
║  │       → /scratch/pass2/*.png                                             ║
║  │   analysis.alignment.score_all(pass2_dir)                                ║
║  │                                                                           ║
║  └─ Phase 3: Visual verification (selective)                                 ║
║      molmoweb.verify(worst_5_urls, task="describe alignment quality")        ║
║          → HTTP POST MolmoWeb :8001 → Playwright + MolmoWeb-4B model        ║
║          → natural-language observations + trajectory HTML                   ║
║                                                                              ║
║  reporter.save_results(scores, screenshots, urls, observations)              ║
║      → /results/report.html  +  /results/ng_states.txt                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
        │ HTTP                              │ HTTP
        ▼                                  ▼
╔═══════════════════════╗      ╔═══════════════════════════╗
║  neuroglancer-chat    ║      ║  MolmoWeb FastAPI          ║
║  (olmo-local branch)  ║      ║  (upstream, unmodified)    ║
║  FastAPI  :8000       ║      ║  FastAPI  :8001            ║
║                       ║      ║                            ║
║  OLMo adapter         ║      ║  MolmoWeb-4B (~8 GB)       ║
║    ↕ HTTP             ║      ║  Playwright (in-process)   ║
║  vLLM  :8002          ║      ╚═══════════════════════════╝
║  OLMo-3-7B-Instruct  ║
║  (~14 GB bfloat16)    ║
╚═══════════════════════╝
```

---

## molmo-glancer Package Structure

```
src/molmo_glancer/
│
├── __init__.py
│
├── browser/                        # Playwright automation (no model inference)
│   ├── __init__.py
│   ├── screenshot_sweep.py         # capture(urls, out_dir, render_wait) → list[Path]
│   └── render_wait.py              # RenderWait strategies: FixedSleep | ScreenshotDiff | NetworkIdle
│
├── analysis/                       # CV on screenshots — pure numpy, no GPU
│   ├── __init__.py
│   ├── alignment.py                # score_screenshot(png) → dict[str,float]
│   │                               # score_all(directory) → polars.DataFrame
│   └── nucleus.py                  # (future) blob_detect(png) → list[centroid]
│
├── ng/                             # Neuroglancer state utilities
│   ├── __init__.py
│   ├── grid.py                     # make_position_grid(bounds, step_um) → list[(x,y,z)]
│   └── client.py                   # NGChatClient — HTTP wrapper around neuroglancer-chat API
│                                   # generate_urls(positions, state) → list[str]
│                                   # state_load(url) → dict
│                                   # state_link() → str
│
├── molmoweb/                       # MolmoWeb client utilities
│   ├── __init__.py
│   ├── client.py                   # MolmoGlancerClient wrapping molmoweb's MolmoWeb class
│   └── reporter.py                 # save_trajectory(traj, out_dir)
│                                   # save_screenshots_png(traj, out_dir)
│
└── pipeline/                       # High-level orchestration
    ├── __init__.py
    ├── base.py                     # PipelineStep ABC, Result dataclass
    └── alignment_sweep.py          # AlignmentSweep — the two-pass + verify loop
```

---

## molmo-glancer-capsule Structure

```
code/
├── run_capsule.py                  # CodeOcean entry point — calls a configured task
│
├── tasks/
│   ├── alignment_sweep.py          # "Find bad alignment regions in this NG link"
│   └── neuron_count.py             # "Count nuclei and annotate this volume tile"
│
└── notebooks/
    ├── 01_setup_check.ipynb        # Verify services, GPU, model weights
    ├── 02_alignment_demo.ipynb     # Walk through alignment sweep interactively
    └── 03_neuron_count_demo.ipynb  # Walk through nucleus detection interactively

environment/
└── Dockerfile                      # Extends base; installs all service deps

# Config / data (in capsule root, mirrored from /capsule)
example_ng_link.txt
CLAUDE.md
```

---

## neuroglancer-chat Fork — Changes Required

Branch: `olmo-local` in `/code/neuroglancer-chat`

### New file: `src/neuroglancer_chat/backend/adapters/olmo_adapter.py`

Responsibilities:
1. Build the full OLMo prompt: system message + `<functions>` XML block + conversation history formatted as `<|im_start|>role\ncontent<|im_end|>` turns + `<|im_start|>assistant\n` suffix
2. POST to vLLM `/v1/completions` (plain text, not chat/tools — the adapter owns the format)
3. Extract `<function_calls>...</function_calls>` from the response
4. Parse each line as a Python-style call: `tool_name(arg=val, arg2=val2)`
5. Return an OpenAI-shaped response dict so `main.py`'s agent loop needs zero changes

Key parsing challenge — OLMo outputs Python-syntax kwargs. The parser must handle:
- Simple scalars: `ng_set_view(zoom=2.0)`
- String values: `ng_set_lut(layer="l", vmin=1, vmax=800)`
- Nested dicts: `ng_set_view(center={"x":100,"y":200,"z":50})`
- Arrays: `ng_annotations_add(items=[{"type":"point","center":{"x":1,"y":2,"z":3}}])`

Strategy: regex to split `name(...)` into name and args string; use `ast.parse` on a dummy assignment `f({args_string})` to get a safe AST; walk the AST to reconstruct a Python dict. Fall back to `json.loads` wrapping if the args look like pure JSON (OLMo sometimes outputs JSON kwargs).

### Modified file: `src/neuroglancer_chat/backend/adapters/llm.py`

Add at top:
```python
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")  # "openai" | "olmo"

if LLM_BACKEND == "olmo":
    from .olmo_adapter import run_chat, run_chat_stream
```

Remove `reasoning_effort="minimal"` from the OpenAI call (it's OpenAI-only and will error on other endpoints). Everything else in `llm.py` is unchanged.

### No other files need changing in neuroglancer-chat.

---

## Target Models

Both models are fixed for this project. All code in molmo-glancer and molmo-glancer-capsule is written with these specific models in mind — no model-agnostic abstractions needed yet.

| Role | Model | HF ID | VRAM (bfloat16) |
|---|---|---|---|
| Text LLM + tool calling | OLMo-3-7B-Instruct | `allenai/OLMo-3-7B-Instruct` | ~14 GB |
| Visual web agent | MolmoWeb-4B | `allenai/MolmoWeb-4B` | ~8 GB |

These run **sequentially**, not simultaneously, on the Tesla T4 (15 GB VRAM). The pipeline is structured so they naturally occupy separate phases with no overlap.

---

## GPU Scheduling and Startup Times

### Why sequential is fine for this pipeline

The bulk of the alignment sweep pipeline — position grid generation, Playwright screenshot capture, and color analysis — **uses no GPU at all**. GPU is only needed at the bookends:

```
[OLMo: task parse + URL gen]  [CPU-only work: ~80% of runtime]  [MolmoWeb: verify]
       GPU phase 1                                                    GPU phase 2
```

Switching happens exactly once per pipeline run.

### Load/unload timing

| Event | Time | Notes |
|---|---|---|
| vLLM starts, loads OLMo-3-7B-Instruct | ~60–90 sec | 14 GB disk → GPU, CUDA graph compile |
| OLMo LLM calls (2–3 per run) | ~10–20 sec | Short prompts, fast on GPU |
| Kill vLLM | ~2 sec | GPU memory freed immediately |
| MolmoWeb starts, loads MolmoWeb-4B | ~30–60 sec | 8 GB disk → GPU |
| MolmoWeb verifications (5 positions) | ~25–50 sec | ~5–10 sec per screenshot+inference |
| Kill MolmoWeb | ~2 sec | |

**Page cache effect:** The capsule has 128 GB system RAM. After the first load of each model, the OS keeps the weight files in the page cache. Subsequent loads in the same session read from RAM rather than disk — roughly 3–5× faster. First run is the slow one.

### GPU availability during the screenshot sweep

When OLMo is unloaded before the Playwright sweep, all 15 GB is free. Chromium uses the GPU for WebGL rendering (Neuroglancer is a WebGL app), so the sweep runs with a fully available GPU for rendering. Attempting to screenshot while OLMo is still loaded would leave only ~1 GB for Chromium — possible but risky for large tile loads.

Color analysis (`score_screenshot`) is pure numpy/PIL on CPU regardless. GPU state has no effect on it.

### Pipeline phase boundaries (what should be running when)

```
Phase          vLLM (OLMo)   MolmoWeb    Playwright   numpy/CPU
─────────────────────────────────────────────────────────────────
Task parse       RUNNING        off          off          off
Grid gen         optional*      off          off          RUNNING
Screenshot sweep   off          off        RUNNING       RUNNING
Color analysis     off          off          off          RUNNING
Result interp    RUNNING        off          off          off
MolmoWeb verify    off        RUNNING      RUNNING        off
─────────────────────────────────────────────────────────────────
* Grid gen uses NeuroglancerState directly (Python library, no LLM).
  OLMo not required unless the task requires reasoning about bounds.
```

### Practical sequence for a beginner

The `AlignmentSweep` pipeline class will print explicit instructions at each phase boundary:

```
[molmo-glancer] OLMo phase complete. To continue:
  1. Kill vLLM:  kill <PID>  (or Ctrl+C in its terminal)
  2. Confirm GPU is free:  nvidia-smi
  3. Press Enter to start the screenshot sweep...
```

No scripting required. The heavy automation is in the sweep itself — you're just managing two terminal windows.

---

## Implementation Phases

### Phase 0 — Environment (prerequisite for everything)

| Step | Command | Notes |
|---|---|---|
| Install vLLM | `pip install vllm` | System Python; ~15 min first time |
| Install zarr/s3fs | `pip install zarr s3fs scikit-image` | For data pipeline tasks |
| Install ng-chat deps | `cd /code/neuroglancer-chat && uv sync && uv sync --group panel` | |
| Install MolmoWeb deps | `cd /code/molmoweb && uv sync` | |
| Install Chromium | `cd /code/molmoweb && uv run playwright install --with-deps chromium` | |
| Download MolmoWeb-4B | `bash /code/molmoweb/scripts/download_weights.sh allenai/MolmoWeb-4B` then `mv /code/molmoweb/checkpoints/MolmoWeb-4B /scratch/checkpoints/` | ~8 GB |
| Download OLMo-3-7B | `huggingface-cli download allenai/OLMo-3-7B-Instruct --local-dir /scratch/checkpoints/OLMo-3-7B-Instruct` | ~14 GB |

---

### Phase 1 — neuroglancer-chat OLMo adapter

**Goal:** neuroglancer-chat backend answers a simple NG query using OLMo-3-7B-Instruct locally, with no OpenAI key.

**Files to write:**
- `/code/neuroglancer-chat/src/neuroglancer_chat/backend/adapters/olmo_adapter.py`

**Files to modify:**
- `/code/neuroglancer-chat/src/neuroglancer_chat/backend/adapters/llm.py` — add backend switch, remove `reasoning_effort`

**Test:**
```bash
# Start vLLM
vllm serve /scratch/checkpoints/OLMo-3-7B-Instruct --port 8002 --max-model-len 8192

# Start ng-chat (olmo-local branch)
cd /code/neuroglancer-chat/src/neuroglancer_chat
LLM_BACKEND=olmo OLMO_URL=http://127.0.0.1:8002 uv run uvicorn backend.main:app --port 8000

# Test: simple state query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Load this link and summarise the layers", "ng_link": "<example_link>"}'
```

**Success criterion:** OLMo correctly calls `state_load` then `ng_state_summary` and returns a coherent response. No OpenAI key involved.

---

### Phase 2 — molmo-glancer: browser screenshot sweep

**Goal:** Given a list of NG URLs, capture a PNG for each using Playwright directly (no MolmoWeb model).

**Files to write:**
- `src/molmo_glancer/browser/render_wait.py` — `FixedSleep(seconds)`, `ScreenshotDiff(threshold, max_wait)`
- `src/molmo_glancer/browser/screenshot_sweep.py` — `capture(urls, out_dir, render_wait, viewport) → list[Path]`

**pyproject.toml deps to add:** `playwright`, `pillow`

**Test:**
```python
from molmo_glancer.browser.screenshot_sweep import capture
from molmo_glancer.browser.render_wait import FixedSleep

urls = [open("/root/capsule/example_ng_link.txt").read().strip()]
paths = capture(urls, out_dir="/scratch/test_sweep", render_wait=FixedSleep(3))
print(paths)  # [PosixPath('/scratch/test_sweep/0000.png')]
```

**Success criterion:** PNG saved, not black, shows tissue data (not a loading spinner).

---

### Phase 3 — molmo-glancer: alignment analysis

**Goal:** Score a directory of PNGs for channel alignment quality.

**Files to write:**
- `src/molmo_glancer/analysis/alignment.py`
  - `score_screenshot(png_path) → dict` — per-image metrics: `alignment`, `misalignment`, `signal_frac`
  - `score_all(directory) → polars.DataFrame` — ranked table

**pyproject.toml deps to add:** `numpy`, `polars`

**Test:**
```python
from molmo_glancer.analysis.alignment import score_all
df = score_all("/scratch/test_sweep")
print(df.sort("misalignment", descending=True))
```

**Success criterion:** Returns a DataFrame with scores; passes basic sanity checks (black images have `signal_frac ≈ 0`, yellow-dominant images have `alignment > misalignment`).

---

### Phase 4 — molmo-glancer: NG position grid + NGChatClient

**Goal:** Generate systematic position grids over a volume; talk to neuroglancer-chat to produce URLs.

**Files to write:**
- `src/molmo_glancer/ng/grid.py`
  - `make_position_grid(center, bounds_um, step_um, z_slices) → list[tuple[float,float,float]]`
- `src/molmo_glancer/ng/client.py`
  - `NGChatClient(base_url)` with methods:
    - `state_load(ng_url)` — POST /chat with `state_load` instruction
    - `generate_view_urls(positions, base_state) → list[str]` — calls `ng_set_view` + `ng_state_link` per position
    - `get_state_summary() → dict`

**pyproject.toml deps to add:** `httpx`

**Note on `generate_view_urls`:** This method drives an OLMo tool-call loop. It sends a batch instruction like "for each of these coordinates, call ng_set_view then ng_state_link and return all the URLs." OLMo handles the iteration. Alternatively (simpler for now): call `NeuroglancerState` directly as a library, bypassing the chat interface, and mutate the state object per-position in Python. The latter is faster and more reliable for a deterministic grid.

**Two-speed design:**
- **Fast path:** `NGChatClient.url_for_position(state, x, y, z)` — calls `NeuroglancerState` directly as a Python library, no LLM needed, generates URLs in milliseconds. Use this for the grid.
- **Slow path (LLM):** Use the chat interface only for ambiguous user queries ("find the brightest region") that need reasoning.

---

### Phase 5 — molmo-glancer: pipeline orchestration

**Goal:** The two-pass alignment sweep as a callable pipeline.

**Files to write:**
- `src/molmo_glancer/pipeline/base.py` — `PipelineResult` dataclass (scores_df, urls, screenshot_paths, observations)
- `src/molmo_glancer/pipeline/alignment_sweep.py` — `AlignmentSweep` class:
  ```python
  sweep = AlignmentSweep(
      ng_url="https://...",
      ng_chat_url="http://localhost:8000",
      molmoweb_url="http://localhost:8001",   # optional
      pass1_step_um=50,     # coarse grid step
      pass1_z_slices=10,
      pass2_top_n=10,       # worst positions to refine
      pass2_orientations=["xy", "xz"],
      verify_worst_n=5,     # how many to send to MolmoWeb
      out_dir="/results/alignment_sweep",
  )
  result = sweep.run()
  ```

**Test:** Run full sweep on `example_ng_link.txt`. Produces:
- `/results/alignment_sweep/pass1/` — coarse PNG grid
- `/results/alignment_sweep/pass2/` — refined PNGs at worst locations
- `/results/alignment_sweep/scores.csv` — full ranking
- `/results/alignment_sweep/report.html` — summary

---

### Phase 6 — molmo-glancer: MolmoWeb integration

**Goal:** Wire in MolmoWeb for final visual verification of the worst-N positions.

**Files to write:**
- `src/molmo_glancer/molmoweb/client.py`
  - `MolmoGlancerClient(endpoint)` — thin wrapper around molmoweb's `MolmoWeb` class
  - `verify_url(url, task_description, max_steps=6) → VerificationResult`
- `src/molmo_glancer/molmoweb/reporter.py`
  - `save_trajectory(traj, out_dir)` — saves HTML + PNGs

**Dependency:** `molmoweb` (editable install from `/code/molmoweb`)

**Test:** Verify one URL, confirm trajectory HTML saved.

---

### Phase 7 — molmo-glancer-capsule: task scripts + entry point

**Goal:** Clean user-facing interface in the capsule.

**Files to write:**

`code/tasks/alignment_sweep.py`:
```python
"""
Task: Find channel misalignment regions in a Neuroglancer link.
Configure via environment variables or edit the constants below.
"""
from molmo_glancer.pipeline.alignment_sweep import AlignmentSweep

NG_URL = os.getenv("NG_URL", open("/capsule/example_ng_link.txt").read().strip())
OUT_DIR = "/results/alignment_sweep"

sweep = AlignmentSweep(ng_url=NG_URL, out_dir=OUT_DIR, ...)
result = sweep.run()
print(f"Found {len(result.bad_alignment_positions)} misalignment regions")
print(f"Results: {OUT_DIR}")
```

`code/run_capsule.py` (replace stub):
```python
"""CodeOcean entry point — dispatches to named task."""
TASK = os.getenv("TASK", "alignment_sweep")
if TASK == "alignment_sweep":
    from tasks.alignment_sweep import run; run()
elif TASK == "neuron_count":
    from tasks.neuron_count import run; run()
```

---

## Service Startup Reference

All five processes for a full run:

```bash
# 1. OLMo-3-7B-Instruct (Phase 1 GPU — text reasoning + tool calls)
vllm serve /scratch/checkpoints/OLMo-3-7B-Instruct \
  --port 8002 --max-model-len 8192

# 2. neuroglancer-chat backend (olmo-local branch)
cd /code/neuroglancer-chat/src/neuroglancer_chat
LLM_BACKEND=olmo OLMO_URL=http://127.0.0.1:8002 \
  uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 3. neuroglancer-chat Panel UI (optional — for interactive exploration)
cd /code/neuroglancer-chat/src/neuroglancer_chat
BACKEND=http://127.0.0.1:8000 \
  uv run python -m panel serve panel/panel_app.py \
  --port 8006 --address 127.0.0.1 --allow-websocket-origin=127.0.0.1:8006

# 4. MolmoWeb model server (Phase 2 GPU — visual verification only)
#    Start after OLMo work is done, or run simultaneously if OLMo is INT4
cd /code/molmoweb
CKPT=/scratch/checkpoints/MolmoWeb-4B \
  bash scripts/start_server.sh   # port 8001, PREDICTOR_TYPE=native

# 5. Run a task (capsule entry point)
cd /code/molmo-glancer-capsule
NG_URL="$(cat /capsule/example_ng_link.txt)" \
  python code/tasks/alignment_sweep.py
```

---

## What Belongs Where — Quick Reference

| Code | Repo | Rationale |
|---|---|---|
| OLMo adapter (tool format translation) | `neuroglancer-chat` (olmo-local branch) | Modifies neuroglancer-chat's LLM interface |
| Playwright bulk screenshotter | `molmo-glancer` | Reusable browser automation, no ng-chat dependency |
| Screenshot color analysis | `molmo-glancer` | Pure CV, reusable across tasks |
| NG position grid generation | `molmo-glancer` | Reusable geometry utility |
| NGChatClient (HTTP wrapper) | `molmo-glancer` | Encapsulates ng-chat service interface |
| MolmoWeb client wrapper | `molmo-glancer` | Encapsulates molmoweb service interface |
| Two-pass pipeline orchestration | `molmo-glancer` | The core logic of this project |
| Task scripts (alignment_sweep, neuron_count) | `molmo-glancer-capsule` | User-facing, capsule-specific config |
| Notebooks | `molmo-glancer-capsule` | Interactive, capsule-specific |
| Run entry point (`run_capsule.py`) | `molmo-glancer-capsule` | CodeOcean-specific |
| Model weights | `/scratch/checkpoints/` | Not in any repo |
| Results / outputs | `/results/` | Not in any repo |

---

## Open Questions Before Implementation

1. **Volume bounds for pass 1 grid**: The example NG link has no explicit volume bounds in the state JSON — just a current position. Do we infer bounds from the Zarr array metadata (call `zarr.open(s3_path).shape`), ask OLMo to estimate from context, or require the user to specify a bounding box?

2. **OLMo multi-tool-call in one response**: For `generate_view_urls`, does OLMo reliably emit multiple `<function_calls>` blocks in a single response, or one per response turn? This determines whether URL generation is one LLM call or N calls. We should test this early in Phase 1.

3. **Render wait for S3 Zarr data**: The example link streams from `s3://aind-open-data/`. Tile load time depends on S3 latency and zoom level. What's the minimum zoom-out level where pass-1 screenshots are useful? Worth empirically measuring before setting the fixed-sleep duration.

4. **`ng_set_shader` tool**: The alignment sweep assumes additive blend mode is already set in the provided NG link. If not, neuroglancer-chat needs a new tool to set the shader. Should this be built in Phase 1 alongside the adapter, or deferred to when a user provides a link that needs it?
