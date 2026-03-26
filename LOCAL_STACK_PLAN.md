# Local Stack Plan: End-to-End Neuroglancer Question Answering

## System Description

This system combines three independently running services to achieve autonomous, locally-executed Neuroglancer question answering — no external API calls required.

**neuroglancer-chat** is the reasoning and state-manipulation layer. It takes a natural-language question about a Neuroglancer view, uses a local text LLM (OLMo-3-7B-Instruct) to plan a sequence of tool calls, and executes those calls against a `NeuroglancerState` object that parses and regenerates Neuroglancer URLs. The output is a new URL encoding the updated viewer state (repositioned camera, new annotation layers, changed LUTs, etc.).

**MolmoWeb-4B** is the visual verification layer. It controls a headless Chromium browser via Playwright, navigates to Neuroglancer URLs, screenshots the rendered WebGL canvas, and interprets what it sees. Because Neuroglancer is an opaque WebGL app with no DOM, MolmoWeb is the only component that can confirm a URL change actually produced the expected visual result.

**OLMo-3-7B-Instruct** is the local text LLM backbone for neuroglancer-chat. It was SFT-trained on AllenAI's `olmo-toolu-*` tool-use datasets and natively produces structured function calls using a custom XML+Python-syntax format. A thin adapter layer (to be written in the neuroglancer-chat fork) handles translation between OLMo's format and the internal tool dispatch.

The two GPU models (OLMo-3-7B-Instruct and MolmoWeb-4B) are used in separate phases of the workflow and share the single T4 GPU sequentially in their default bfloat16 precision. If simultaneous operation is needed, INT4 quantization of OLMo-3-7B reduces its footprint from ~14 GB to ~3.5 GB, allowing both to run at once.

---

## Workflow Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  USER                                                                        ║
║  "Load this NG link, find the brightest cell layer, add annotation points"  ║
╚══════════════════════════╦═══════════════════════════════════════════════════╝
                           ║ HTTP / WebSocket
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  neuroglancer-chat Panel UI  (port 8006)                                    ║
║  • Panel + panel-neuroglancer — live embedded NG viewer                     ║
║  • Chat input / response display / plot/table rendering                     ║
╚══════════════════════════╦═══════════════════════════════════════════════════╝
                           ║ HTTP POST /chat
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  neuroglancer-chat FastAPI Backend  (port 8000)           [FORK BRANCH]     ║
║                                                                              ║
║  ┌──────────────────────────────────────────────────────┐                   ║
║  │  Agent loop (max 10 iterations)                      │                   ║
║  │  1. Build messages + tool schemas                    │                   ║
║  │  2. Call OLMo adapter → get tool calls               │                   ║
║  │  3. Dispatch tool calls → results                    │                   ║
║  │  4. Append results to history, loop                  │                   ║
║  └────────────┬──────────────────────────┬──────────────┘                   ║
║               │                          │                                   ║
║               ▼                          ▼                                   ║
║  ┌────────────────────┐   ┌──────────────────────────────────────────────┐  ║
║  │  OLMo Adapter      │   │  Tool Dispatch                               │  ║
║  │  (adapters/        │   │  ng_set_view  → NeuroglancerState            │  ║
║  │   olmo_adapter.py) │   │  ng_set_lut   → NeuroglancerState            │  ║
║  │                    │   │  ng_add_layer → NeuroglancerState            │  ║
║  │  • format TOOLS    │   │  ng_annotations_add → NeuroglancerState      │  ║
║  │    → <functions>   │   │  data_query_polars  → Polars engine          │  ║
║  │    XML block        │   │  data_ng_annotations_from_data → NG state   │  ║
║  │  • parse response  │   │  ng_state_link → to_url()                   │  ║
║  │    <function_calls> │   │  state_load   → from_url()                  │  ║
║  │    Python syntax   │   └───────────────────┬──────────────────────────┘  ║
║  │  • return OpenAI-  │                       │                              ║
║  │    shaped dicts    │   ┌───────────────────▼──────────────────────────┐  ║
║  └────────┬───────────┘   │  NeuroglancerState  (Python library)         │  ║
║           │               │  • from_url() — parse NG URL hash → dict     │  ║
║           │               │  • mutate layers, position, annotations      │  ║
║           │               │  • to_url()  — serialize back to NG URL      │  ║
║           │               └──────────────────────────────────────────────┘  ║
╚═══════════╬══════════════════════════════════════════════════════════════════╝
            ║ HTTP POST /v1/completions
            ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  vLLM  (port 8002)                                          [T4 GPU phase 1] ║
║  serving: allenai/OLMo-3-7B-Instruct                                        ║
║  • ~14 GB bfloat16  (or ~3.5 GB INT4 for simultaneous operation)            ║
║  • Plain text completions — adapter owns the <functions> prompt formatting  ║
║  • Responds with <function_calls> XML blocks                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

                    ┆  (new NG URL returned to Panel UI → viewer updates)
                    ┆
                    ┆  ─ ─ ─ Optional: Visual Verification Phase ─ ─ ─
                    ┆
                    ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  MolmoWeb FastAPI  (port 8001)                              [T4 GPU phase 2] ║
║  serving: allenai/MolmoWeb-4B  (~8 GB bfloat16)                             ║
║  • Receives task + target NG URL                                             ║
║  • Drives Playwright/Chromium → navigates to URL                            ║
║  • Screenshots WebGL canvas → MolmoWeb-4B → action/observation              ║
║  • Returns: trajectory HTML + visual confirmation                           ║
╚═══════════════════════╦══════════════════════════════════════════════════════╝
                        ║ Playwright CDP
                        ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  Chromium (headless)                                                         ║
║  • Renders Neuroglancer WebGL canvas                                         ║
║  • URL controlled by MolmoWeb goto() actions                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### GPU time-sharing on the T4 (15 GB)

```
Timeline:
──────────────────────────────────────────────────────────────────────────▶
  Phase 1: neuroglancer-chat reasoning
  ┌──────────────────────────────────────┐
  │  OLMo-3-7B-Instruct  (14 GB)        │  ← vLLM loaded
  │  tool call loop (1–10 iterations)   │
  └──────────────────────────────────────┘
                                          Phase 2: visual verification
                                          ┌──────────────────────────────┐
                                          │  MolmoWeb-4B  (8 GB)        │  ← loaded
                                          │  screenshot loop (1–8 steps) │
                                          └──────────────────────────────┘

Alternatively (INT4 OLMo-3-7B, ~3.5 GB):
  ┌──────────────────────────────────┬──────────────────────────────────┐
  │  OLMo-3-7B INT4  (3.5 GB)       │  MolmoWeb-4B  (8 GB)            │
  │  both loaded simultaneously     │  combined ~11.5 GB < 15 GB ✓    │
  └──────────────────────────────────┴──────────────────────────────────┘
```

---

## Missing Components

Everything below needs to exist before the system can run end-to-end.

### Model weights (nothing downloaded yet)

| Model | HF ID | Target path | Size |
|-------|-------|-------------|------|
| MolmoWeb-4B | `allenai/MolmoWeb-4B` | `/scratch/checkpoints/MolmoWeb-4B` | ~8 GB |
| OLMo-3-7B-Instruct | `allenai/OLMo-3-7B-Instruct` | `/scratch/checkpoints/OLMo-3-7B-Instruct` | ~14 GB |

### Python packages (not installed)

| Package | Needed for | Install via |
|---------|-----------|-------------|
| `vllm` | Serving OLMo-3-7B-Instruct | `pip install vllm` (system Python) |
| `zarr` | Reading OME-Zarr data from S3 | `pip install zarr` |
| `s3fs` | S3 filesystem for zarr | `pip install s3fs` |
| `scikit-image` | Nucleus blob detection | `pip install scikit-image` |
| neuroglancer-chat core | Backend + data tools | `cd /code/neuroglancer-chat && uv sync` |
| neuroglancer-chat panel | Panel UI + panel-neuroglancer | `uv sync --group panel` |
| MolmoWeb core | Inference client + FastAPI server | `cd /code/molmoweb && uv sync` |
| Playwright Chromium | Headless browser for MolmoWeb | `uv run playwright install --with-deps chromium` |

### Code to write (neuroglancer-chat fork)

| File | What it does | Status |
|------|-------------|--------|
| `adapters/olmo_adapter.py` | Translates TOOLS → `<functions>` XML; parses `<function_calls>` → OpenAI-shaped dicts | **Not written** |
| `adapters/llm.py` | Add env-var switch: `LLM_BACKEND=olmo` loads `olmo_adapter.py` instead of OpenAI client | **Needs modification** |

This is the only code change required. Everything else in neuroglancer-chat (agent loop, tool dispatch, state management, Panel UI) stays unchanged.

### Infrastructure

| Item | Status | Notes |
|------|--------|-------|
| Git branch for neuroglancer-chat | Not created | `git checkout -b olmo-local` in `/code/neuroglancer-chat` |
| `/scratch/checkpoints/` directory | Does not exist | Created on first `mv` after weight download |
| `OPENAI_API_KEY` | Not needed | Adapter bypasses OpenAI client entirely |

---

## Full Setup Sequence

### Step 1: Install system-level packages

```bash
pip install vllm zarr s3fs scikit-image
```

### Step 2: Install neuroglancer-chat

```bash
cd /code/neuroglancer-chat
uv sync                    # core backend + data tools
uv sync --group panel      # Panel UI + panel-neuroglancer
```

### Step 3: Install MolmoWeb + browser

```bash
cd /code/molmoweb
uv sync
uv run playwright install --with-deps chromium
```

### Step 4: Download model weights

```bash
# MolmoWeb-4B (~8 GB)
cd /code/molmoweb
bash scripts/download_weights.sh allenai/MolmoWeb-4B
mv checkpoints/MolmoWeb-4B /scratch/checkpoints/MolmoWeb-4B

# OLMo-3-7B-Instruct (~14 GB)
huggingface-cli download allenai/OLMo-3-7B-Instruct \
  --local-dir /scratch/checkpoints/OLMo-3-7B-Instruct
```

### Step 5: Create neuroglancer-chat fork branch

```bash
cd /code/neuroglancer-chat
git checkout -b olmo-local
```

Then write `adapters/olmo_adapter.py` and patch `adapters/llm.py` — see next section.

---

## OLMo Adapter: What Needs Writing

This is the critical new code. It lives at:
`/code/neuroglancer-chat/src/neuroglancer_chat/backend/adapters/olmo_adapter.py`

**Responsibilities:**

1. **Format TOOLS → `<functions>` block**
   The existing `TOOLS` list in `llm.py` is a list of OpenAI-format dicts. The adapter renders these as a JSON array inside `<functions>` tags and prepends it to the system message.

2. **Send completion request to vLLM**
   Use plain `/v1/completions` (not `/v1/chat/completions`) — OLMo's format is simpler to control via raw text completion. The adapter builds the full prompt string (system + `<functions>` block + conversation history + `<|im_start|>assistant`) and sends it to vLLM.

3. **Parse `<function_calls>` from response**
   Extract everything between `<function_calls>` and `</function_calls>`. Each line is a Python-style call: `tool_name(arg=val, arg2=val2)`. Parse these into `{"name": ..., "arguments": {...}}` dicts — careful handling needed for nested dicts/strings in args.

4. **Return OpenAI-shaped response dict**
   The existing agent loop in `main.py` expects:
   ```python
   response["choices"][0]["message"]["tool_calls"][i]["function"]["name"]
   response["choices"][0]["message"]["tool_calls"][i]["function"]["arguments"]
   response["choices"][0]["message"]["content"]
   ```
   The adapter must return exactly this structure.

5. **Env-var switch in `llm.py`**
   ```python
   LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")  # "openai" or "olmo"
   if LLM_BACKEND == "olmo":
       from .olmo_adapter import run_chat, run_chat_stream
   ```

---

## Startup Sequence (full system running)

All five processes, assuming bfloat16 sequential GPU use:

```bash
# Terminal 1 — OLMo-3-7B-Instruct via vLLM (phase 1: reasoning)
vllm serve /scratch/checkpoints/OLMo-3-7B-Instruct \
  --port 8002 \
  --max-model-len 8192

# Terminal 2 — neuroglancer-chat backend (fork branch)
cd /code/neuroglancer-chat/src/neuroglancer_chat
export LLM_BACKEND=olmo
export OLMO_URL=http://127.0.0.1:8002
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 3 — neuroglancer-chat Panel UI
cd /code/neuroglancer-chat/src/neuroglancer_chat
export BACKEND=http://127.0.0.1:8000
uv run python -m panel serve panel/panel_app.py \
  --port 8006 --address 127.0.0.1 \
  --allow-websocket-origin=127.0.0.1:8006

# After neuroglancer-chat reasoning phase completes:
# Kill vLLM (Terminal 1), then start MolmoWeb model server

# Terminal 4 — MolmoWeb model server (phase 2: visual verification)
cd /code/molmoweb
export CKPT=/scratch/checkpoints/MolmoWeb-4B
bash scripts/start_server.sh   # PREDICTOR_TYPE=native, PORT=8001

# Terminal 5 — MolmoWeb client (run a task)
cd /code/molmoweb
uv run python - <<'EOF'
from inference import MolmoWeb
client = MolmoWeb(endpoint="http://127.0.0.1:8001", local=True, headless=True)
ng_url = open("/results/annotated_ng_url.txt").read().strip()
traj = client.run(
    f"Navigate to {ng_url}, wait for the volume to render, screenshot it.",
    max_steps=6
)
traj.save_html(query="verification")
EOF
```

---

## End-to-End Demo Query

With all services running, the full loop for "count neurons and annotate":

1. **User** opens Panel UI at `http://localhost:8006`
2. **User** pastes the example NG link and asks: *"Load this link, identify the nuclear channel, and add point annotations for the 50 brightest spots"*
3. **neuroglancer-chat** calls `state_load(link=<url>)` → `NeuroglancerState.from_url()`
4. **OLMo-3-7B-Instruct** decides to call `ng_state_summary` to inspect layers
5. **OLMo** calls `ng_set_lut(layer="l", vmin=200, vmax=800)` to adjust contrast
6. *(In practice: the 50-brightest-spots task requires the CV pipeline from DEMO_PLAN.md to produce coordinates; those coordinates are fed back in as a CSV, then `data_ng_annotations_from_data` ingests them)*
7. **OLMo** calls `ng_state_link` → returns annotated URL
8. **MolmoWeb** (after GPU swap) navigates to URL, screenshots, confirms annotations visible
9. **Result**: annotated URL + screenshot saved to `/results/`

---

## Open Questions / Risks

| Issue | Notes |
|-------|-------|
| OLMo-3-7B-Instruct parallel tool calls | neuroglancer-chat issues 5–8 tool calls per LLM iteration for batch operations. OLMo's `<function_calls>` block may support multi-line calls (one per line) — needs testing |
| Python-syntax arg parsing reliability | `ng_annotations_add(items=[{"type":"point","center":{"x":100,...}}])` — deeply nested JSON inside Python call syntax. May need a fallback parser |
| vLLM `--max-model-len` | OLMo-3-7B-Instruct has 65k context. Cap at 8192 for T4 KV cache headroom |
| panel-neuroglancer S3 data loading | The example link uses `s3://aind-open-data/...` — panel-neuroglancer renders this client-side in the browser via the public NG app URL; no server-side S3 access needed for display |
| MolmoWeb NG rendering wait time | Zarr data loads from S3; may need `noop` steps while tiles stream in |
