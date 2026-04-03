# v2 Plan: Molmo2-Powered Neuroglancer Question Answering

## Goal

A system that takes an open-ended question about 3D data in Neuroglancer and iteratively plans views, screenshots them, visually interprets them, and synthesizes a confident answer — all driven by a single unified model.

---

## Architecture

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
+----------------+  +-------------------+
| NeuroglancerState |  | Molmo2-O-7B       |
| (URL builder)     |  | (image + text)    |
+----------------+  +-------------------+
    |                       ^
    | NG URLs               | screenshots
    v                       |
+---------------------------+
|  Playwright (headless)    |
|  navigate + screenshot    |
+---------------------------+
```

### Single Model: Molmo2-O-7B

- Built on OLMo3-7B-Instruct backbone + SigLIP 2 vision encoder
- Handles BOTH text-only orchestration AND visual interpretation
- Loaded once, stays resident for the entire run
- 8-bit quantization (~8 GB) fits comfortably on T4 (15 GB VRAM)
- HuggingFace transformers inference (`AutoModelForImageTextToText`)

### Loop Steps

1. **Plan** (text-only call) — given the question + findings so far, output a list of view specs (position, zoom, layout, which layers visible)
2. **Generate URLs** (NeuroglancerState) — deterministic, no LLM needed
3. **Screenshot** (Playwright) — navigate to each URL, wait for data load, capture PNG
4. **Interpret** (image+text call) — pass screenshot + domain-specific prompt with context on what to look for
5. **Decide** (text-only call) — enough information to answer? If not, loop to step 1. If yes, synthesize final answer.

Steps 1 and 5 are narrow focused prompts to keep output quality high from a 7B model.

---

## Editable Repos & What Changes

### This workspace (`/root/capsule/`)

New files:
- `v2_plan.md` — this file
- `code/molmo_glancer.py` — **main v2 pipeline script** (replaces `ng_explore.py` and `run_capsule.py`)
- `code/run_v2` — shell entry point
- `code/_download_weights_v2.sh` — downloads Molmo2-O-7B checkpoint

Updated files:
- `CLAUDE.md` — rewrite for v2 architecture
- `code/_dev_startup2.sh` — add `transformers>=4.57`, `bitsandbytes`, `molmo_utils`, `decord2`; remove vllm if no longer needed

### `code/lib/neuroglancer-chat/`

Used as a **library only** — we import `NeuroglancerState` directly, not the FastAPI backend or Panel UI.

Changes needed:
- Possibly none — `NeuroglancerState` is already a clean standalone class
- If imports pull in heavy deps (FastAPI, Panel, OpenAI), add a thin import shim or import only `neuroglancer_chat.backend.tools.neuroglancer_state`

### `code/lib/molmoweb/`

**Not used in v2.** MolmoWeb's action loop, web agent, FastAPI server — all replaced by direct Molmo2 inference + Playwright screenshots. No changes needed; just don't import it.

### `code/lib/OLMo-core/`

**Not used in v2.** Molmo2-O-7B loads via HuggingFace transformers, not OLMo-core's native loader. No changes needed.

---

## v1 Cleanup (safe to delete on v2 branch)

### v1 pipeline files (replaced)
- `code/ng_explore.py` — v1 MolmoWeb exploration script
- `code/run_ng_explore` — v1 shell entry point
- `code/cleanup_ng_explore` — v1 cleanup script
- `code/run_capsule.py` — v1 grid-search pipeline
- `code/run` — v1 shell entry point for run_capsule
- `code/cleanup` — v1 cleanup script
- `code/ng_link_utils.py` — v1 NG link generation (replaced by NeuroglancerState)

### v1 planning/investigation docs (superseded by this plan)
- `IMPLEMENTATION_PLAN.md`
- `LOCAL_STACK_PLAN.md`
- `DEMO_PLAN.md`
- `OLMO_INTEGRATION.md`
- `investigation/ng_navigation_overhaul.md`
- `environment_build_setup_notes.md`

### v1 example/data files (review, likely delete)
- `example_ng_link_generation_script.py`
- `example_dockerfile`
- `ground_truth_ng_link_z_16.json`

### Keep
- `CLAUDE.md` — rewrite for v2
- `SETUP_NOTES.md` — still relevant env issues
- `MEMORY_*.md` — session memory, still useful
- `README.md` — rewrite for v2
- `example_ng_link.txt`, `example_r2r_ng_link.txt`, `thyme_r2r_ng_link.txt` — test data for the pipeline
- `code/_dev_startup2.sh`, `code/_download_weights.sh` — update for v2 deps
- `code/lib/neuroglancer-chat/` — used as library
- `code/lib/molmoweb/` — not imported but no need to delete (separate git repo)
- `code/lib/OLMo-core/` — same, leave in place
- `environment/` — Dockerfile + postInstall, update as needed

---

## Key Design Decisions

**Focused prompts over monolithic prompts.** Each call to Molmo2 has a single clear job (plan, interpret, decide, synthesize). This keeps a 7B model reliable.

**Domain context injection.** Every visual interpretation call includes a preamble describing the microscopy domain: what colors mean, what good/bad alignment looks like, what cell bodies look like, expected spatial scales. Turns an open-domain VQA problem into guided pattern matching.

**NeuroglancerState as the navigation layer.** All positioning, zoom, layout, and layer visibility is done via deterministic URL construction. No browser interaction beyond navigate-and-screenshot.

**Configurable per-dataset.** Each dataset/question gets a config specifying: NG link (or base state), layer descriptions, domain context preamble, spatial extent, and default zoom levels. This decouples the pipeline from any single dataset.

---

## Dependencies

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

## Milestones

1. **Model loads** — download Molmo2-O-7B, load in 8-bit, run a basic VQA test on a sample image
2. **Screenshot pipeline** — Playwright navigates to NG URL, waits, captures PNG, passes to Molmo2
3. **NeuroglancerState integration** — generate view URLs from position/zoom/layer specs
4. **Single-pass QA** — one plan → screenshot → interpret → answer cycle end-to-end
5. **Full loop** — multi-iteration with decide step, tested on registration dataset
6. **Evaluation** — run on example_ng_link, example_r2r_ng_link, thyme_r2r_ng_link with real questions
