# v4 Plan — OLMo 3.1 32B Think as Reasoning Model

## Goal

Add OLMo 3.1 32B Think as a dedicated text-reasoning model that swaps in/out of GPU
with Molmo2-O-7B. Molmo handles all vision tasks (screenshots, scans, pointing);
the 32B Think model handles planning, decision-making, reasoning, and final synthesis.

## Why

Molmo2's backbone is OLMo3-7B-Instruct. Every `ask_text` call (planning, decisions,
reasoning, forced answers) uses a 7B model for pure text — no vision needed. Swapping
in a 32B reasoning model for those calls gives massive quality gains:

| Benchmark       | OLMo3 7B Instruct (current) | OLMo 3.1 32B Think |
|-----------------|:---------------------------:|:-------------------:|
| MMLU            | 69.1                        | 86.4                |
| MATH            | 87.3                        | 96.2                |
| AIME 2025       | 32.5                        | 78.1                |
| HumanEvalPlus   | 77.2                        | 91.5                |
| IFEval          | 85.6                        | 93.8                |

## Target Machine

| Resource  | Spec         |
|-----------|--------------|
| GPU       | 1× L40S      |
| GPU VRAM  | 45 GB        |
| CPU cores | 4            |
| CPU RAM   | 32 GB        |

## GPU Budget

| Component           | VRAM       | Notes                      |
|---------------------|------------|----------------------------|
| Molmo2-O-7B fp16    | ~14.5 GB   | Vision + image generation  |
| OLMo 3.1 32B INT8   | ~34 GB     | Text reasoning (swapped)   |
| KV cache + overhead | ~8-11 GB   | Depends on context length  |

**Strategy: swap, not side-by-side.** Only one model on GPU at a time. Each model
gets nearly the full 45 GB.

## Decisions

### D1: Quantization — bitsandbytes INT8

**Choice:** `load_in_8bit=True` via bitsandbytes.

**Why INT8 over alternatives:**

| Method         | VRAM   | Benchmark loss | Pre-quant weights? | Extra deps?          |
|----------------|--------|:--------------:|:------------------:|----------------------|
| BF16           | ~64 GB | baseline       | No                 | None                 |
| **bnb INT8**   | ~34 GB | <1%            | No (quantize on load) | Already installed |
| bnb NF4        | ~20 GB | 1–3%           | No                 | Already installed    |
| GPTQ           | ~20 GB | 1–2%           | Yes (separate DL)  | auto-gptq            |
| GGUF Q8_0      | ~34 GB | <1%            | Yes (separate DL)  | llama-cpp-python     |

- INT8 is near-lossless (<1% degradation on MMLU/MATH)
- Zero new dependencies — bitsandbytes + accelerate already in `_dev_startup.sh`
- No separate quantized weights — loads the BF16 safetensors from disk, quantizes per-layer on the fly
- 34 GB fits in 45 GB with ~11 GB headroom for KV cache

**No fallback quantization.** No Q5/Q4 side-by-side mode. INT8 swap or nothing.
v4 targets L40S exclusively — no T4/compact fallback (see D7).

### D2: Swap strategy — asymmetric (CPU for Molmo, disk for OLMo)

**Problem:** 32 GB CPU RAM can't hold both models.
- Molmo fp16 on CPU: ~14.5 GB — fits
- OLMo INT8 on CPU: ~34 GB — does NOT fit

**Solution: asymmetric swap.**

| Direction       | Method                                             | Est. time  |
|-----------------|----------------------------------------------------|------------|
| Molmo → CPU     | `model.to("cpu")` + `torch.cuda.empty_cache()`    | ~5s        |
| OLMo → GPU      | Load from /scratch SSD, INT8 quantize on the fly   | ~30-60s    |
| OLMo → delete   | `del model` + `torch.cuda.empty_cache()`           | instant    |
| Molmo → GPU     | `model.to("cuda")`                                 | ~5s        |

Molmo persists on CPU between swaps (fast round-trip). OLMo is loaded fresh from
disk each time and deleted when done (slower, but only happens once per batch).

**Full swap cycle (Molmo→OLMo→Molmo):** ~40-70s total.

With batched vision/text phases (max 2 swaps per iteration), a 10-iteration run
adds ~7-12 min of swap overhead. Acceptable given the quality jump.

**CPU RAM budget during OLMo phase:**
- Molmo on CPU: ~14.5 GB
- OLMo loading overhead (one shard at a time): ~2-4 GB transient
- OS + Python + Playwright: ~2-3 GB
- Total peak: ~19-22 GB of 32 GB — safe

### D3: Think token handling

OLMo Think models emit `<think>...</think>` reasoning chains before the answer.
- Strip `<think>` blocks from the returned text (the agent loop parses JSON actions)
- Log full output (including think blocks) to transcript.md for debugging
- Token budget: 2048 for decisions, 4096 for synthesis/forced-answer
- Think tokens cost generation time but improve answer quality — that's the point

### D4: Context window

OLMo 3.1 32B: max 32K tokens natively. Molmo2: 65K with YaRN RoPE.
Track OLMo context budget separately. Keep text prompts under 32K.
The `max_context_tokens` in GPU profile stays at 55K for Molmo (vision);
add `max_olmo_context_tokens: 32000` for text calls.

### D5: Screenshot readiness — chunk-stability poll + JS canvas capture ✅ validated

**Problem:** v3's pixel-polling readiness check (`_wait_for_canvas_stable`) captures
screenshots before data is fully loaded. Pixel hashes can match during brief network
pauses while chunks are still streaming.

**Discovery:** `window.viewer` IS accessible on neuroglancer-demo.appspot.com.
`layerChunkProgressInfo` gives `numVisibleChunksNeeded` vs `numVisibleChunksAvailable`.
This exposes exact readiness state directly — no need to infer from network traffic.

**Key finding: `available` will NOT always reach `needed`.** Timeline probe (60s):

```
example_ng_link (4panel layout, single layer):
  0.3s:  156/400  (39%)  ← loading
  2.4s:  280/400  (70%)  ← loading
  4.4s:  319/400  (80%)  ← STALLS HERE PERMANENTLY
  ...60s: 319/400 (80%)  ← never changes
```

The remaining 81 chunks are counted as "needed" but never fetched — dataset-specific
behaviour (chunking, resolution pyramid, or S3 serving). Other 4panel links all reach
100%. The readiness check must handle plateau < 100% as a valid ready state.

**Solution: chunk-stability poll.** Poll `(available, needed)` until the pair is
unchanged for N consecutive reads. No CDP required — Neuroglancer already exposes
readiness; there is nothing to infer from network events.

- Handles plateau: 319/400 stable for 4 polls → ready (no 100% requirement)
- Handles cache: data loaded with 0 HTTP requests → stable immediately, no
  `completed_count > 0` gate to deadlock on
- Handles streaming: counts keep changing → stable counter resets, waits longer
- No event wiring, no `last_activity` tracking, no `pending` sets

```python
def _wait_for_data_loaded(page, timeout_s=60.0, stable_polls=4, poll_s=0.5):
    """Wait until chunk counts (available, needed) are stable for stable_polls
    consecutive reads. Handles both 100%-loaded and plateau cases."""
    t0 = time.time()
    prev = (-1, -1)
    stable_count = 0

    while time.time() - t0 < timeout_s:
        time.sleep(poll_s)
        needed, available = _get_chunk_counts(page)
        cur = (available, needed)
        if cur == prev and needed > 0:
            stable_count += 1
            if stable_count >= stable_polls:
                return  # ready
        else:
            stable_count = 0
        prev = cur


def _get_chunk_counts(page) -> tuple[int, int]:
    """Query total (needed, available) across all visible render layers."""
    result = page.evaluate("""(() => {
        const v = window.viewer;
        if (!v || !v.layerManager) return null;
        let needed = 0, available = 0;
        for (const ml of v.layerManager.managedLayers) {
            if (!ml.layer || !ml.layer.renderLayers) continue;
            for (const rl of ml.layer.renderLayers) {
                const info = rl.layerChunkProgressInfo;
                if (info && info.numVisibleChunksNeeded > 0) {
                    needed += info.numVisibleChunksNeeded;
                    available += info.numVisibleChunksAvailable;
                }
            }
        }
        return {needed, available};
    })()""")
    if result is None:
        return (0, 0)
    return (result["needed"], result["available"])
```

**Screenshot capture: JS `canvas.toDataURL()` with `preserveDrawingBuffer` patch.**

Playwright's `page.screenshot()` and `locator.screenshot()` block on internal page
stabilisation (font loading, animation settling) and timeout even when the canvas is
fully rendered. The root cause is Playwright's `wait_for_selector` / font-ready wait.

WebGL canvases return black from `toDataURL()` by default because `preserveDrawingBuffer`
is `false` — the framebuffer is cleared immediately after compositing. Fix: patch
`HTMLCanvasElement.prototype.getContext` via `add_init_script` *before* navigation so
Neuroglancer's WebGL context is created with `preserveDrawingBuffer: true`. Then
`toDataURL()` reads the live framebuffer directly, bypassing all Playwright machinery.

```python
# Must be called before page.goto()
page.add_init_script("""
    const _orig = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(type, attrs) {
        if (type === 'webgl' || type === 'webgl2') {
            attrs = Object.assign({}, attrs || {}, {preserveDrawingBuffer: true});
        }
        return _orig.call(this, type, attrs);
    };
""")

# After _wait_for_data_loaded + CSS hide:
data_url = page.evaluate("""() => {
    const canvas = document.querySelector('canvas');
    return canvas ? canvas.toDataURL('image/png') : null;
}""")
png_bytes = base64.b64decode(data_url.split(',', 1)[1])
img = Image.open(BytesIO(png_bytes)).convert('RGB')
```

**Validated: 9/9 links PASS.** Screenshots visually confirmed correct.

| Link | Chunks | % | Ready |
|------|--------|---|-------|
| alignment_loop | 70/70 | 100% | 3.1s |
| ccf_cells | 362/362 | 100% | 52.8s |
| ccf_ng | 596/596 | 100% | 6.9s |
| example_ng_link | 319/400 | 80% | 7.6s (dataset plateau) |
| example_r2r_ng_link | 70/70 | 100% | 3.2s |
| large_ng_link | 45/45 | 100% | 2.5s |
| segmentation | 81/81 | 100% | 3.1s |
| smartspim_ng | 162/162 | 100% | 3.9s |
| thyme_r2r_ng_link | 232/232 | 100% | 6.5s |

Validation script: `code/_data_ready_simple.py` (keep as diagnostic tool).

**Capture flow:**
1. `add_init_script` to patch `getContext` with `preserveDrawingBuffer: true`
2. `page.goto(ng_link)` — patch is in place before Neuroglancer initialises WebGL
3. Wait for `needed > 0` (viewer warmup, up to 10s)
4. `_wait_for_data_loaded(page)` — chunk-stability poll (4 × 0.5s)
5. `page.add_style_tag(NG_HIDE_CSS)` — hide UI chrome
6. `canvas.toDataURL('image/png')` via `page.evaluate` — direct framebuffer read
7. `_canvas_has_data()` sanity check — abort if blank

**File: `code/visual_capture.py`** — replace `_wait_for_canvas_stable` with
`_wait_for_data_loaded` + `toDataURL` capture; add `preserveDrawingBuffer` init script.

### D6: Multi-view planning — amortize swaps across views

In v3, each iteration plans one action, captures one view, interprets it, then reasons.
That's 1:1 — one swap cycle per view. Expensive with ~40-70s swap overhead.

**v4 changes the ratio: OLMo plans multiple views per iteration, Molmo interprets
each one, then OLMo reasons over all findings at once.**

Example iteration:
1. **OLMo plans** → returns 3 actions: `[screenshot(z=100, xy), screenshot(z=200, xy), scan(z=100..200, xz)]`
2. **Swap to Molmo** → capture + interpret all 3 (no swap between them)
3. **Swap to OLMo** → receives 3 findings, reasons over the batch, decides next step or answers

This cuts swaps from 2×N (where N = views per iteration) down to **2 per iteration, fixed**,
regardless of how many views OLMo requests. A 10-view investigation that would have taken
10 swap cycles now takes 1.

**Plan action format change:**

v3 decision output (single action):
```json
{"action": "screenshot", "view": {...}, "prompt": "..."}
```

v4 decision output (action batch):
```json
{"actions": [
    {"action": "screenshot", "view": {...}, "prompt": "..."},
    {"action": "screenshot", "view": {...}, "prompt": "..."},
    {"action": "scan", "scan": {...}, "prompt": "..."}
],
 "reasoning": "I want to compare these three regions because..."}
```

**Guardrails:**
- Max actions per batch: `max_actions_per_plan` config key (default 4, cap at 6)
- Total VRAM per batch must fit — large scans (many frames) eat more VRAM than screenshots
- `answer` and `reason` actions are still single (they don't involve vision)
- If OLMo returns a single action, treat it as a batch of 1 — backward compatible

### D7: Drop T4 / compact profile support

**Decision:** v4 targets L40S (45 GB) exclusively. Remove the compact (T4 / 15 GB)
profile and all code paths that branch on it.

**Why:**
- v4's core value is the OLMo 32B reasoning model. On T4 (15 GB) OLMo can't load at
  all — the capsule would fall back to Molmo-only, which is just v3. Maintaining a
  "v4 that runs like v3" doubles the test surface for zero user benefit.
- 4-bit NF4 quantization of Molmo (the compact path) masks vision quality issues that
  aren't relevant to the L40S target. Debugging two quantization regimes wastes time.
- The compact profile adds branching in gpu_config, visual_capture, and molmo_glancer
  (image downscaling, max_image_side guards, profile detection, T4-specific Chromium
  args). Removing it simplifies every file and the test matrix.
- The `max_actions_per_plan=1` fallback for compact profile adds a parallel prompting
  path (single-action schema + concrete JSON examples) that diverges from the batched
  v4 design. One prompting path is easier to iterate on.

**What to remove:**
- `GPU_PROFILES["compact"]` dict and all `"4bit"` / `BitsAndBytesConfig` code in
  `gpu_config.py`
- `detect_gpu_profile()` — replace with a simple VRAM assertion (≥40 GB or abort)
- `max_image_side` config key and all image-downscaling guards in `molmo_glancer.py`
  (`ask_vision`, `ask_vision_pointing`) and `visual_capture.py` (`capture_screenshot`,
  `execute_scan`)
- T4-specific Chromium args (`_CHROMIUM_ARGS_BASE` vs `_CHROMIUM_ARGS_GPU` branching)
  — always use GPU-accelerated rendering
- References to "compact profile" in comments, docstrings, and `_dev_startup.sh` /
  `_download_weights.sh`

**What remains:**
- A single `CONFIG` dict (renamed from `GPU_PROFILES["full"]`) with the L40S parameters
- `load_model()` always loads Molmo fp16, no quantization
- `ModelManager` always enables OLMo swap — no "skip swaps" branch
- Chromium always launches with `--use-gl=egl`

## Architecture

### Current flow (v3)
```
main() → load_model(Molmo2) → run_agent(model, processor, config, ...)
  Phase 1: ask_vision(Molmo2)     ← vision
  Phase 2 loop (1 view per iteration):
    ask_text(Molmo2)              ← plan 1 action (7B text)
    ask_vision/ask_scan(Molmo2)   ← capture + interpret 1 view
    ask_text(Molmo2)              ← reason over 1 finding (7B text)
  Synthesis: ask_text(Molmo2)     ← final answer (7B text)
```

### Proposed flow (v4)
```
main() → ModelManager(molmo, olmo_path) → run_agent(manager, ...)

  Phase 1 — First Look (Molmo on GPU):
    capture + interpret initial view

  Phase 2 loop (multiple views per iteration, 2 swaps per iteration):
    ┌─ OLMo phase (swap_to_olmo) ─────────────────────────────────┐
    │  ask_text(OLMo-32B): plan N actions from accumulated history │
    │  → returns [{action, view, prompt}, ...]                     │
    └──────────────────────────────────────────────────────────────┘
    ┌─ Molmo phase (swap_to_molmo) ────────────────────────────────┐
    │  for each planned action:                                     │
    │    capture screenshot/scan                                    │
    │    ask_vision(Molmo2): interpret → finding                   │
    │  collect [finding_1, finding_2, ..., finding_N]              │
    └──────────────────────────────────────────────────────────────┘
    ┌─ OLMo phase (swap_to_olmo) ─────────────────────────────────┐
    │  ask_text(OLMo-32B): reason over N findings                  │
    │  → decide: plan more views, or answer                        │
    └──────────────────────────────────────────────────────────────┘

  Synthesis (OLMo still on GPU):
    ask_text(OLMo-32B) → final answer (32B reasoning)
```

### Swap budget

| Scenario             | Swaps per iter | 5-iter run | 10-iter run |
|----------------------|:--------------:|:----------:|:-----------:|
| v3 (no swap)         | 0              | 0          | 0           |
| v4 naive (per-call)  | ~4             | ~20        | ~40         |
| **v4 batched**       | **2**          | **10**     | **20**      |

At ~40-70s per swap cycle, a 5-iteration v4 run adds ~3.5-6 min swap overhead.
Each iteration covers more ground (multiple views), so total iterations should drop
compared to v3's one-view-at-a-time approach.

## Files to Change

### 1. `code/gpu_config.py` — Rewrite: single profile + ModelManager

Remove:
- `GPU_PROFILES` dict (both "compact" and "full" entries)
- `detect_gpu_profile()` function
- All 4-bit NF4 / `BitsAndBytesConfig` code paths
- `max_image_side` config key
- `_CHROMIUM_ARGS_BASE` vs `_CHROMIUM_ARGS_GPU` branching in `visual_capture.py`

Replace with:
- Single `CONFIG` dict with L40S parameters (fp16, no downscale, 20 iterations, etc.)
- `assert_gpu()` — verify ≥40 GB VRAM or abort with clear error
- `OLMO_CHECKPOINT` path constant (`/scratch/checkpoints/Olmo-3.1-32B-Think`)
- `ModelManager` class:
  - `load_molmo()` → load Molmo2 fp16 to GPU, returns (model, processor)
  - `load_olmo()` → load OLMo 3.1 32B INT8 from disk to GPU, returns (model, tokenizer)
  - `swap_to_molmo()` → delete OLMo from GPU, move Molmo from CPU to GPU
  - `swap_to_olmo()` → move Molmo to CPU, load OLMo from disk to GPU (INT8)
  - `active_model` property → which model is currently on GPU
  - VRAM reporting after each swap
- OLMo swap always enabled — no conditional branching

### 2. `code/molmo_glancer.py` — Agent loop integration

Modify:
- `ask_text_olmo()` → new function for OLMo generation
  - Standard `AutoModelForCausalLM` + `AutoTokenizer`
  - Chat template: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
  - Strip `<think>...</think>` from returned text, log full output to transcript
- `run_agent()` → accept `ModelManager` instead of raw `(model, processor)`
  - Restructure iteration loop into three phases:
    1. **OLMo plan phase** — `swap_to_olmo()`, call `ask_text_olmo()` to plan N actions
    2. **Molmo vision phase** — `swap_to_molmo()`, execute all N actions (capture + interpret each)
    3. **OLMo reason phase** — `swap_to_olmo()`, pass all N findings, decide next step or answer
- Remove all `max_image_side` downscaling guards from `ask_vision`, `ask_vision_pointing`
- `parse_action()` → extend to parse `{"actions": [...]}` batch format alongside
  single `{"action": ...}` (a single action is treated as a batch of 1)
- Decision prompt → instruct OLMo to return an `actions` array (1-4 views per batch);
  add `max_actions_per_plan` to config
- `main()` → construct `ModelManager`, pass to `run_agent()`

### 3. `code/_download_weights.sh` — Download OLMo weights

Add second download block:
```bash
OLMO_DEST=/scratch/checkpoints/Olmo-3.1-32B-Think
huggingface-cli download allenai/Olmo-3.1-32B-Think \
    --local-dir "$OLMO_DEST"
```

~64 GB on disk (BF16 safetensors). Quantization happens at load time via
bitsandbytes, not on disk.

### 4. `code/_dev_startup.sh` — No changes expected

`transformers`, `bitsandbytes`, `accelerate` are already installed.
OLMo 3.1 uses standard HuggingFace `AutoModelForCausalLM` — no extra deps.
Verify `ai2-olmo-core` version compatibility (currently 2.4.0 in Dockerfile).

### 5. `environment/Dockerfile` — Possibly bump `ai2-olmo-core`

Check if OLMo 3.1 32B needs a newer version than 2.4.0. If so, bump in Dockerfile.
If 3.1 loads purely via transformers (likely), this may not be needed at all.

### 6. `code/visual_capture.py` — Chunk-stability readiness + JS canvas capture (D5) ✅ validated

Replace pixel-polling readiness and Playwright screenshot with chunk-stability poll
+ direct WebGL framebuffer read:
- Add `_get_chunk_counts(page)` — JS eval returning `(needed, available)`
- Add `_wait_for_data_loaded(page)` — poll `(available, needed)` until stable for
  4 consecutive 0.5s reads. Validated on all 9 preset links with visual confirmation.
- Add `_async_wait_for_data_loaded(page)` — async version for scan frames
  (`stable_polls=2` since adjacent frames share ~90% of chunks)
- Add `preserveDrawingBuffer` init script to every new page (before `goto`) so
  `canvas.toDataURL()` returns real content instead of black
- Update `capture_screenshot()` — use `_wait_for_data_loaded` + `toDataURL` capture
- Update `execute_scan()` / `_run_sequential()` — use async version for scan frames
- Keep `_canvas_has_data` as sanity check (abort if canvas still blank after readiness)
- Remove `_wait_for_canvas_stable`, `_async_wait_for_canvas_stable`, and all
  `page.screenshot()` / `locator.screenshot()` calls

### 7. `REFERENCES.md` — Add OLMo 3.1 sources ✅

Done — OLMo 3.1 32B Think section added with model cards, blog posts, GGUF sources,
and VRAM estimates table.

### 8. Probe scripts — Delete before merging v4

Diagnostic scripts not needed in production:
- `code/probe_ng_viewer.py`
- `code/probe_chunk_timeline.py`
- `code/_data_ready_check.py` (two-phase CDP probe — superseded)
- `code/_data_ready_simple.py` (chunk-stability probe — keep as diagnostic tool, do not ship in prod)

## Implementation Order

1. ~~**REFERENCES.md** — add sources~~ ✅
2. ~~**D5 validation** — chunk-stability poll + `preserveDrawingBuffer`/`toDataURL` capture, 9/9 links PASS~~ ✅
3. **gpu_config.py** — strip compact profile, single CONFIG dict, `assert_gpu()` (D7)
4. **visual_capture.py** — implement two-phase readiness + remove T4 branches (D5 + D7)
5. **molmo_glancer.py** — remove `max_image_side` guards + T4 comments (D7)
6. **_download_weights.sh** — add OLMo download, remove T4 references
7. **gpu_config.py** — add `ModelManager` with asymmetric swap logic
8. **molmo_glancer.py** — `ask_text_olmo()` + agent loop integration + v4 prompting
9. **Test on L40S** — verify swap cycle, VRAM usage, generation quality
10. **Delete probe scripts** — clean up before merge

## Open Questions

1. **Think token budget:** OLMo Think models can generate long `<think>` chains.
   Need to set a reasonable `max_new_tokens` — maybe 2048 for decisions, 4096 for
   synthesis. The thinking tokens are "free" in terms of quality but cost time.

2. **Swap latency in practice:** Estimated ~40-70s per full swap cycle. Need to
   measure actual load time for 64 GB BF16 → INT8 from /scratch SSD. If too slow,
   could explore pre-quantized INT8 checkpoints to skip on-the-fly quantization.

3. **OLMo 3.1 + ai2-olmo-core compatibility:** Verify whether OLMo 3.1 32B loads
   purely via transformers `AutoModelForCausalLM` or needs `ai2-olmo-core`. If the
   latter, check version requirements against the 2.4.0 in Dockerfile.
