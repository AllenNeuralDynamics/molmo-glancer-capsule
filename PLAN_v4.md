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
On compact profile (T4), OLMo is simply not loaded — Molmo handles everything (same as v3).

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

### D5: Screenshot readiness — two-phase: CDP network idle → chunk stall

**Problem:** v3's pixel-polling readiness check (`_wait_for_canvas_stable`) captures
screenshots before data is fully loaded. Pixel hashes can match during brief network
pauses while chunks are still streaming.

**Discovery:** `window.viewer` IS accessible on neuroglancer-demo.appspot.com.
`layerChunkProgressInfo` gives `numVisibleChunksNeeded` vs `numVisibleChunksAvailable`.

**Key finding: `available` will NOT always reach `needed`.** Timeline probe (60s):

```
example_ng_link (4panel layout, single layer):
  0.3s:  156/400  (39%)  ← loading
  2.4s:  280/400  (70%)  ← loading
  4.4s:  319/400  (80%)  ← STALLS HERE PERMANENTLY
  ...60s: 319/400 (80%)  ← never changes
```

The remaining 81 chunks are counted as "needed" but never fetched. Initially
attributed to the 4panel layout, but other 4panel links (thyme_r2r, alignment_loop,
example_r2r, large_ng_link) all reach 100%. The stall is specific to this zarr
dataset — likely its chunking, resolution pyramid, or S3 serving behavior. 5 of 6
preset links use 4panel; only segmentation uses a single-panel layout (yz).

**Why every single-signal approach fails:**

| Approach | Weakness | Evidence |
|----------|----------|----------|
| Pixel hash stall | Network pause → identical frames → false ready | v3 bug reports |
| Chunk count stall | Network pause → identical counts → false ready | Same root cause |
| CDP network idle only | Misses post-download JS decoding | thyme_r2r: network idle at 1s, but only 14% of chunks available. NG fetches zarr in bulk (9 HTTP requests), then decompresses in JS for 4 more seconds |
| Ratio threshold | Magic number — steady-state varies by dataset | example_ng_link: 80%, thyme_r2r: 100%, alignment_loop: 100% |

**NG loading pipeline (observed):**

```
HTTP fetch (fast)           JS decode (slow)           Ready
  9 requests → done ~1s  →  chunks 14% → 60% → 100%  →  ~5s
                            ↑ no network activity here
```

NG fetches zarr data in bulk HTTP responses, then decompresses/decodes chunks in
JavaScript *after* network transfer completes. CDP sees "idle" long before chunks
are ready. Chunk counts see "stall" during network pauses before HTTP is done.

**Solution: two-phase detection.** Each phase covers the other's weakness.

```
Phase 1: CDP network idle (1s)     →  "HTTP transfers are done"
Phase 2: Chunk count stall (1.5s)  →  "JS decoding is done"
```

Phase 1 (CDP) is immune to network pauses — a paused request is still "pending."
Phase 2 (chunk stall) starts only *after* CDP confirms network is idle, so there
are no network pauses to worry about — all remaining chunk increases are from JS
decoding already-downloaded data.

**Validated: 6/6 preset links PASS.** Screenshots visually confirmed correct.

| Link | Chunks | % | Ready | Screenshot |
|------|--------|---|-------|------------|
| alignment_loop | 70/70 | 100% | ~2s | correct |
| example_ng_link | 319/400 | 80% | ~5s | correct (dataset-specific stall) |
| example_r2r_ng_link | 70/70 | 100% | ~2s | correct |
| large_ng_link | 45/45 | 100% | ~2s | correct |
| segmentation | 81/81 | 100% | ~2s | correct |
| thyme_r2r_ng_link | 232/232 | 100% | ~6s | correct |

Validation script: `code/probe_cdp_readiness.py` (delete before merging v4).

```python
def _wait_for_data_loaded(page, timeout_s: float = 60,
                          net_idle_s: float = 1.0, chunk_stable_polls: int = 3,
                          chunk_poll_ms: int = 500):
    """Two-phase readiness: CDP network idle → chunk count stall.

    Phase 1 — CDP network idle: wait until 0 pending HTTP requests for
    `net_idle_s` seconds. This confirms all zarr data has been fetched.
    Immune to network pauses (a paused request stays "pending").

    Phase 2 — Chunk count stall: wait until `available` stops increasing
    for `chunk_stable_polls` consecutive polls. This confirms NG has
    finished decoding all downloaded chunks into GPU textures.
    Safe because Phase 1 guarantees no network activity — any stall is
    genuine processing completion, not a network pause.
    """
    # ── Phase 1: CDP network idle ──────────────────────────────────────
    cdp = page.context.new_cdp_session(page)
    cdp.send("Network.enable")

    pending = set()
    last_activity = time.time()

    def on_request(params):
        nonlocal last_activity
        pending.add(params["requestId"])
        last_activity = time.time()

    def on_finished(params):
        nonlocal last_activity
        pending.discard(params.get("requestId"))
        last_activity = time.time()

    def on_failed(params):
        nonlocal last_activity
        pending.discard(params.get("requestId"))
        last_activity = time.time()

    cdp.on("Network.requestWillBeSent", on_request)
    cdp.on("Network.loadingFinished", on_finished)
    cdp.on("Network.loadingFailed", on_failed)

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(0.25)
        needed, available = _get_chunk_counts(page)
        if available == 0:
            continue
        if len(pending) == 0 and (time.time() - last_activity) >= net_idle_s:
            break
    else:
        cdp.detach()
        print(f"  WARNING: network idle timeout ({len(pending)} pending)")
        return

    cdp.detach()
    pct = available / needed * 100 if needed > 0 else 0
    print(f"  Phase 1 done: network idle, chunks {available}/{needed} ({pct:.0f}%)")

    # ── Phase 2: Chunk count stall ─────────────────────────────────────
    prev_available = -1
    stable_count = 0
    remaining = timeout_s - (time.time() - t0)

    t1 = time.time()
    while time.time() - t1 < remaining:
        needed, available = _get_chunk_counts(page)
        if available == prev_available:
            stable_count += 1
            if stable_count >= chunk_stable_polls:
                pct = available / needed * 100 if needed > 0 else 0
                print(f"  Phase 2 done: chunks stable {available}/{needed} ({pct:.0f}%)")
                return
        else:
            stable_count = 0
        prev_available = available
        time.sleep(chunk_poll_ms / 1000)

    print(f"  WARNING: chunk stability timeout ({available}/{needed})")


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

**Why two-phase > any single signal:**
- Phase 1 alone declares thyme_r2r ready at 14% — misses JS decoding
- Phase 2 alone is fooled by network pauses during active loading
- Together: Phase 1 confirms "no more data coming" → Phase 2 safely detects
  "done processing" without false stalls from network interruptions

**Three-gate capture flow:**
1. `_wait_for_data_loaded(page)` — two-phase: CDP idle → chunk stall
2. One final pixel stability check (2 frames, 200ms) — catches WebGL rendering lag
   after chunks arrive but before canvas redraws
3. `_canvas_has_data()` sanity check — abort if canvas is still blank

**For scan frames (async path):** Same two-phase approach via async Playwright.
Use `net_idle_s=0.5, chunk_stable_polls=2` since adjacent frames share ~90%
chunks — less data per frame, faster turnaround.

**File: `code/visual_capture.py`** — replace `_wait_for_canvas_stable` with
`_wait_for_data_loaded` + final pixel gate.

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

### 1. `code/gpu_config.py` — Model manager with swap support

Add:
- `OLMO_CHECKPOINT` path constant (`/scratch/checkpoints/Olmo-3.1-32B-Think`)
- `ModelManager` class:
  - `load_molmo()` → load Molmo2 to GPU, returns (model, processor)
  - `load_olmo()` → load OLMo 3.1 32B INT8 from disk to GPU, returns (model, tokenizer)
  - `swap_to_molmo()` → delete OLMo from GPU, move Molmo from CPU to GPU
  - `swap_to_olmo()` → move Molmo to CPU, load OLMo from disk to GPU (INT8)
  - `active_model` property → which model is currently on GPU
  - VRAM reporting after each swap
- Only activate on "full" profile. On "compact" (T4): no OLMo, no swaps.

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
  - On "compact" profile: skip swaps, use Molmo for everything (same as v3)
- `parse_action()` → extend to parse `{"actions": [...]}` batch format alongside
  single `{"action": ...}` for backward compat
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

### 6. `code/visual_capture.py` — Two-phase readiness (D5) ✅ validated

Replace pixel-polling readiness with two-phase CDP + chunk stall detection:
- Add `_get_chunk_counts(page)` — JS eval returning `(needed, available)`
- Add `_wait_for_data_loaded(page)` — Phase 1: CDP network idle (1s), Phase 2: chunk
  count stall (3 × 500ms). Validated on all 6 preset links with visual confirmation.
- Add `_async_wait_for_data_loaded(page)` — async version for scan frames
  (`net_idle_s=0.5, chunk_stable_polls=2`)
- Update `capture_screenshot()` — use `_wait_for_data_loaded` + final pixel gate
- Update `execute_scan()` / `_run_sequential()` — use async version for scan frames
- Keep `_canvas_has_data` as sanity check (abort if canvas still blank after readiness)
- Remove `_wait_for_canvas_stable` and `_async_wait_for_canvas_stable`

### 7. `REFERENCES.md` — Add OLMo 3.1 sources ✅

Done — OLMo 3.1 32B Think section added with model cards, blog posts, GGUF sources,
and VRAM estimates table.

### 8. Probe scripts — Delete before merging v4

Diagnostic scripts not needed in production:
- `code/probe_ng_viewer.py`
- `code/probe_chunk_timeline.py`
- `code/probe_cdp_readiness.py`

## Implementation Order

1. ~~**REFERENCES.md** — add sources~~ ✅
2. ~~**D5 validation** — two-phase readiness probe on all 6 links~~ ✅
3. **visual_capture.py** — implement two-phase readiness (can land independently of OLMo)
4. **_download_weights.sh** — add OLMo download
5. **gpu_config.py** — `ModelManager` with asymmetric swap logic
6. **molmo_glancer.py** — `ask_text_olmo()` + agent loop integration
7. **Test on T4** — verify compact profile still works (no OLMo, same as v3)
8. **Test on L40S** — verify swap cycle, VRAM usage, generation quality
9. **Delete probe scripts** — clean up before merge

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
