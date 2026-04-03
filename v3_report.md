# molmo-glancer v3 — MVP Report

## Executive Summary

molmo-glancer v3 is an autonomous visual analysis system for 3D volumetric neuroscience data. Given a Neuroglancer link and a natural-language question, it runs a free-form agent loop: a single vision-language model (Molmo2-O-7B) decides what to look at, the system captures it via headless browser, the model interprets what it sees, and the cycle repeats until the model is confident in its answer.

The system runs on Code Ocean with auto-detected GPU profiles: 4-bit quantized on T4 (dev) or full fp16 on L40S (production). It produces a final text answer plus all visual artifacts (screenshots, scan videos, per-iteration findings).

---

## Model: Molmo2-O-7B

### Why Molmo2

Molmo2-O-7B is a 7.76B-parameter multimodal model from the Allen Institute for AI (Ai2). It was selected for this project because it uniquely combines:

1. **Image, video, and pointing in one model.** Most open VLMs handle images only. Molmo2 natively processes video (up to 384 frames at 81 tokens/frame via 3x3 pooling) and outputs structured pointing coordinates — critical for object counting in volumetric data.

2. **Fully open.** Molmo2 is released under Apache 2.0 with open weights, open training data, and open training code. Ai2 describes this as an **"open weight and data"** model, part of their broader **"open science"** commitment. The training datasets are published on HuggingFace (`allenai/Molmo2-Cap`, `allenai/Molmo2-VideoCapQA`, `allenai/Molmo2-VideoPoint`, etc.). This distinguishes Molmo2 from "open-weight" models (like Llama) that release weights but not training data.

3. **Strong benchmarks at 7B scale.** Molmo2-O-7B scores 59.7 average across 15 academic benchmarks, competitive with much larger proprietary models and outperforming other open models on short video, counting, and captioning tasks.

### Architecture

| Component | Detail |
|---|---|
| Vision encoder | SigLIP 2 (google/siglip2-so400m-patch14-384), ~400M params |
| Language backbone | OLMo3-7B-Instruct, 32 layers, full MHA (32 KV heads) |
| Image processing | 378x378 crops, 2x2 pooling, ~169-196 tokens/crop, max 8 crops (24 for detail) |
| Video processing | 378x378 frames, 3x3 pooling, 81 tokens/frame, max 384 frames |
| Context window | 65,536 tokens (YaRN RoPE) |
| KV cache | 0.5 MB/token |
| Chat template | `<\|im_start\|>user/assistant` — no system role supported |

### Inference Modes

The system uses three distinct inference pipelines, each matching how Molmo2 was trained:

| Mode | Function | Pipeline | Use Case |
|---|---|---|---|
| Text-only | `ask_text()` | `apply_chat_template(tokenize=True)` | Action decisions, planning, reasoning |
| Image+text | `ask_vision()` | `apply_chat_template(tokenize=True)` | Screenshot interpretation |
| Video+text | `ask_scan()` | `apply_chat_template` with `VideoMetadata(fps=0.5)` | Scan interpretation (qualitative) |
| Video pointing | `ask_scan_pointing()` | `process_vision_info()` + `processor()` with `do_sample_frames=False` | Object counting via pointing |
| Image pointing | `ask_vision_pointing()` | `apply_chat_template(tokenize=True)` | Single-image object counting |

**Key implementation detail:** Molmo2's video processor subsamples frames when `fps >= max_fps` (default 2.0). Since our scan frames are deliberately captured positions (not redundant video frames), we must prevent subsampling. For text/QA mode, we use `synthetic_fps=0.5` to stay below the threshold. For pointing mode, we use the `process_vision_info()` pipeline from the model card, which sets `do_sample_frames=False` to bypass sampling entirely.

### Pointing and Counting

Molmo2's pointing capability is a core differentiator. When prompted with "Point to the [target]", the model outputs structured XML coordinates:

```
<points coords="frame_id point_idx x y; frame_id point_idx x y; ..."/>
```

Coordinates are normalized to [0, 1000] and decoded to pixel coordinates via regex extraction. For video pointing, each point includes a frame ID, enabling spatial localization of objects across a Z-sweep or other scan.

The `count` action uses this capability: the system captures a scan, runs the pointing pipeline with a simple prompt like `"Point to the neurons."`, counts the returned points, and then uses a text-only call to reason about the raw count (adjusting for objects that span multiple frames).

---

## System Architecture

### Pipeline Overview

```
User Question + NG Link
        |
        v
  Volume Metadata Discovery
  (zarr shape, voxel scales, axis names, layer info)
        |
        v
  Phase 1: First Look (center screenshot, "what am I looking at?")
        |
        v
  Phase 2: Plan (text-only, model plans exploration strategy)
        |
        v
  +---------------------------------------------+
  |              AGENT LOOP                      |
  |                                              |
  |  Model decides next action (text-only)       |
  |    |-- screenshot --> Playwright capture      |
  |    |-- scan ---------> Playwright video sweep |
  |    |-- count --------> Playwright + pointing  |
  |    |-- think --------> reasoning only         |
  |    +-- answer -------> final synthesis        |
  |                                              |
  |  System executes, model interprets result    |
  |  Findings accumulate, loop repeats           |
  +---------------------------------------------+
        |
        v
  Final Answer + Artifacts
  (screenshots, scan videos, findings.json, token_usage.json)
```

### Module Decomposition

| Module | File | Responsibility |
|---|---|---|
| Agent loop | `code/molmo_glancer.py` | Prompt construction, action parsing, inference calls, history management, output saving |
| GPU config | `code/gpu_config.py` | Hardware detection, profile selection, model loading (fp16 or 4-bit) |
| Visual capture | `code/visual_capture.py` | NG state building, CSS injection, canvas screenshot, scan frame generation, video saving |
| Volume info | `code/volume_info.py` | Zarr metadata discovery, FOV computation, zoom level resolution |

### GPU Profiles

The system auto-detects hardware and configures itself accordingly:

| Parameter | T4 (compact) | L40S (full) |
|---|---|---|
| VRAM | 15 GB | 45 GB |
| Quantization | 4-bit NF4 | fp16 (none) |
| Model VRAM | ~5.2 GB | ~14.5 GB |
| Max image side | 512 px | None (full 1024) |
| Max crops | 4 | 8 |
| Max scan frames | 50 | 50 |
| Max agent iterations | 8 | 20 |
| Max context tokens | 16,000 | 55,000 |
| Chromium GPU rendering | Disabled (GPU reserved for inference) | Enabled (`--use-gl=egl`) |

### Actions Available to the Agent

| Action | Visual Input | Output | When to Use |
|---|---|---|---|
| `screenshot` | Single image (1024x1024) | Text description of the view | Examine a specific position, orientation, zoom level |
| `scan` | Video (up to 50 frames) | Text description of changes across the sweep | Survey a region, observe how structures change |
| `count` | Video (up to 50 frames) | Structured point coordinates + count | Locate and count specific objects |
| `think` | None | Reasoning text | Synthesize findings, plan next steps |
| `answer` | None | Final answer text | Confident enough to respond |

---

## Visual Capture Strategy

### Screenshot Capture

1. **Build clean NG state** — apply view spec (position, layout, zoom, orientation) to base state, hide overlays (axis lines off, scale bar on, default annotations on for bounding box, statistics hidden, selected layer panel hidden)
2. **CSS injection** — hide all remaining Neuroglancer UI chrome (toolbar, layer panel, side panel, statistics panel) via `display: none !important`
3. **Navigate** — `page.goto(url, wait_until="domcontentloaded")` (not `networkidle`, which is unreliable for NG's streaming connections)
4. **Wait for data** — poll canvas pixels until >2% are non-black (`_canvas_has_data()`), then wait for consecutive canvas hashes to stabilize (`_wait_for_canvas_stable()`)
5. **Capture canvas only** — `page.locator("canvas").first.screenshot()`, bypassing all UI elements

### Scan Capture (Single-Page Sequential)

This was the most technically challenging component. After iterating through several approaches (ThreadPoolExecutor, async multi-page batched, async multi-page with shared context), we converged on **single-page sequential with hash-fragment updates**.

**Why single-page sequential wins for Neuroglancer:**

Adjacent Z-slices share ~90% of their zarr chunk data. When we update the NG state via hash-fragment (`location.hash = '!' + stateJSON`), the existing page retains all chunks in its WebGL texture cache. Only the newly-needed chunks are fetched. In contrast, separate browser contexts isolate HTTP cache, so each page independently fetches all chunks from S3 — slow and wasteful.

The implementation runs in a separate thread with its own async event loop to avoid conflicting with Playwright's sync API event loop:

```python
async def _run_sequential():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=_chromium_args(config))
        page = await ctx.new_page()
        # First frame: full navigation + wait for data
        await page.goto(states[0].to_url(), wait_until="domcontentloaded")
        await _async_wait_for_canvas_stable(page, wait_for_data=True)
        for i, state in enumerate(states):
            if i > 0:
                # Hash-fragment update — reuses cached chunks
                await page.evaluate("(h) => { location.hash = '!' + h }", state_json)
                await _async_wait_for_canvas_stable(page)
            png_bytes = await canvas.screenshot()
            ...

# Run in separate thread to avoid sync Playwright event loop conflict
t = threading.Thread(target=_run_in_thread)
t.start()
t.join()
```

### Canvas Readiness Detection

We went through five iterations on readiness detection before arriving at the current approach:

| Attempt | Method | Problem |
|---|---|---|
| 1 | `viewer.isReady()` JS polling | Always timed out on remote NG demo |
| 2 | `networkidle` wait | NG keeps streaming connections alive; timeout or unreliable |
| 3 | Canvas hash change detection | Fired on UI chrome rendering before data loaded |
| 4 | Fixed 3-second delay | Not enough for slow S3 loads |
| 5 | **`_canvas_has_data()` + hash stability** | Works reliably |

The final approach: NG UI chrome (axis labels, crosshairs, scale bar) covers <1% of canvas pixels. Actual volume data fills much more. `_canvas_has_data()` checks if >2% of pixels are non-black, reliably distinguishing empty-with-chrome from data-loaded. After data appears, `_wait_for_canvas_stable()` polls until consecutive MD5 hashes match, indicating rendering has settled.

### Video Artifact Saving

Scan videos are saved as 5-second mp4 files regardless of frame count (fps = num_frames / 5.0). The model receives frames at `synthetic_fps=0.5` to prevent Molmo2's frame sampler from discarding any frames.

---

## Prompt Strategy

### Design Principles

1. **No system role.** Molmo2's chat template only supports user/assistant roles. All instructions go in the user message.
2. **Concise prompts.** Molmo2 is a 7B model — shorter, more direct prompts outperform verbose structured instructions. We removed the original OBSERVATION/ASSESSMENT/NEXT structure in favor of simple directives.
3. **Pointing prompts are minimal.** Following the model card's examples: `"Point to the neurons."` — no elaboration needed to trigger the pointing output format.
4. **Spatial context in interpret prompts.** Scan interpretation prompts include frame spacing in micrometers and total distance, so the model can reason about physical scale.
5. **Named zoom levels.** The model picks from `wide`, `full`, `region`, `close-up`, `single-cell` instead of raw crossSectionScale floats. The system resolves these to concrete values based on volume dimensions.

### Prompt Flow

| Step | Input Type | Purpose |
|---|---|---|
| First Look | Image + text | "What kind of data is this?" |
| Plan | Text only | "What views/scans do you need?" |
| Decision (each iteration) | Text only | "What is your next action?" (JSON output) |
| Screenshot Interpret | Image + text | "Describe what you see." |
| Scan Interpret | Video + text | "Describe what you see across the frames." |
| Count Point | Video + text | "Point to the [target]." |
| Count Interpret | Text only | "Based on N points across M frames, estimate the count." |
| Forced Answer | Text only | "You must answer now." (at max iterations) |

### Avoiding Redundant Visual Processing

Each visual asset (screenshot or scan video) is sent to the model exactly once. The decision step is always text-only — the model chooses its next action based on accumulated textual findings, not by re-examining images. If the model needs both a qualitative description and a quantitative count from the same region, it issues separate `scan` and `count` actions. The count interpretation step is text-only, reasoning about point statistics without re-sending the video.

---

## Guardrails

| Guardrail | Implementation |
|---|---|
| Duplicate detection | Fingerprint-based: rounds position to nearest 5 units, compares layout/zoom/scan params |
| Position clamping | All coordinates clamped to [0, volume_shape] |
| Scale validation | Negative/zero scales default to fit zoom |
| JSON parse retry | If model output isn't valid JSON, re-prompt once with format reminder |
| Forced answer | At max iterations or after 3 consecutive duplicate actions, force answer synthesis |
| VRAM monitoring | Logged each iteration; available for triggering context compression |

---

## Hardware and Infrastructure

### Code Ocean Capsule

The system runs as a Code Ocean capsule with two hardware configurations:

- **Development:** `g4dn.4xlarge` — T4 GPU (15 GB), 16 cores, 64 GB RAM
- **Production:** `g6e.16xlarge` — L40S GPU (45 GB), 16 cores, 64 GB RAM

### Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `transformers` | >=4.57 | Molmo2 model and processor |
| `molmo_utils` | 0.0.1 | `process_vision_info` for pointing pipeline |
| `bitsandbytes` | latest | 4-bit NF4 quantization (T4 only) |
| `accelerate` | latest | `device_map="auto"` |
| `playwright` | latest | Headless Chromium for screenshot capture |
| `zarr` / `s3fs` | latest | Volume metadata from zarr sources |
| `imageio` + `pyav` | latest | Scan video encoding (libx264) |

### Chromium Optimization

- **Temp files redirected** to `/scratch/tmp` (the default `/tmp` has a 5 GB limit on Code Ocean)
- **GPU-accelerated WebGL rendering** (`--use-gl=egl`) enabled on L40S only — provides ~40% speedup for NG's WebGL canvas rendering. T4 reserves its GPU entirely for model inference.
- **Disk cache** set to `/scratch/tmp/chromium-cache` for zarr chunk reuse across sessions.

---

## Current Status and Results

### What Works (Validated on L40S)

- Model loads in fp16, occupies 14.5 GB VRAM with ~29 GB headroom for KV cache
- First look screenshot captures clean, data-filled images
- Full Z-sweep scans (50 frames) complete successfully with hash-fragment updates
- Model autonomously chooses appropriate actions: on the neuron-counting question, it first does a qualitative `scan` to survey the volume, then immediately chooses a `count` action to use pointing for precise enumeration
- Video artifacts saved as 5-second mp4 files
- All outputs saved to `/results/`

### Example Run (L40S, neuron counting)

```
Iteration 1: scan (z_sweep, 50 frames) — "I can count approximately 100 neurons"
Iteration 2: count (z_sweep, 50 frames, target=neurons) — pointing pipeline activated
```

The model's autonomous decision to use `count` after an initial qualitative scan validates the tool design — it recognized that the question required precise enumeration and selected the appropriate tool without being told to.

### Remaining Work

- **Validation across datasets:** Test with additional NG links (example_r2r, thyme_r2r)
- **Token calibration:** Measure actual vision token counts to refine VRAM budgets
- **Detail mode:** Verify max_crops=24 override for high-resolution image analysis
- **Context compression:** Trigger compression when approaching token limits
- **Rotation scans:** Quaternion interpolation for 3D view rotation sweeps
- **v2 vs v3 comparison:** Systematic quality and efficiency comparison

---

## Key Technical Decisions and Rationale

### 1. Single model, multiple inference modes (not multi-model)

Molmo2 handles orchestration, visual interpretation, and structured output (pointing) in one model loaded once. This avoids VRAM contention between multiple models and simplifies the pipeline.

### 2. Text-only decision step (not multimodal)

The agent decides its next action from textual findings history, not by re-examining images. This saves significant tokens (images cost 169-784 tokens each; videos cost 81*N tokens) and allows the context window to hold more history.

### 3. Single-page sequential scans (not parallel)

Counter-intuitive but empirically correct: one browser page sequentially updating its hash fragment outperforms parallel pages because Neuroglancer's zarr chunk cache is per-page. Adjacent Z-slices share ~90% of chunk data.

### 4. Pointing for counting (not text estimation)

Molmo2 was trained on pointing/counting tasks. Asking it to "point to the neurons" produces structured coordinates that can be precisely counted, rather than asking it to estimate a number from a video — a task where 7B models tend to hallucinate.

### 5. Named zoom levels (not raw scale values)

The model picks from semantic zoom names (`wide`, `full`, `region`, `close-up`, `single-cell`) rather than guessing crossSectionScale floats. The system resolves these to volume-appropriate concrete values. This prevents the model from requesting invalid or useless zoom levels.

### 6. 1024x1024 fixed square viewport

No directional bias in the captured images. Maps cleanly to Molmo2's crop/tile strategy (378x378 base, 2x2 pooling). Consistent across all screenshots and scan frames.

---

*Report generated 2026-04-02. Covers v3 development through the pointing/counting integration phase.*
