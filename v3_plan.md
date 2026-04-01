# v3 Plan: Autonomous Neuroglancer Visual Analysis

## Goal

Given an open-ended question about 3D data in Neuroglancer and an NG link, autonomously explore the volume — scanning, screenshotting, reasoning — and synthesize a confident answer. A single model (Molmo2-O-7B) handles all orchestration, visual interpretation, and reasoning, loaded once and resident throughout the session.

### What v3 solves over v2

v2 was a rigid 9-step pipeline: plan all views upfront → screenshot all → interpret all → synthesize. The model couldn't react to what it saw. In practice this produced 15 near-identical views (only Z varied, x/y/layout never changed) with copy-pasted hallucinated findings.

v3 fixes every layer:

| Problem | v2 | v3 |
|---|---|---|
| Workflow | Fixed step sequence | Free-form agent loop — model decides what/when |
| Visual input | 31% wasted on UI chrome, downscaled to 512px | Clean canvas-only screenshots, full resolution |
| View diversity | Only position (x,y,z) + layout | Full NG state: zoom, orientation, layers, contrast |
| Spatial survey | No survey capability | Video scans (z-sweep, pan, rotation) |
| Data adaptivity | Hardcoded for 495×495×215 | Arbitrary shapes via zoom + volume metadata |
| GPU utilization | 4-bit quantized, constrained | fp16 on L40S with 29 GB KV headroom |
| Long sessions | N/A (single pass) | Context compression for unlimited iterations |

---

## Architecture

```
                    ┌──────────────────┐
                    │   User Question  │
                    │   + NG Link      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Volume Metadata  │  ← zarr shape, voxel scales,
                    │  Discovery        │    layer names, anisotropy
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │                             │
              │        AGENT LOOP           │
              │                             │
              │  context = {question,       │
              │    volume_info, findings,    │
              │    action_history}           │
              │                             │
              │  ┌────────────────────┐     │
              │  │ Model decides next │◄────┤
              │  │ action (text-only) │     │
              │  └──┬──┬──┬──┬───────┘     │
              │     │  │  │  │             │
              │     │  │  │  └─ answer ────┼──► Final Answer
              │     │  │  │                │
              │     │  │  └─── think ──────┤
              │     │  │                   │
              │     │  └── scan ───┐       │
              │     │              │       │
              │     └── screenshot │       │
              │            │       │       │
              │     ┌──────▼───────▼──┐    │
              │     │ build_clean_    │    │
              │     │ state()         │    │
              │     │ (NG state +     │    │
              │     │  overlay hiding)│    │
              │     └──────┬──────────┘    │
              │            │               │
              │     ┌──────▼──────────┐    │
              │     │ Playwright      │    │
              │     │ 1024×1024       │    │
              │     │ canvas-only     │    │
              │     └──────┬──────────┘    │
              │            │               │
              │     ┌──────▼──────────┐    │
              │     │ Model interprets│    │
              │     │ (image or video │    │
              │     │  + text prompt) │────┘
              │     └─────────────────┘
              │
              └─────────────────────────────┘
```

### Component roles

| Component | Responsibility |
|---|---|
| **Molmo2-O-7B** | All reasoning: decides actions, interprets images/video, synthesizes answers |
| **NeuroglancerState** | Deterministic URL construction from view specs. Library import only. |
| **Playwright** | Headless Chromium: navigate to NG URL, wait for data, capture canvas |
| **Volume metadata** | Reads zarr shape/scales at startup, feeds to model for scale-aware planning |
| **GPU config** | Auto-detects hardware, selects quantization/resolution/budget parameters |

---

## Hardware & Model

### Decision: Molmo2-O-7B at fp16 on L40S

| Factor | Decision | Rationale |
|---|---|---|
| **Model** | Molmo2-O-7B (7.76B params) | Single model for text + vision + video. OLMo3-7B-Instruct backbone + SigLIP 2 encoder. |
| **Precision (L40S)** | fp16, no quantization | Full model quality. 14.5 GB weights, leaves 29.5 GB for KV cache (~59K token context). |
| **Precision (T4)** | 4-bit NF4 | Only option that fits. 3.6 GB weights, ~10.4 GB KV headroom. |
| **Why not fp32** | Halves KV budget for negligible quality gain | fp16 is standard for inference. fp32 adds nothing for this use case. |
| **Why not a larger model** | 13B at fp16 would use ~26 GB weights, leaving only ~18 GB for KV | Exploration breadth (more scans/screenshots per session) matters more than marginal per-token reasoning. The bottleneck is visual input, not model reasoning. |

### KV cache math

OLMo3-7B uses full MHA (32 KV heads), making KV cache 4× larger than GQA models:

```
KV cache per token = 2 × 32 layers × 32 heads × 128 dim × 2 bytes = 0.5 MB
```

| GPU | Weights | KV headroom | Max context | Practical session budget |
|---|---|---|---|---|
| T4 (4-bit) | 3.6 GB | ~10.4 GB | ~21K tokens | 8 iterations, 50-frame scans |
| L40S (fp16) | 14.5 GB | ~29.5 GB | ~59K tokens | 20+ iterations, 200-frame scans |

### Dual-GPU strategy

Same code, different config. Auto-detected at startup:

```python
GPU_PROFILES = {
    "compact": {      # T4 (15 GB) — dev/test
        "quantization": "4bit",
        "torch_dtype": torch.float16,
        "max_image_side": 512,
        "max_crops": 4,
        "max_scan_frames": 50,
        "max_agent_iterations": 8,
        "max_context_tokens": 16000,
    },
    "full": {         # L40S (45 GB) — production
        "quantization": None,         # pure fp16
        "torch_dtype": torch.float16,
        "max_image_side": None,       # no downscale
        "max_crops": 8,               # 24 on demand for detail
        "max_scan_frames": 200,
        "max_agent_iterations": 20,
        "max_context_tokens": 55000,
    },
}
```

**Development workflow:** write and debug on T4 (free), run for quality on L40S. All logic is identical — only numerical parameters change. A bug caught on T4 is a bug fixed on L40S.

---

## Agent Loop

### Actions

The model has 4 actions. Each iteration it picks one and specifies parameters.

#### `screenshot` — single high-detail view

```jsonc
{
    "action": "screenshot",
    "view": {
        "x": 25000, "y": 12000, "z": 500,
        "layout": "xy",
        "crossSectionScale": 0.5,
        "projectionOrientation": [0.3, 0.1, 0.0, 0.95],
        "layerVisibility": {"ch_405": true, "segmentation": false},
        "shaderRange": [0, 400]
    },
    "prompt": "Count the distinct bright cell bodies in this zoomed-in XY view."
}
```

The system renders the view and the model interprets the resulting image.

**Token cost:** ~845–980 vision tokens (1024×1024 image, 2×2=4 high-res crops + 1 base crop = 5 crops, ~169–196 tokens/crop depending on pooling floor/ceil). Exact count should be logged empirically on first run.

**When to use:** Detail work — counting, measuring, classifying specific structures.

#### `scan` — video sweep through the data

```jsonc
{
    "action": "scan",
    "scan_type": "z_sweep",
    "start": {"x": 25000, "y": 12000, "z": 0},
    "end":   {"x": 25000, "y": 12000, "z": 2000},
    "frames": 50,
    "layout": "xy",
    "crossSectionScale": 5.0,
    "prompt": "Watch this Z-sweep and identify depth ranges with high neuron density."
}
```

The system generates a sequence of Playwright screenshots with interpolated positions, saves the sequence as an mp4 artifact for debugging, and feeds the frames directly to Molmo2 as video (81 tokens/frame via 3×3 pooling, no multi-crop).

**Token cost:** 81 × N frames. 50-frame scan = ~4,050 tokens (vs ~42,250 as 50 individual images).

**When to use:** Survey, spatial orientation, finding regions of interest before zooming in.

**Scan types:**

| Type | What varies | Use case |
|---|---|---|
| `z_sweep` | Z position | Survey depth extent |
| `x_pan` / `y_pan` | X or Y position | Survey a large plane |
| `rotation` | projectionOrientation | Understand 3D structure |
| `zoom_ramp` | crossSectionScale | Find appropriate detail level |
| `freeform` | arbitrary position vector | Trace a structure |

#### `think` — internal reasoning (no visual input)

```jsonc
{
    "action": "think",
    "reasoning": "I've surveyed the full Z range and found high density at Z=400-600..."
}
```

**Token cost:** Text-only, minimal. No Playwright, no vision tokens.

**When to use:** Mid-session synthesis, re-planning, coverage assessment.

#### `answer` — final synthesis

```jsonc
{
    "action": "answer",
    "answer": "Based on my analysis of 8 views across the volume..."
}
```

Terminates the loop. Output saved to results.

### Decision prompt

**Important:** Molmo2's chat template does NOT support a `system` role — only `user` and `assistant`. All instructions must be in the user message.

Each iteration, the model receives a single user message:

```
You are a neuroglancer data analyst. You explore 3D microscopy data
by taking screenshots and video scans, then synthesize an answer.

ACTIONS AVAILABLE:
  - screenshot: Take a single high-detail view (specify full view params)
  - scan: Sweep through data as video (specify axis, range, frames)
  - think: Reason about findings so far (no visual input)
  - answer: Provide your final answer (only when confident)

VOLUME INFO:
  Shape: 50000 × 50000 × 2000 voxels (x × y × z)
  Voxel size: 8nm × 8nm × 40nm (anisotropic, z is 5× coarser)
  Layers: [ch_405 (image), segmentation (segmentation)]
  Viewport: 1024×1024 (square). At crossSectionScale=S, shows S·1024 × S·1024 voxels.

QUESTION: How many neurons can you count in this volume?

COVERAGE SUMMARY (iterations 1-10):
  Surveyed full Z range. Neurons found throughout z=100-1800...

RECENT FINDINGS (iterations 11-15):
  [action 11: screenshot, xy, x=12500, z=600, scale=0.3]
  [finding 11: "Dense cluster of 31 neurons. Three unusually large..."]
  [visible window: x=[12347..12653], y=[24847..25153] (307×307 voxels)]
  ...

What is your next action? Respond with a JSON object.
```

### Guardrails

| Guardrail | Purpose | Implementation |
|---|---|---|
| **Max iterations** | Prevent infinite loops | Configurable (8 on T4, 20 on L40S). At limit, force answer. |
| **Action validation** | Reject invalid specs | Clamp positions to volume bounds, validate scales, check layout values. |
| **Duplicate detection** | Prevent redundant views | Skip if new view overlaps >80% with a prior view (same layout, position within FOV, similar scale). |
| **Output format retry** | Handle malformed JSON | If model output isn't valid JSON, re-prompt once with format example. |
| **VRAM monitoring** | Prevent OOM | Check `torch.cuda.memory_allocated()` before each action. If tight, trigger context compression. |

---

## Visual Input Pipeline

### Decision: Fixed 1024×1024 square viewport

| Factor | Decision | Rationale |
|---|---|---|
| **Shape** | Square (1:1) | No directional bias. Cross-sections show equal X and Y extent. Data dimensions are symmetric — no reason to see more in one axis. |
| **Size** | 1024×1024 | At max_crops=8, tiles into 2×2=4 crop grid (~845 tokens). At max_crops=9+, tiles into 3×3=9 for more detail. Downscales cleanly to 512×512 on T4. |
| **Fixed vs dynamic** | Fixed | Consistent crop tiling → predictable token counts. Stays in model's training distribution. Extreme aspect ratios (10:1) would produce unusual crop layouts. |
| **Data shape handling** | Via `crossSectionScale` | The viewport is a fixed window. The model zooms to fill it for any data shape. A 50000×200 dataset needs a scan along the long axis, not a 250:1 viewport. |

### Decision: Clean screenshots (no UI chrome)

Every generated NG state hides overlays:

```python
state["showAxisLines"] = False           # no crosshair
state["showScaleBar"] = False            # no scale bar
state["showDefaultAnnotations"] = False  # no yellow bounding box
state["crossSectionBackgroundColor"] = "#000000"
state["selectedLayer"] = {"visible": False}
state["statistics"] = {"visible": False}
```

Playwright injects CSS to hide remaining UI:

```python
page.add_style_tag(content="""
    .neuroglancer-viewer-top-row { display: none !important; }
    .neuroglancer-layer-panel { display: none !important; }
    .neuroglancer-layer-side-panel { display: none !important; }
    .neuroglancer-statistics-panel { display: none !important; }
""")
```

Screenshot captures only the canvas element:

```python
canvas = page.locator("canvas").first
png_bytes = canvas.screenshot()
```

**Result:** 100% of pixels are data. v2 wasted ~31% on UI chrome.

### Decision: Readiness polling, not sleep

Replace `time.sleep(12)` with:

```python
page.wait_for_function("""() => {
    const v = window.viewer;
    return v && typeof v.isReady === 'function' && v.isReady();
}""", timeout=30000)
```

Faster (returns as soon as data loads), more reliable (no risk of capturing mid-load).

### Full NG state prediction

The model controls the complete view spec, not just position:

| Property | NG state key | What it controls |
|---|---|---|
| Position | `position` | Where in the volume |
| Layout | `layout` | `xy`, `xz`, `yz`, `3d`, `4panel`, `xy-3d`, etc. |
| 2D zoom | `crossSectionScale` | Canonical voxels per pixel. <1 = zoom in, >1 = zoom out |
| 3D zoom | `projectionScale` | Canonical voxels per viewport height |
| 3D rotation | `projectionOrientation` | Camera quaternion [x, y, z, w] |
| Oblique slice | `crossSectionOrientation` | Slice plane quaternion |
| Layer visibility | per-layer `visible` | Toggle channels on/off |
| Contrast | `shaderControls.normalized.range` | [vmin, vmax] for brightness |
| Slice planes in 3D | `showSlices` | Show/hide cross-section planes in 3D view |

All constructed deterministically by `build_clean_state()` from the model's view spec. No browser interaction beyond navigate-and-screenshot.

---

## Adapting to Arbitrary Data

### Volume metadata discovery

At pipeline startup, before any model calls:

```python
def discover_volume(ng_state: dict) -> VolumeInfo:
    dims = ng_state["dimensions"]
    voxel_scales = [v[0] for v in dims.values()]
    shape = read_shape_from_source(ng_state["layers"][0]["source"])

    canonical = min(voxel_scales[:3])
    factors = [s / canonical for s in voxel_scales[:3]]

    return VolumeInfo(
        shape=shape,
        voxel_scales=voxel_scales,
        axis_names=list(dims.keys()),
        layers=[{"name": l["name"], "type": l["type"]} for l in ng_state["layers"]],
        canonical_factors=factors,
        anisotropy_ratio=max(factors) / min(factors),
    )
```

This tells the model:
- **How big the data is** — to plan position ranges and avoid out-of-bounds
- **What zoom levels are meaningful** — `crossSectionScale = shape[0] / 1024` fits the full X extent
- **How many voxels are visible** — at scale S, the square viewport shows S·1024 × S·1024 canonical voxels
- **Whether the data is anisotropic** — to interpret stretched views correctly

### FOV feedback

After each screenshot/scan, the system computes and reports what was visible:

```python
VIEWPORT_SIZE = 1024

def compute_fov(scale, canonical_factors):
    return [VIEWPORT_SIZE * scale / f for f in canonical_factors]

# Fed back in history:
# "Visible window: x=[24488..25512], y=[11488..12512] (1024×1024 voxels at scale=1.0)"
```

This prevents redundant views and helps the model plan gap-filling coverage.

### Scale-aware scanning

For large data, the natural strategy is survey → region of interest → detail:

1. **Survey scan** — high crossSectionScale, sweep an axis → identify structure
2. **Region screenshots** — medium zoom on regions of interest → characterize features
3. **Detail screenshots** — low crossSectionScale → count, measure, classify

The model chooses this strategy organically through the agent loop. Volume metadata in the prompt gives it the numbers to set meaningful zoom levels.

---

## Scan Implementation

### Generating frames

```python
def execute_scan(page, base_state, scan_spec, volume_info, config, scan_id):
    positions = np.linspace(
        np.clip([scan_spec["start"][k] for k in "xyz"], 0, volume_info.shape[:3]),
        np.clip([scan_spec["end"][k] for k in "xyz"], 0, volume_info.shape[:3]),
        min(scan_spec["frames"], config["max_scan_frames"]),
    )

    frames = []
    for pos in positions:
        state = build_clean_state(base_state, {
            "x": pos[0], "y": pos[1], "z": pos[2],
            "layout": scan_spec.get("layout", "xy"),
            "crossSectionScale": scan_spec.get("crossSectionScale", 1.0),
        })
        # Update hash in-place (faster than full page.goto)
        page.evaluate("(h) => { location.hash = '!' + h }",
                      json.dumps(state.data, separators=(",", ":")))
        page.wait_for_function(
            "() => window.viewer && window.viewer.isReady()", timeout=15000)

        canvas = page.locator("canvas").first
        frames.append(Image.open(BytesIO(canvas.screenshot())).convert("RGB"))

    # Save video artifact for debugging / review
    save_scan_video(frames, scan_id)

    return frames
```

### Saving scan videos

Every scan is saved as an mp4 alongside the screenshot PNGs, for debugging and review:

```python
def save_scan_video(frames, scan_id, fps=4):
    import imageio.v3 as iio

    video_path = RESULTS_DIR / "scans" / f"scan_{scan_id:03d}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    iio.imwrite(video_path, [np.array(f) for f in frames], fps=fps)
    print(f"  Scan video saved to {video_path} ({len(frames)} frames)")
```

The video files are output artifacts only — the model receives the PIL Image list directly, not the mp4 file. This avoids an encode→decode roundtrip through `decord2` and keeps the frames pixel-identical to what Playwright captured.

### Feeding video to Molmo2

Frames are passed directly as PIL Images — no intermediate video file in the inference path:

```python
def ask_scan(model, processor, frames, prompt, max_new_tokens=1024):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames},  # list of PIL Images, passed directly
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
```

**Early validation required (Phase 3, step 11):** Two things to test:

1. **Frame input format.** The video processor's `load_video()` accepts non-string inputs (returns as-is), so a list of PIL Images *should* work with `{"type": "video", "video": frames}`. But no official Molmo2 example shows this — the model card only shows URL strings. If it doesn't work, fall back to passing the saved mp4 path (already saved by `save_scan_video()`) and let `decord2` decode it. Also test passing a numpy array of shape `(N, H, W, 3)` as a third option.

2. **Frame sub-sampling.** The video processor has `do_sample_frames=true` and `max_fps=2` by default. When we pass pre-captured frames (no real timestamps), this sampling logic might discard frames. Verify that pre-decoded frame lists bypass sampling, or explicitly override with `do_sample_frames=false` or `max_fps` set high enough.

Video mode uses 3×3 pooling → 81 tokens/frame (vs 169–196 per image crop). No multi-crop tiling, no cross-frame compression — efficiency comes from coarser pooling and single-crop-per-frame.

---

## Context Management

### KV cache clears naturally

HuggingFace `model.generate()` builds KV cache per call and discards it on return. There is no persistent KV across the agent loop. Each iteration pays only for its own prompt.

The growing cost is **text history** in the prompt — each iteration appends ~200–400 text tokens (action + finding). After 50 iterations, that's ~10,000–20,000 tokens of history per call, which is manageable (5–10 GB KV on L40S).

### Context compression for long sessions

For sessions exceeding ~30 iterations, compress older findings:

```python
def build_history(findings, max_recent=10, max_summary_tokens=2000):
    if len(findings) <= max_recent:
        return format_findings_full(findings)

    older = findings[:-max_recent]
    recent = findings[-max_recent:]

    summary = compress_to_summary(older)
    # e.g.: "Iterations 1-25: Surveyed full Z via 3 scans. Neurons throughout
    #         z=100-1800, density ~20/window center, ~12/window edges.
    #         Two dense clusters at (12000,25000) and (38000,25000)."

    return f"COVERAGE SUMMARY (iterations 1-{len(older)}):\n{summary}\n\n" \
           f"RECENT FINDINGS (iterations {len(older)+1}-{len(findings)}):\n" \
           f"{format_findings_full(recent)}"
```

**Sliding window:** last N findings in full detail, older findings compressed to a coverage summary. The summary itself can be model-generated (via a `think` action) or simple text truncation.

This means sessions of 50, 100, or more iterations are possible — the prompt stays bounded while preserving cumulative knowledge + recent detail.

---

## Outputs

| File | Content |
|---|---|
| `results/answer.txt` | Final synthesized answer |
| `results/findings.json` | Per-iteration action + finding + FOV metadata |
| `results/screenshots/` | PNGs: `view_001.png`, `view_005.png`, etc. |
| `results/scans/` | MP4s: `scan_002.mp4`, `scan_006.mp4`, etc. (one video per scan action) |
| `results/ng_states.json` | NG state used for each screenshot/scan frame |
| `results/token_usage.json` | Per-iteration token counts (vision + text, in + out) |
| `results/output.log` | Full pipeline log (via tee) |

---

## Molmo2-O-7B: Capability Notes & Known Limitations

Findings from auditing the model card, tech report, and processor source code.

### Strengths we're leveraging

- **Strong instruction-following** (IFEval 85.6) — good for structured JSON action output
- **Strong reasoning** (MATH 87.3, AIME 44.3) — good for spatial reasoning and counting
- **Native video support** with efficient 3×3 pooling (81 tokens/frame)
- **Multi-image support** — multiple `{"type": "image"}` entries in one message work
- **Configurable `max_crops`** — overridable per-call for detail vs standard views
- **Image + text in same message** — no need for separate calls

### Limitations to design around

| Limitation | Impact | Mitigation |
|---|---|---|
| No system message in chat template | Can't use `{"role": "system"}` | All instructions in user message (already updated above) |
| MMMU 45.8 (weak visual reasoning) | May struggle with complex multi-step visual analysis | Focused single-task prompts per call, not monolithic reasoning |
| -O variant weaker than 8B Qwen variant | 59.7 vs 63.1 avg benchmark score | Accepted tradeoff for fully open model |
| Extended context trained briefly | Quality may degrade past ~16K tokens | Context compression keeps prompts bounded |
| Full MHA (not GQA) | 4× KV cache cost vs comparable models | Accounted for in VRAM budgets |
| Video benchmarks unpublished for -O | May underperform on video vs 8B variant | Scans are for survey (coarse), not fine analysis |
| `do_sample_frames=true` default | May sub-sample our pre-captured frames | Validate in Phase 3; override if needed |

### API details confirmed

| API aspect | Confirmed behavior |
|---|---|
| Image input | `{"type": "image", "image": <PIL Image>}` via `apply_chat_template` |
| Text-only input | Works with no image/video content |
| Chat template | `<\|im_start\|>user/assistant` format, no system role |
| `max_crops` override | Pass via processor kwargs; default 8, up to 24 |
| Small images (<378px) | Upscaled to 378×378 via bilinear interpolation |
| Video pooling | 3×3 → 81 tokens/frame, confirmed in `video_preprocessor_config.json` |
| Image pooling | 2×2 → 169–196 tokens/crop, confirmed in `preprocessor_config.json` |
| Max context | 65,536 tokens (YaRN RoPE from base 8,192) |
| Video frame limit | 384 frames max |
| `trust_remote_code` | Required for both model and processor |

---

## Dependencies

| Package | Purpose | Notes |
|---|---|---|
| `transformers>=4.57` | Molmo2 inference | Core |
| `bitsandbytes` | 4-bit quantization (T4 only) | Not used on L40S |
| `accelerate` | `device_map="auto"` | HF standard |
| `molmo_utils` | Molmo2 processor helpers | Required per HF model card |
| `decord2` | Video frame decoding for Molmo2 | Required by processor; used if mp4 fallback path needed |
| `torch` | Already installed | |
| `playwright` | Headless browser screenshots | Already installed |
| `neuroglancer-chat` | `NeuroglancerState` import | In `lib/`, used as library |
| `zarr` / `s3fs` | Volume metadata discovery | New — read shape/scales from zarr |
| `imageio` | Save scan videos as mp4 | New — debug/review artifacts |
| `numpy` | Scan position generation, FOV math | Already installed |

---

## Implementation Plan

### Phase 1: Infrastructure

1. **GPU config module** — `detect_gpu_profile()`, `GPU_PROFILES`, `load_model(config)`
2. **Volume metadata discovery** — `discover_volume()` reads zarr shape/scales from NG state
3. **Clean state builder** — `build_clean_state()` applies overlay hiding + full view spec
4. **Clean screenshot capture** — CSS injection, canvas-only screenshot, readiness polling
5. **Test on T4** — model loads, screenshot is clean, NG state is correct

### Phase 2: Agent Loop

6. **Action parser** — parse model JSON output into `screenshot`/`scan`/`think`/`answer`
7. **Agent loop skeleton** — iterate: prompt → parse → execute → append to context → repeat
8. **FOV computation** — compute and report visible window after each action
9. **Guardrails** — max iterations, duplicate detection, format retry, position validation
10. **Test on T4** — full loop runs, terminates, produces answer

### Phase 3: Scans

11. **Video API validation** — minimal test: create 5 dummy PIL Images, pass as `{"type": "video"}` to processor, confirm model generates a response. Test frame sub-sampling behavior. Determine working input format (PIL list, numpy array, or mp4 path fallback). **Do this first — it gates the rest of Phase 3.**
12. **Scan executor** — generate frame sequence, hash-fragment updates, readiness polling
13. **Video inference** — `ask_scan()` feeds frames to Molmo2 in video mode
14. **Scan types** — z_sweep, x_pan, y_pan, rotation, zoom_ramp
15. **Test on T4** — scan produces frames, model interprets video, findings make sense

### Phase 4: Context & Scale

16. **Context compression** — sliding window with summary for long sessions
17. **Scale-aware prompting** — volume metadata in prompts, FOV feedback
18. **Detail mode** — max_crops=24 for critical views (L40S only). Verify `max_crops` override API via processor kwargs.
19. **Test on L40S** — full session with fp16, long scans, detailed screenshots

### Phase 5: Validation

20. **Empirical token calibration** — log actual vision token counts for 1024×1024 images at max_crops=4,8,24 and for video frames. Update VRAM budget estimates with real numbers.
21. **Run on existing test data** — `example_ng_link.txt` with known questions
22. **Run on large data** — test with a larger volume to validate scale handling
23. **Compare v2 vs v3** — same question, same data, compare answer quality and exploration diversity

---

## Example Sessions

### Small dataset (495×495×215)

```
1. [scan z_sweep, 20 frames, z=0..214, scale=1.0]
   → "Bright cell bodies throughout. Highest density z=100-150."

2. [screenshot, xy, z=125, scale=0.5]
   → "23 distinct neurons in this 512×512 window at center."

3. [screenshot, xz, z=125, scale=1.0]
   → "Neurons extend ~50 voxels in Z. Layered structure."

4. [think]
   → "Need edge samples and different Z depths to estimate total."

5. [screenshot, xy, z=125, x=100, y=100, scale=0.5]
   → "17 neurons — lower density at edge."

6. [answer]
   → "Estimated 800-1,200 neurons based on sampling."
```

### Large dataset (50,000 × 50,000 × 2,000)

```
1. [scan z_sweep, 40 frames, z=0..2000, scale=40]
   → "Dark region z=0-300, dense tissue z=300-1800, surface z=1800+."

2. [scan x_pan, 30 frames, x=0..50000, y=25000, z=1000, scale=20]
   → "Two distinct regions: left darker, right brighter."

3. [screenshot, xy, x=12000, y=25000, z=1000, scale=2.0]
   → "Medium zoom (2048×2048 window): ~20 cell bodies visible."

4. [screenshot, xy, x=12000, y=25000, z=1000, scale=0.3]
   → "High zoom (307×307 window): 8 neurons with visible dendrites."

5. [think]
   → "Need to sample right side for density comparison."

6. [screenshot, xy, x=38000, y=25000, z=1000, scale=0.3]
   → "12 neurons — higher density on right side."

7. [scan rotation, 16 frames, position=(25000,25000,1000), scale=30000]
   → "3D: elongated structure running left-right. Layer boundaries."

8. [answer]
```

---

## Key Files

| Path | Purpose |
|---|---|
| `code/molmo_glancer.py` | Main v3 pipeline — agent loop, actions, prompt construction |
| `code/gpu_config.py` | GPU detection, profiles, model loading |
| `code/visual_capture.py` | Playwright capture: clean state, CSS injection, canvas screenshot, scans |
| `code/volume_info.py` | Volume metadata discovery from zarr sources |
| `code/run_v3` | Shell entry point (env vars, logging) |
| `code/_dev_startup.sh` | Dependency installation |
| `code/_download_weights.sh` | Model weight download |
| `code/lib/neuroglancer-chat/` | NeuroglancerState (library import) |

---

*v3 plan finalized 2026-04-01. Supersedes v2_plan.md and all draft/proposal documents.*
