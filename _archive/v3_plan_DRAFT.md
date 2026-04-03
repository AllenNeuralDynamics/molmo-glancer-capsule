# v3 Plan — DRAFT: Inputs & Workflow

> This draft covers visual input generation, the agentic workflow loop, and data-shape adaptivity. GPU/model configuration is deferred to a separate section.

---

## 1. Goal (unchanged)

Given an open-ended question about 3D data in Neuroglancer, autonomously explore the data and synthesize a confident answer. All orchestration, visual understanding, and reasoning use a single model loaded once.

**What changes in v3:**
- The model has **full control** over what to do and when — no fixed step sequence
- **Scans** (video sweeps) are a first-class action alongside single screenshots
- **Arbitrary data shapes and scales** are handled — the pipeline adapts to the data, not the other way around
- **Clean visual input** — 100% data pixels, no UI chrome

---

## 2. Agentic Loop

### v2 was a fixed pipeline

```
plan → screenshot all → interpret all → synthesize (one pass, rigid)
```

The model planned all views upfront, saw all screenshots in batch, then answered. It could not react to what it saw, zoom in on something interesting, or decide it needed a different orientation.

### v3 is a free-form agent loop

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│   CONTEXT (persists across iterations)               │
│   ┌────────────────────────────────────────────┐     │
│   │ question, volume_info, accumulated_findings │     │
│   │ action_history, coverage_map               │     │
│   └────────────────────────────────────────────┘     │
│                                                      │
│   while True:                                        │
│       action = model.decide(context)                 │
│                                                      │
│       switch action.type:                            │
│           "screenshot" → take + interpret image      │
│           "scan"       → take + interpret video      │
│           "think"      → internal reasoning step     │
│           "answer"     → final synthesis → exit      │
│                                                      │
│       context.append(action, result)                 │
│                                                      │
└──────────────────────────────────────────────────────┘
```

The model sees its full history and decides what to do next. It can take 2 actions or 20. It can scan, then screenshot a detail, then scan a different axis, then answer.

### Actions

#### `screenshot` — single high-detail view

The model specifies a full NG view spec. The system renders a clean screenshot and the model interprets it.

```jsonc
{
    "action": "screenshot",
    "view": {
        "x": 25000, "y": 12000, "z": 500,
        "layout": "xy",
        "crossSectionScale": 0.5,
        "projectionOrientation": null,
        "layerVisibility": {"ch_405": true, "segmentation": false},
        "shaderRange": [0, 400]
    },
    "prompt": "Count the distinct bright cell bodies in this zoomed-in XY view."
}
```

**When to use:** Detail work — counting, measuring, classifying specific structures.

**Token cost:** ~1,000–1,500 vision tokens per image (full resolution, multi-crop).

#### `scan` — video sweep through the data

The model specifies scan parameters. The system generates a sequence of screenshots, stitches them into a video, and the model watches it in video mode.

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

Scan types:

| Type | What varies | Use case |
|------|------------|----------|
| `z_sweep` | Z position | Survey depth extent |
| `x_pan` / `y_pan` | X or Y position | Survey a large plane |
| `rotation` | projectionOrientation | Understand 3D structure |
| `zoom_ramp` | crossSectionScale | Find appropriate detail level |
| `freeform` | arbitrary position path | Trace a structure, follow a trajectory |

**When to use:** Survey, spatial orientation, finding regions of interest.

**Token cost:** ~81 tokens/frame × N frames. A 50-frame scan costs ~4,050 tokens — vs ~75,000 for 50 individual screenshots.

#### `think` — internal reasoning (no visual input)

The model reasons about what it's learned so far without taking a new view. Useful for mid-exploration synthesis, re-planning, or deciding whether to continue.

```jsonc
{
    "action": "think",
    "reasoning": "I've surveyed the full Z range and found high density at Z=400-600 and Z=1200-1400. I should zoom into these two regions in XY to count individual neurons."
}
```

**Token cost:** Text-only, minimal.

#### `answer` — final synthesis

The model has enough information and produces the final answer.

```jsonc
{
    "action": "answer",
    "answer": "Based on my analysis of 8 views across the volume..."
}
```

### Decision prompt structure

Each iteration, the model receives:

```
SYSTEM: You are a neuroglancer data analyst. You have access to these actions:
  - screenshot: Take a single high-detail view (specify full view params)
  - scan: Sweep through data as video (specify axis, range, frames)
  - think: Reason about findings so far
  - answer: Provide your final answer (only when confident)

VOLUME INFO:
  Shape: {shape} voxels
  Voxel size: {voxel_scales}
  Layers: {layer_names}
  Anisotropy: {ratio}
  Viewport: 1024×1024 (square). At crossSectionScale=1, shows 1024×1024 voxels.

QUESTION: {question}

HISTORY:
  [action 1: scan z_sweep, 50 frames, z=0..2000]
  [finding 1: "High neuron density at z=400-600, sparse at z=0-200..."]
  [action 2: screenshot, xy at z=500, scale=0.5]
  [finding 2: "Counted 23 distinct cell bodies in this 512×512 voxel window..."]

What is your next action? Respond with a JSON object.
```

### Guardrails

A 7B model in a free-form loop needs some constraints:

| Guardrail | Purpose |
|-----------|---------|
| **Max iterations** (configurable, default 15) | Prevent infinite loops |
| **Max total vision tokens** (configurable) | Stay within VRAM/time budget |
| **Action validation** | Clamp positions to volume bounds, validate scales, reject degenerate views |
| **Duplicate detection** | Warn/skip if a new view overlaps >80% with a previous view |
| **Forced answer** | At max iterations, force an answer with whatever findings exist |
| **Output format retry** | If model output isn't valid JSON, re-prompt once with format reminder |

---

## 3. Clean Visual Input

### NG state: hide all overlays

Every generated state includes:

```python
state["showAxisLines"] = False           # no crosshair
state["showScaleBar"] = False            # no scale bar
state["showDefaultAnnotations"] = False  # no yellow bounding box
state["crossSectionBackgroundColor"] = "#000000"
state["selectedLayer"] = {"visible": False}   # close side panel
state["statistics"] = {"visible": False}      # close stats panel
```

### Playwright: canvas-only screenshot

```python
# Inject CSS to hide any remaining UI
page.add_style_tag(content="""
    .neuroglancer-viewer-top-row { display: none !important; }
    .neuroglancer-layer-panel { display: none !important; }
    .neuroglancer-layer-side-panel { display: none !important; }
    .neuroglancer-statistics-panel { display: none !important; }
""")

# Wait for data readiness (no more sleep(12))
page.wait_for_function("""() => {
    const v = window.viewer;
    return v && typeof v.isReady === 'function' && v.isReady();
}""", timeout=30000)

# Screenshot only the canvas
canvas = page.locator("canvas").first
png_bytes = canvas.screenshot()
```

**Result:** 100% of pixels are data. No overlays, no chrome.

### Fixed square viewport (1024×1024)

The viewport is **fixed at 1024×1024** for all views (cross-sections, 3D, scans). Rationale:

- **No directional bias.** Cross-sections show equal extent in both displayed axes. There's no reason to see more X than Y — the data dimensions are symmetric.
- **Square crops tile cleanly.** Molmo2 tiles images into 378×378 crops. A square image produces symmetric grids (2×2=4, 3×3=9) with minimal waste. Asymmetric viewports (16:9) produce lopsided grids.
- **Consistent vision tokens.** Every screenshot has the same shape → predictable crop tiling → predictable token counts across the session.
- **Stays in training distribution.** The model sees a consistent input shape rather than arbitrary aspect ratios that may be out of distribution.
- **crossSectionScale handles data shape, not the viewport.** For any data shape — square, elongated, anisotropic — the model zooms to fill the fixed viewport. A 50000×200 dataset doesn't need a wide viewport; it needs a scan along the long axis at appropriate zoom.

At `crossSectionScale=1.0`, the viewport shows **1024×1024 canonical voxels**. The model controls what it sees via position + zoom, not viewport shape.

With `max_crops=8`, a 1024×1024 image tiles into a 2×2=4 crop grid (plus 1 global = 5 total, ~845 vision tokens). With `max_crops=9+`, it can use 3×3=9 crops for more detail (~1,521 tokens). The `max_crops` parameter is the control knob for detail level, not the viewport size.

On T4 with downscaling to 512px, the 1024×1024 capture becomes 512×512 — also square, also clean.

---

## 4. Adapting to Arbitrary Data

### Volume metadata discovery

At pipeline startup, before any model calls:

```python
def discover_volume(ng_state: dict) -> VolumeInfo:
    """Read shape, voxel scales, layer info from NG state + zarr metadata."""
    dims = ng_state["dimensions"]
    voxel_scales = [v[0] for v in dims.values()]
    axis_names = list(dims.keys())

    shape = read_shape_from_source(ng_state["layers"][0]["source"])

    layers = [{"name": l["name"], "type": l["type"],
               "visible": l.get("visible", True)} for l in ng_state["layers"]]

    canonical = min(voxel_scales[:3])
    factors = [s / canonical for s in voxel_scales[:3]]

    return VolumeInfo(
        shape=shape,
        voxel_scales=voxel_scales,
        axis_names=axis_names,
        layers=layers,
        canonical_factors=factors,
        anisotropy_ratio=max(factors) / min(factors),
    )
```

This feeds into every model prompt, so the model knows:
- How big the data is (to plan position ranges)
- What zoom levels are meaningful (to set crossSectionScale)
- How many voxels are visible at each zoom (to estimate coverage)
- Whether the data is anisotropic (to interpret stretched views correctly)

### FOV computation & feedback

After each screenshot/scan, the system computes what was actually visible and feeds it back:

```python
VIEWPORT_SIZE = 1024  # square viewport

def compute_fov(scale, canonical_factors):
    fov_voxels = [
        VIEWPORT_SIZE * scale / canonical_factors[0],
        VIEWPORT_SIZE * scale / canonical_factors[1],
    ]
    return fov_voxels

# Fed back to model in history:
# "Visible window: x=[24488..25512], y=[11488..12512] (1024×1024 voxels at scale=1.0)"
```

This prevents the model from re-scanning areas it's already covered and helps it plan gap-filling views. The square FOV simplifies this — at scale S, the visible window is always S·1024 × S·1024 canonical voxels (before anisotropy factors).

### Scale-aware scan planning

For scans, the system computes frame positions from the model's start/end/frames spec:

```python
def generate_scan_positions(scan_spec, volume_info):
    start = np.array([scan_spec["start"]["x"], scan_spec["start"]["y"], scan_spec["start"]["z"]])
    end = np.array([scan_spec["end"]["x"], scan_spec["end"]["y"], scan_spec["end"]["z"]])

    # Clamp to volume bounds
    start = np.clip(start, [0, 0, 0], volume_info.shape[:3])
    end = np.clip(end, [0, 0, 0], volume_info.shape[:3])

    positions = np.linspace(start, end, scan_spec["frames"])
    return positions
```

---

## 5. Scan Implementation

### Generating the video

```python
def execute_scan(page, base_state, scan_spec, volume_info):
    frames = []
    positions = generate_scan_positions(scan_spec, volume_info)

    for pos in positions:
        state = build_clean_state(base_state, {
            "x": pos[0], "y": pos[1], "z": pos[2],
            "layout": scan_spec.get("layout", "xy"),
            "crossSectionScale": scan_spec.get("crossSectionScale", 1.0),
            # ... other view params
        })
        url = state.to_url()

        # Update URL without full reload (faster)
        page.evaluate(f"window.location.hash = '!' + decodeURIComponent('{quote(json.dumps(state.data))}')")
        page.wait_for_function("() => window.viewer && window.viewer.isReady()", timeout=15000)

        canvas = page.locator("canvas").first
        png_bytes = canvas.screenshot()
        frames.append(Image.open(BytesIO(png_bytes)).convert("RGB"))

    return frames  # list of PIL Images → feed as video to Molmo2
```

### Feeding to Molmo2 in video mode

Molmo2's video mode uses 3×3 pooling (81 tokens/frame) vs 2×2 for images (169+/frame). Frames are passed as a list:

```python
def ask_scan(model, processor, frames, prompt, max_new_tokens=1024):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames},  # list of PIL Images
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
    # ... decode and return
```

### Scan speed optimization

For Z-sweeps and pans, the data source doesn't change — only the position. Instead of navigating to a new URL each frame, we can update the hash fragment in-place:

```python
# Much faster than page.goto() — avoids full page reload
page.evaluate("""(newHash) => { window.location.hash = '!' + newHash; }""",
              json.dumps(state.data, separators=(",", ":")))
```

Combined with `viewer.isReady()` polling, this could achieve 1-3 frames/second depending on data load time — a 50-frame scan in 20-50 seconds instead of 50 × 12 = 600 seconds.

---

## 6. Typical Session Flow

### Small dataset (current scale: 495×495×215)

```
1. [scan z_sweep, 20 frames, z=0..214, scale=1.0]
   → "Bright cell bodies visible throughout. Highest density around z=100-150."

2. [screenshot, xy, z=125, scale=0.5, zoomed in]
   → "23 distinct neurons visible in this 512×512 voxel window at center."

3. [screenshot, xz, z=125, scale=1.0]
   → "Neurons extend ~50 voxels in Z. Clear layered structure."

4. [think]
   → "Density ~23 per 512×512 window. Volume is 495×495. Roughly 1 window across.
      Need to sample edges and different Z depths to estimate total."

5. [screenshot, xy, z=125, x=100, y=100, scale=0.5]
   → "17 neurons at edge — lower density."

6. [answer]
   → "Estimated 800-1,200 neurons based on sampling at 3 positions and 1 depth."
```

### Large dataset (e.g., 50,000 × 50,000 × 2,000)

```
1. [scan z_sweep, 40 frames, z=0..2000, scale=40]
   → "Overview: dark region z=0-300, dense tissue z=300-1800, surface z=1800+."

2. [scan x_pan, 30 frames, x=0..50000, y=25000, z=1000, scale=20]
   → "Tissue spans full X. Two distinct regions: left half darker, right half brighter."

3. [screenshot, xy, x=12000, y=25000, z=1000, scale=2.0]
   → "Medium zoom (2048×2048 voxel window): can see ~20 cell bodies."

4. [screenshot, xy, x=12000, y=25000, z=1000, scale=0.3]
   → "High zoom (307×307 voxel window): 8 neurons clearly resolved with dendrites."

5. [think]
   → "Need to sample right side for comparison."

6. [screenshot, xy, x=38000, y=25000, z=1000, scale=0.3]
   → "12 neurons here — higher density on right side."

7. [scan rotation, 16 frames, position=(25000,25000,1000), scale=30000]
   → "3D: elongated structure running left-right. Layer boundaries visible."

8. [answer]
   → "..."
```

---

## 7. Architecture Diagram

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
              │     │  │  └─── think ──────┤ (no I/O, feeds
              │     │  │                   │  back to context)
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
              │     │ (canvas-only    │    │
              │     │  screenshot,    │    │
              │     │  dynamic        │    │
              │     │  viewport)      │    │
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

---

## 8. Key Design Decisions

### Model is the agent, system is the executor

The model decides **what** to do (which action, what parameters). The system handles **how** (NG state construction, Playwright capture, viewport sizing, position clamping). The model never touches browser APIs or file I/O directly.

### Scans are first-class, not an optimization

Scans aren't just "batch screenshots stitched together." They're a distinct modality: the model requests a spatial sweep, watches the result as video (81 tokens/frame), and reasons about spatial patterns that are hard to see in isolated snapshots. Survey-then-detail is the natural exploration strategy for large data.

### Full NG state prediction, not just position

The model controls the complete view: position, zoom, orientation, layout, layer visibility, contrast. This matches how a human expert uses Neuroglancer — they don't just scroll Z, they zoom, rotate, toggle layers, adjust contrast.

### Accumulated context, not stateless calls

Each action's result is appended to context. The model sees its full exploration history when deciding the next action. This prevents redundant views and enables reasoning about coverage ("I've scanned the left half, now I should check the right").

### Guardrails, not rigid steps

Instead of forcing a fixed plan→execute→synthesize sequence, the system provides guardrails (max iterations, duplicate detection, format validation) while letting the model navigate freely within them. A confident answer on iteration 3 is better than forcing the model through 15 planned views.

---

## 9. Dependencies (v3 additions)

| Package | Purpose | Notes |
|---------|---------|-------|
| `transformers>=4.57` | Molmo2 inference | Unchanged from v2 |
| `bitsandbytes` | Quantization | Unchanged |
| `accelerate` | device_map | Unchanged |
| `molmo_utils` | Processor helpers | Unchanged |
| `decord2` | Video frame decoding | **Now actually used** for scan/video |
| `torch` | Already installed | |
| `playwright` | Headless screenshots | Unchanged |
| `neuroglancer-chat` | NeuroglancerState | Unchanged |
| `zarr` / `s3fs` | Volume metadata discovery | **New** — read shape from zarr sources |
| `numpy` | Scan position generation, FOV math | Already installed |

---

## 10. Open Questions (to resolve in GPU section)

- What precision / quantization given the target GPU?
- Does video mode (decord2) work with 4-bit quantized models?
- What is the practical max context length (KV cache budget) for a long exploration session?
- Should we use KV cache reuse across the agent loop, or independent calls per action?
- Is Molmo2-O-7B the right model, or does a larger GPU unlock better options?

---

*DRAFT — covers inputs and workflow only. GPU configuration and model selection TBD.*
