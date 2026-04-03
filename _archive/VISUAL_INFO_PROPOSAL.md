# Visual Information Generation: Analysis & Proposal

> Diagnosis of current screenshot quality, evaluation of alternatives, and a concrete plan for maximizing visual information delivered to the model.

---

## 1. What's Wrong Right Now

### 1.1 The screenshots are mostly UI chrome, not data

Examining all 15 screenshots from `data/molmo-glancer-v2-multi-step/screenshots/`:

```
+-----------------------------------------------+
| [toolbar: File, Nav, Icons]          ~30px     |  <- wasted
|----+-----------------------------------------+|
|    |                                         ||
| L  |     Yellow bounding box                 ||
| a  |  +-----------------------------------+  ||
| y  |  |                                   |  ||
| e  |  |   Actual microscopy data          |  ||
| r  |  |   with crosshair overlay          |  ||
|    |  |                                   |  ||
| P  |  +-----------------------------------+  ||
| a  |                                         ||
| n  +-----------------------------------------+|
| e  |  [scale bar]                            ||
| l  +-----------------------------------------+|
+-----------------------------------------------+
 ~200px                  ~1080px
```

**Pixel budget breakdown (1280×720 viewport):**

| Region | Approx pixels | % of image | Useful? |
|--------|--------------|------------|---------|
| Top toolbar | 1280×30 = 38K | 4% | No |
| Left layer panel | 200×690 = 138K | 15% | No |
| Scale bar overlay | ~100×20 = 2K | <1% | Marginal |
| Yellow bounding box border | ~4K | <1% | No — actively harmful |
| Red/green crosshair lines | ~2K | <1% | No — actively harmful |
| Dark margins (outside data) | ~100K | 11% | No |
| **Actual data rendering** | **~640K** | **~69%** | **Yes** |

**~31% of every screenshot is wasted on UI elements the model doesn't need.** Worse, the yellow bounding box and crosshair overlay *on top of* the data, occluding biological structures.

### 1.2 The NG state is underspecified

The current `build_view_urls()` only sets 3 properties per view:

```python
view.data["layout"] = spec.get("layout", "xy")  # always "xy"
pos[0] = spec.get("x", cx)                       # always 247.5
pos[1] = spec.get("y", cy)                       # always 247.5
pos[2] = spec.get("z", cz)                       # varies
```

**Properties available but never used:**

| NG State Key | What it controls | Current value | Impact if set |
|---|---|---|---|
| `showAxisLines` | Red/green crosshair | `true` (default) | Set `false` → clean image |
| `showScaleBar` | Scale bar overlay | `true` (default) | Set `false` → clean image |
| `showDefaultAnnotations` | **Yellow bounding box** | `true` (default) | Set `false` → removes the box |
| `crossSectionScale` | Zoom level (2D views) | unset (auto-fit) | Control zoom for detail vs. overview |
| `projectionScale` | Zoom level (3D view) | unset | Required for meaningful 3D views |
| `projectionOrientation` | 3D camera angle (quaternion) | unset | Enables arbitrary 3D rotations |
| `crossSectionOrientation` | Oblique slice angle (quaternion) | unset | Enables arbitrary slice planes |
| `showSlices` | Cross-section planes in 3D view | `true` (default) | Toggle for cleaner 3D renders |
| `crossSectionBackgroundColor` | Background color | dark gray | Set to black for better contrast |

### 1.3 View planning is degenerate

From the output log and findings:

- **All 15 views use `layout: "xy"`** — never `xz`, `yz`, `3d`, or `4panel`
- **X, Y never change** — always center (247.5, 247.5)
- **Only Z varies** — and the model proposed z values beyond the volume (225, 235, 245, 255, 265) which were clamped to z=214
- **Views 9–14 are identical** (all z=214, same x, y, layout)
- **Guidance text is copy-pasted** across all views — identical checklist
- **Model responses for views 10–14 are word-for-word identical** — classic hallucination on duplicate inputs

The model is only exploring **one axis of one dimension** of the full Neuroglancer state space.

### 1.4 Current view spec schema is too narrow

```python
# Current: model outputs this
{"z": 145}
# or at best
{"x": 100, "y": 200, "z": 145, "layout": "xy"}
```

The model can only specify `x, y, z, layout`. It has no way to request:
- Zoom level changes
- 3D rotation angles
- Layer visibility toggling
- Contrast/brightness adjustments
- Oblique cross-sections
- Multi-panel layouts with specific orientations

---

## 2. Is Screenshotting the Right Approach?

**Yes — but the current implementation leaves most of its potential on the table.** Here's why screenshotting is fundamentally correct, and what the real alternatives are:

### 2.1 Why Neuroglancer rendering is the right source

| Approach | Pros | Cons |
|----------|------|------|
| **NG screenshot** (current) | Full rendering pipeline: shaders, multi-res, segmentation overlays, 3D meshes | Browser overhead, UI chrome |
| Direct zarr slice extraction | No browser, fast, clean images | Loses colormaps, multi-layer compositing, segmentation rendering, 3D views, annotations |
| napari rendering | Python-native, good 3D | Requires reimplementing all view specs, different renderer, no NG state compatibility |
| vtk/pyvista | Full 3D control | Massive reimplementation, different look from NG |

**Neuroglancer is the viewer this system is designed around.** The model predicts NG state, the user provides NG links, and the goal is to answer questions *about what's in Neuroglancer*. Rendering outside NG would disconnect the model's predictions from reality.

### 2.2 Three tiers of screenshot improvement

#### Tier 1: Clean up current Playwright screenshots (easy, high impact)

Set NG state keys to hide overlays before generating URLs:

```python
view.data["showAxisLines"] = False
view.data["showScaleBar"] = False
view.data["showDefaultAnnotations"] = False  # kills the yellow box
view.data["crossSectionBackgroundColor"] = "#000000"
```

Inject CSS via Playwright to hide UI chrome:

```python
page.add_style_tag(content="""
    .neuroglancer-layer-panel { display: none !important; }
    .neuroglancer-viewer-top-row { display: none !important; }
    .neuroglancer-statistics-panel { display: none !important; }
""")
```

Or screenshot just the canvas element:

```python
canvas = page.locator("canvas").first
canvas.screenshot(path="view.png")
```

Replace `time.sleep(12)` with readiness polling:

```python
page.wait_for_function("""
    () => {
        const v = window.viewer;
        return v && v.isReady && v.isReady();
    }
""", timeout=30000)
```

**Expected result: ~100% of pixels are data. No overlays. Faster capture.**

#### Tier 2: Use neuroglancer Python API's built-in screenshot (medium effort, best quality)

The `neuroglancer` Python package has a dedicated screenshot system:

```python
import neuroglancer

viewer = neuroglancer.Viewer()
# ... set state ...
screenshot = viewer.screenshot(size=(1280, 720))
image_array = screenshot.screenshot.image_pixels  # numpy RGBA
```

Or the CLI tool:

```bash
python -m neuroglancer.tool.screenshot \
    --width 1280 --height 720 \
    --hide-axis-lines --hide-default-annotations \
    --cross-section-background-color "#000000" \
    "neuroglancer_state.json" output.png
```

This automatically:
- Hides all UI controls, side panels, panel borders
- Waits for data readiness (no sleep heuristic)
- Supports tiled rendering for ultra-high-res captures
- Uses Selenium + headless Chrome (similar browser overhead to Playwright)

**Tradeoff:** Requires installing selenium + chromedriver alongside or instead of Playwright. The neuroglancer package may need to be installed from source (it's in `code/lib/neuroglancer/` if present, or from PyPI).

#### Tier 3: Direct zarr slice rendering for simple cross-sections (supplementary)

For pure 2D cross-sections (xy, xz, yz), the data can be loaded and rendered directly:

```python
import zarr
import numpy as np
from PIL import Image

store = zarr.open("s3://aind-open-data/.../ch_405.zarr", mode="r")
arr = store[0]  # full resolution level

# Extract XY slice at z=145
slice_data = arr[0, 145, :, :]  # shape: (495, 495)

# Apply contrast normalization (matching NG shader range [0, 400])
normalized = np.clip(slice_data / 400.0, 0, 1)
img = Image.fromarray((normalized * 255).astype(np.uint8), mode="L")
```

**Pros:** Zero browser overhead, instant, pixel-perfect, no UI artifacts
**Cons:** No 3D views, no segmentation overlays, no multi-layer compositing, no shader effects, need to reimplement colormap logic

**Best used as a supplement** — for rapid survey slices where full NG rendering isn't needed.

---

## 3. The Full NG State Prediction Space

The current pipeline only predicts `{x, y, z, layout}`. Here's the complete space the model *should* be able to predict:

### 3.1 Proposed view spec schema

```jsonc
{
    // Position (required)
    "x": 247.5,
    "y": 247.5,
    "z": 145.0,

    // Layout (required) — what panels to show
    "layout": "xy",
    // Options: "xy", "xz", "yz", "3d", "4panel", "4panel-alt",
    //          "xy-3d", "xz-3d", "yz-3d"

    // Zoom (optional)
    "crossSectionScale": 1.0,    // 2D zoom: <1 = zoom in, >1 = zoom out
    "projectionScale": 4096,     // 3D zoom: distance in voxels

    // 3D orientation (optional) — quaternion [x, y, z, w]
    "projectionOrientation": [0.3, 0.1, 0.0, 0.95],

    // Oblique slicing (optional) — quaternion
    "crossSectionOrientation": [0, 0, 0, 1],

    // Layer control (optional)
    "layerVisibility": {
        "channel_405": true,
        "segmentation": false
    },

    // Contrast (optional)
    "shaderRange": [0, 400],     // invlerp normalized range

    // Display hints (optional)
    "showSlices": false,         // for 3D views: hide slice planes
    "orthographic": true         // for 3D views: orthographic vs perspective
}
```

### 3.2 What this unlocks

| Capability | Current | Proposed | Why it matters |
|---|---|---|---|
| Navigate XY plane | Yes | Yes | Baseline |
| Navigate XZ/YZ planes | In theory (layout key) | Yes, with zoom | Orthogonal views show different structures |
| 3D overview | No | Yes (projectionScale + orientation) | Global structure, spatial relationships |
| Zoom in for detail | No | Yes (crossSectionScale) | Fine structures: dendrites, synapses |
| Zoom out for context | No | Yes (crossSectionScale) | Volume-wide patterns, clustering |
| Oblique slicing | No | Yes (crossSectionOrientation) | Cut along structures, not just axes |
| Toggle layers | No | Yes (layerVisibility) | Isolate channels, show segmentation |
| Adjust contrast | No | Yes (shaderRange) | Reveal dim structures, reduce saturation |
| Multi-panel views | No | Yes (4panel, xy-3d, etc.) | Simultaneous context |

### 3.3 Example view plans the model should be able to produce

**For "How many neurons can you count in this volume?":**

```json
[
    {"x": 247, "y": 247, "z": 107, "layout": "xy", "crossSectionScale": 1.0,
     "purpose": "overview of center XY slice"},

    {"x": 247, "y": 247, "z": 107, "layout": "xy", "crossSectionScale": 0.3,
     "purpose": "zoom into center to count individual neurons"},

    {"x": 100, "y": 100, "z": 107, "layout": "xy", "crossSectionScale": 0.3,
     "purpose": "zoom into corner to check neuron density variation"},

    {"x": 247, "y": 247, "z": 107, "layout": "xz",
     "purpose": "orthogonal XZ view to verify 3D neuron extent"},

    {"x": 247, "y": 247, "z": 107, "layout": "3d",
     "projectionScale": 600, "projectionOrientation": [0.5, 0.1, 0.0, 0.85],
     "purpose": "3D overview to see spatial distribution"},

    {"x": 247, "y": 247, "z": 50, "layout": "xy", "crossSectionScale": 0.5,
     "purpose": "sample different Z depth for counting"},

    {"x": 400, "y": 400, "z": 107, "layout": "xy", "crossSectionScale": 0.5,
     "purpose": "sample edge region for density comparison"}
]
```

Compare to what the model actually produced: 15 views at center XY with only Z varying by 10 per step.

---

## 4. Proposed Architecture

### 4.1 Screenshot pipeline (revised)

```
Model predicts full view spec
        |
        v
+------------------------+
| build_clean_state()    |  Set showAxisLines=false, showDefaultAnnotations=false,
| (NeuroglancerState)    |  showScaleBar=false, crossSectionScale, projectionScale,
|                        |  projectionOrientation, layer visibility, shader range
+------------------------+
        |
        v
+------------------------+
| take_screenshot()      |  Playwright with CSS injection to hide remaining UI,
| (Playwright)           |  canvas-element-only screenshot, readiness polling
+------------------------+
        |
        v
+------------------------+
| Optional: crop/pad     |  If canvas aspect ratio doesn't match viewport,
|                        |  center-crop to data region
+------------------------+
        |
        v
    Clean PNG (100% data, no UI artifacts)
```

### 4.2 Implementation: `build_clean_state()`

```python
def build_clean_state(base_state: NeuroglancerState, spec: dict) -> NeuroglancerState:
    view = base_state.clone()

    # Position
    pos = view.data["position"]
    pos[0] = spec.get("x", cx)
    pos[1] = spec.get("y", cy)
    pos[2] = spec.get("z", cz)

    # Layout
    view.data["layout"] = spec.get("layout", "xy")

    # Zoom
    if "crossSectionScale" in spec:
        view.data["crossSectionScale"] = spec["crossSectionScale"]
    if "projectionScale" in spec:
        view.data["projectionScale"] = spec["projectionScale"]

    # Orientation
    if "projectionOrientation" in spec:
        view.data["projectionOrientation"] = spec["projectionOrientation"]
    if "crossSectionOrientation" in spec:
        view.data["crossSectionOrientation"] = spec["crossSectionOrientation"]

    # Layer visibility
    for layer_name, visible in spec.get("layerVisibility", {}).items():
        view.set_layer_visibility(layer_name, visible)

    # Contrast
    if "shaderRange" in spec:
        for L in view.data.get("layers", []):
            if L.get("type") == "image":
                L.setdefault("shaderControls", {}).setdefault(
                    "normalized", {})["range"] = spec["shaderRange"]

    # Clean display — hide all overlays
    view.data["showAxisLines"] = False
    view.data["showScaleBar"] = False
    view.data["showDefaultAnnotations"] = False
    if spec.get("layout") == "3d" or spec.get("layout", "").endswith("-3d"):
        view.data["showSlices"] = spec.get("showSlices", False)
    view.data["crossSectionBackgroundColor"] = "#000000"

    # Hide side panels via state (close selected layer, stats)
    view.data["selectedLayer"] = {"visible": False}
    view.data["statistics"] = {"visible": False}

    return view
```

### 4.3 Implementation: clean Playwright capture

```python
def take_clean_screenshot(page, url: str, timeout: int = 30000) -> Image.Image:
    page.goto(url, wait_until="domcontentloaded")

    # Hide remaining UI chrome via CSS injection
    page.add_style_tag(content="""
        .neuroglancer-viewer-top-row { display: none !important; }
        .neuroglancer-layer-panel { display: none !important; }
        .neuroglancer-layer-side-panel { display: none !important; }
        .neuroglancer-statistics-panel { display: none !important; }
        .neuroglancer-panel-border { display: none !important; }
    """)

    # Wait for data readiness instead of sleeping
    try:
        page.wait_for_function("""
            () => {
                const v = window.viewer;
                return v && typeof v.isReady === 'function' && v.isReady();
            }
        """, timeout=timeout)
    except:
        # Fallback: wait fixed time if viewer API not available
        import time
        time.sleep(12)

    # Screenshot just the canvas — no UI chrome
    canvas = page.locator("canvas").first
    png_bytes = canvas.screenshot()
    return Image.open(BytesIO(png_bytes)).convert("RGB")
```

### 4.4 What about Neuroglancer's Python screenshot API?

The `neuroglancer` Python package has `viewer.screenshot(size=(W,H))` which automatically hides all UI and waits for readiness. It also has a CLI tool (`python -m neuroglancer.tool.screenshot`) with all the right flags built in.

**However, it requires:**
- Selenium + chromedriver (instead of Playwright)
- Running a local neuroglancer server (Tornado-based)
- The `neuroglancer` package installed from source or PyPI

**Recommendation:** The Playwright approach with CSS injection and state-level overlay hiding achieves the same result with fewer dependencies and is more consistent with the current stack. The neuroglancer screenshot API is a better long-term choice if the project moves to a local NG server setup.

---

## 5. Impact Assessment

### 5.1 Pixel efficiency

| Config | Data pixels | UI/overlay pixels | Useful % |
|--------|-----------|-----------------|----------|
| **Current** (raw Playwright screenshot) | ~640K | ~282K | ~69% |
| **+ NG state overlay hiding** (no crosshair/box/scalebar) | ~750K | ~172K | ~81% |
| **+ CSS injection** (hide toolbar/sidebar) | ~900K | ~22K | ~98% |
| **+ Canvas-only screenshot** | ~922K | 0 | **~100%** |

### 5.2 Vision token efficiency

With the image downscaled to 512px max side before model inference:

| Config | Useful visual content in tokens | Wasted tokens on UI |
|--------|-------------------------------|-------------------|
| Current (512px, ~69% data) | ~470 of ~680 | ~210 (~31%) |
| Clean screenshot (512px, ~100% data) | ~680 of ~680 | ~0 |
| Clean + full res (1280px, 8 crops) | ~1,521 of ~1,521 | ~0 |

**Cleaning up screenshots effectively gives 31% more visual information at zero additional VRAM cost.**

### 5.3 View diversity impact

The current run produced 15 views with only Z varying. A richer view spec would enable:

- **3× spatial coverage** — varying X, Y in addition to Z
- **3× orientation coverage** — XY, XZ, YZ planes
- **Zoom stratification** — overview + detail views
- **3D context** — spatial relationships between structures

Conservatively, this could improve answer quality as much as improving the model itself, since the current system is *information-starved* by its own view planning.

---

## 6. Prioritized Implementation Plan

### Phase 1: Quick wins (no architecture change)

1. **Add overlay-hiding keys to every generated state** — `showAxisLines`, `showScaleBar`, `showDefaultAnnotations` all set to `false`
2. **CSS injection** in Playwright to hide toolbar and side panel
3. **Canvas-element screenshot** instead of full-page screenshot
4. **Readiness polling** via `viewer.isReady()` instead of `time.sleep(12)`
5. **Set `crossSectionBackgroundColor: "#000000"`** for better contrast

**Effort: ~30 minutes. Impact: 31% more useful pixels, faster capture, no overlays.**

### Phase 2: Expand view spec schema

1. **Add `crossSectionScale`** to view spec — enable zoom control
2. **Add `projectionScale` + `projectionOrientation`** — enable 3D views
3. **Add `layerVisibility`** — enable layer toggling
4. **Add `shaderRange`** — enable contrast control
5. **Update the model prompt** to explain and request these new parameters
6. **Add view spec validation** — clamp coordinates to volume bounds, validate quaternions

**Effort: ~2 hours. Impact: Full NG state prediction capability.**

### Phase 3: Improve view planning quality

1. **Inject volume metadata** into the planning prompt — bounds, resolution, layer names
2. **Require diverse layouts** — prompt the model to use at least 2 different layouts
3. **Require spatial diversity** — enforce minimum distance between view positions
4. **Deduplicate views** — detect and remove identical/near-identical specs before screenshotting
5. **Adaptive view count** — start with fewer views, add more if findings are ambiguous

**Effort: ~1 day. Impact: Eliminates degenerate view plans like the 6 identical z=214 views.**

### Phase 4: Consider supplementary rendering (optional)

1. **Direct zarr slice rendering** for rapid spatial survey (no browser)
2. **Maximum intensity projections (MIPs)** along each axis for quick overview
3. **Neuroglancer Python screenshot API** if moving to local server setup

**Effort: ~1 day. Impact: Faster survey phase, potential hybrid approach.**

---

## 7. Summary

| Problem | Root cause | Fix |
|---------|-----------|-----|
| 31% of pixels wasted on UI | NG overlay defaults + full-page screenshot | State keys + CSS injection + canvas screenshot |
| Yellow box obscures data | `showDefaultAnnotations` defaults to `true` | Set to `false` in state |
| Crosshair obscures data | `showAxisLines` defaults to `true` | Set to `false` in state |
| 12s fixed sleep per view | No readiness detection | Poll `viewer.isReady()` |
| Only XY layout used | View spec schema too narrow | Add layout, zoom, orientation, 3D params |
| Only Z varies across views | Model prompt doesn't encourage diversity | Richer prompts, validation, dedup |
| 6 identical views at z=214 | No deduplication, no bounds clamping feedback | Validate specs, deduplicate before screenshot |
| Model hallucinates on duplicate images | Duplicate inputs → duplicate outputs | Prevent duplicate views from being generated |

**Screenshotting Neuroglancer is the right approach.** The problem isn't the screenshot mechanism — it's that the current implementation captures a cluttered UI at low diversity when it could be capturing clean, information-dense renders across the full NG state space.

---

*Generated 2026-04-01. Based on analysis of 15 screenshots from `data/molmo-glancer-v2-multi-step/`, the NeuroglancerState API, Neuroglancer's state JSON schema, and the Playwright screenshot pipeline in `run_capsule.py`.*
