# Addendum: Arbitrary Data Shapes & Scale

> How the proposed screenshot architecture must adapt when data is no longer 495×495×215 isotropic voxels, but arbitrary shape and potentially very large.

---

## 1. How Neuroglancer Zoom Actually Works

### crossSectionScale = canonical voxels per screen pixel

At `crossSectionScale = 1.0` (the default), each screen pixel shows 1 canonical voxel. The visible field of view (FOV) is:

```
FOV_horizontal = viewport_width  × crossSectionScale  (in canonical voxels)
FOV_vertical   = viewport_height × crossSectionScale  (in canonical voxels)
```

For a **1024×1024 viewport at scale 1.0**: you see **1024×1024 voxels** of data.

**Critically: changing viewport size changes what is rendered.** A larger viewport at the same scale shows more data — it's not just a stretch of the same image. However, we fix the viewport at 1024×1024 (see §3.1) and use `crossSectionScale` as the sole zoom control.

### What this means for different data sizes

| Dataset XY shape | crossSectionScale to fit in 1024px viewport | Voxels per pixel | Detail visible? |
|---|---|---|---|
| 495×495 (current) | 0.48 | <1 (oversampled) | Excellent |
| 1024×1024 | 1.0 | 1:1 | Good |
| 5000×5000 | 4.9 | ~5 | OK for overview |
| 50,000×50,000 | 49 | ~49 | No useful detail |
| 500,000×200,000 | 488 | ~488 | Completely useless |

For the current 495×495 data, **the default view already oversamples** — each voxel gets ~2 pixels. This is why the screenshots look fine without explicit zoom control. For anything larger than ~1024 voxels across, detail starts degrading.

### Anisotropic voxels distort the view

Neuroglancer preserves physical proportions using `canonicalVoxelFactors`:

```
canonicalVoxelFactor[i] = voxelPhysicalScale[i] / min(voxelPhysicalScales)
```

Example: x=1μm, y=1μm, z=5μm → factors [1, 1, 5].

In an XZ cross-section at scale 1.0 in a 1024×1024 viewport:
- X axis: see 1024 voxels (1024 × 1.0 / 1)
- Z axis: see only 205 voxels (1024 × 1.0 / 5) — each Z voxel spans 5 pixels

**Result:** Highly anisotropic data in XZ/YZ views shows less Z coverage per screenshot. The square viewport helps (vs 16:9 which would show even fewer Z voxels), but the model still needs to account for anisotropy when planning view positions.

### Multi-resolution pyramid switching

Neuroglancer automatically switches to coarser pyramid levels when voxels become sub-pixel:

```
Switch when: source_voxel_size > pixel_physical_size × renderScaleTarget × 1.1
```

At default `renderScaleTarget=1.0`, NG switches to a 2× coarser level when `crossSectionScale ≈ 2.0`, to 4× at `≈ 4.0`, etc. This means zoomed-out views of large data use lower-resolution pyramid levels automatically — no VRAM explosion, but also no fine detail.

---

## 2. Problems for Arbitrary Data

### Problem 1: Data doesn't fit the viewport

With a fixed 1024×1024 viewport, arbitrary data at default zoom (scale=1.0):

| Data XY shape | At default zoom (scale=1.0) | Situation |
|---|---|---|
| 495×495 | Fits — centered, oversampled ~2× | Good (current case) |
| 1024×1024 | Fills viewport exactly | Ideal |
| 100×5000 | Center 1024×1024 visible | Use crossSectionScale to zoom/scan |
| 50000×50000 | Center 1024×1024 visible (0.04%) | Must zoom out or scan |
| 200×200 | Centered, oversampled ~5× | Good — small data fits easily |

### Problem 2: No single zoom level works for large data

For a 50,000×50,000 dataset, the model must choose:
- **Overview (scale=39):** See the whole dataset but each pixel is ~39 voxels — no individual features visible
- **Detail (scale=0.5):** See individual voxels clearly but only a 640×360 voxel window — 0.0001% of the data

The model needs to operate at **multiple scales**, not just one.

### Problem 3: Position + zoom must be coupled

When zoomed in on a large dataset, `position` determines which tiny fraction you see. The model needs to understand:
- At scale S, positioned at (px, py), the visible window is [px - 512·S, px + 512·S] × [py - 512·S, py + 512·S] (for isotropic data in the 1024×1024 viewport)
- Moving by more than 512·S produces a completely different view
- Planning adjacent non-overlapping views requires knowing the FOV

### Problem 4: Anisotropic data needs per-axis awareness

With x=8nm, y=8nm, z=40nm voxels:
- XY views are fine (isotropic in-plane)
- XZ views show Z stretched 5× — the data looks distorted
- The model might misinterpret structure orientation in stretched views

---

## 3. Architectural Changes Required

### 3.1 Fixed square viewport (1024×1024)

**Use a fixed 1024×1024 square viewport for all views.** Handle arbitrary data shapes through `crossSectionScale` (zoom), not viewport reshaping.

Rationale:
- **No directional bias.** Cross-sections show equal extent in both displayed axes — no arbitrary preference for horizontal over vertical.
- **Square crops tile cleanly.** Molmo2 tiles images into 378×378 crops. A square image produces symmetric grids (2×2=4, 3×3=9) with minimal waste.
- **Consistent vision tokens.** Every screenshot has the same shape → predictable crop tiling → predictable token counts.
- **Stays in training distribution.** Extreme aspect ratios (10:1 for elongated data) would produce unusual crop layouts the model may not handle well.
- **crossSectionScale is the zoom knob.** For any data shape, the model zooms to fill the viewport. A 50000×200 dataset doesn't need a wide viewport — it needs a scan along the long axis at appropriate zoom.

```python
VIEWPORT = {"width": 1024, "height": 1024}
context = browser.new_context(viewport=VIEWPORT)
```

### 3.2 Scale-aware view planning

The model needs to be told the data extent and how zoom works. The planning prompt should include:

```
Volume extent: 50000 × 50000 × 2000 voxels (x × y × z)
Voxel size: 8nm × 8nm × 40nm (anisotropic, z is 5× coarser)
Viewport: 1024×1024 (fixed square)

At crossSectionScale=1.0, you see 1024×1024 voxels (an 8.19μm × 8.19μm window).
At crossSectionScale=49, you see the full 50000 voxel extent but each pixel covers 49 voxels.
At crossSectionScale=0.5, you see 512×512 voxels at 2× oversampling (high detail).

When planning views, specify BOTH position AND crossSectionScale.
The visible window at position (px, py) and scale s is:
  x: [px - 640·s, px + 640·s]
  y: [py - 360·s, py + 360·s]
```

### 3.3 Multi-scale scanning strategy

For large datasets, view planning should follow a **survey → region of interest → detail** pattern:

```
Level 1: SURVEY (1-3 views)
  - crossSectionScale high enough to see entire dataset extent
  - Purpose: identify regions of interest, overall structure

Level 2: REGION (3-8 views)
  - Medium zoom on each region of interest identified in survey
  - Spread across different positions and orientations
  - Purpose: characterize regions, find specific features

Level 3: DETAIL (2-5 views per region)
  - Low crossSectionScale for high detail
  - Focused on specific structures found in region views
  - Purpose: count, measure, classify specific features
```

This can be implemented as iterative loop passes rather than a single batch of views:

```python
# Pass 1: Survey
survey_specs = model.plan_survey(volume_info)
survey_images = take_screenshots(survey_specs)
survey_findings = model.interpret(survey_images)

# Pass 2: Regions of interest
roi_specs = model.plan_roi(survey_findings, volume_info)
roi_images = take_screenshots(roi_specs)
roi_findings = model.interpret(roi_images)

# Pass 3: Detail (optional)
if model.needs_more_detail(roi_findings):
    detail_specs = model.plan_detail(roi_findings, volume_info)
    ...
```

### 3.4 FOV-aware position planning

The model (or a validation layer) must ensure views don't overlap excessively and do cover the needed area:

```python
VIEWPORT_SIZE = 1024  # fixed square

def compute_fov(scale, canonical_factors=(1, 1)):
    """Compute visible voxel range at given scale (square viewport)."""
    fov_x = VIEWPORT_SIZE * scale / canonical_factors[0]
    fov_y = VIEWPORT_SIZE * scale / canonical_factors[1]
    return fov_x, fov_y

def views_overlap(spec_a, spec_b, threshold=0.8):
    """Check if two views show >threshold fraction of the same area."""
    fov_x, fov_y = compute_fov(spec_a["crossSectionScale"])
    # ... compute intersection of the two FOV rectangles
    overlap_area = ...
    return overlap_area / (fov_x * fov_y) > threshold
```

### 3.5 Expanded view spec schema (revised for scale)

```jsonc
{
    // Position (required)
    "x": 25000, "y": 25000, "z": 1000,

    // Layout (required)
    "layout": "xy",

    // Zoom (now critical, not optional)
    "crossSectionScale": 10.0,

    // Computed by system, returned to model for awareness:
    "_fov_voxels": [10240, 10240],
    "_fov_physical": [81.9, 81.9],  // μm
    "_visible_window": [[19880, 30120], [19880, 30120]],

    // Other fields as before...
    "projectionScale": 4096,
    "projectionOrientation": [0.3, 0.1, 0.0, 0.95],
    "layerVisibility": {"ch_405": true},
    "shaderRange": [0, 400]
}
```

The `_fov_*` and `_visible_window` fields are computed by the system and fed back to the model in subsequent planning rounds, so it knows what it has already seen and what gaps remain.

### 3.6 Volume metadata discovery

The pipeline needs to read volume metadata before planning. This should happen at startup:

```python
def discover_volume_info(ng_state: dict) -> dict:
    """Extract volume metadata from NG state and data sources."""
    dims = ng_state.get("dimensions", {})
    voxel_scales = {k: v[0] for k, v in dims.items() if isinstance(v, list)}
    units = {k: v[1] for k, v in dims.items() if isinstance(v, list)}

    # Read bounds from zarr metadata
    # zarr .zarray has "shape", .zattrs has physical metadata
    shape = read_zarr_shape(ng_state["layers"][0]["source"])

    return {
        "shape_voxels": shape,           # e.g., (50000, 50000, 2000)
        "voxel_scales": voxel_scales,    # e.g., {"x": 8e-9, "y": 8e-9, "z": 4e-8}
        "units": units,                  # e.g., {"x": "m", "y": "m", "z": "m"}
        "physical_extent": {k: shape[i] * voxel_scales[k]
                           for i, k in enumerate(voxel_scales)},
        "is_anisotropic": max(voxel_scales.values()) / min(voxel_scales.values()) > 1.5,
        "anisotropy_ratio": max(voxel_scales.values()) / min(voxel_scales.values()),
        "canonical_voxel_factors": compute_canonical_factors(voxel_scales),
    }
```

---

## 4. How Playwright Architecture Changes

### Before (current)

```
Fixed viewport (1280×720, 16:9)
    → Fixed state (position only, no zoom)
        → Full-page screenshot with UI chrome
            → Downscale to 512px
                → Model
```

Works for the current 495×495 dataset by coincidence (data happens to fit).

### After (arbitrary data)

```
Read volume metadata (shape, voxel scales, anisotropy)
    → Fixed square viewport (1024×1024)
        → Model plans multi-scale views (survey → ROI → detail)
            → Each view has position + crossSectionScale + layout
                → build_clean_state() adds zoom, hides overlays
                    → Playwright canvas-only screenshot (1024×1024)
                        → System computes FOV, feeds back to model
                            → Model plans next round if needed
```

### Specific Playwright changes

| Aspect | Current | Needed |
|--------|---------|--------|
| Viewport size | Fixed 1280×720 (16:9) | Fixed 1024×1024 (square) |
| browser.new_context | Called once | Called once (same viewport for all views) |
| crossSectionScale | Not set (defaults to 1.0) | Explicitly set per view |
| Position validation | Clamp to data bounds | Clamp position ± FOV/2 to data bounds |
| Screenshot target | `page.screenshot()` | `page.locator("canvas").screenshot()` |
| Wait strategy | `time.sleep(12)` | `viewer.isReady()` poll |
| State keys | position, layout | + crossSectionScale, projectionScale, projectionOrientation, display flags |

### Viewport resizing between shots

The viewport is set once at session start and remains 1024×1024 for all views. No resizing between shots — the model adapts to different data shapes through `crossSectionScale` and position, not viewport manipulation.

---

## 5. Edge Cases & Data Shapes

### Very long/thin data (e.g., 200,000 × 500 × 500 — a spinal cord)

- XY view: 500×500 square, fits fine at default zoom
- XZ view: 200,000 × 500 — at scale 1.0 you see only 1024 of 200,000 voxels (0.5%)
- Need: high crossSectionScale for overview (~156), then zoom in to scan along length
- Viewport: wide & short for XZ overview, square for XY

### Very large isotropic cube (e.g., 100,000³ — whole brain)

- At scale 1.0: see 1024×1024 voxels of 100,000³ = 0.0001%
- Need: multi-level scanning. Survey at scale ~78 (whole brain in view), then drill down
- 3D projection view with `projectionScale` ≈ 100,000 for overview

### Highly anisotropic (e.g., 1000 × 1000 × 50, z-voxels 20× larger)

- XY view: fine (1000×1000 isotropic in-plane)
- XZ view: 1000 × 50 voxels, but z is stretched 20× → appears as 1000 × 1000 physical
- The model needs to know that the XZ "tall" appearance is from anisotropy, not actual extent
- Volume metadata should include anisotropy ratio and physical vs voxel dimensions

### Multi-channel / multi-layer (many overlaid sources)

- Layer visibility becomes critical — model may need to isolate channels
- Different channels may have different optimal contrast ranges
- 3D rendering with multiple layers needs careful shader control

### Tiled / multi-tile datasets

- Single NG link may span multiple zarr tiles
- Position space is the physical space, not per-tile
- Scale is global, not per-tile

---

## 6. Summary of Changes to Original Proposal

The original proposal's Phase 1-2 (clean screenshots, expanded view spec) are still correct. The key additions for arbitrary data:

| Original assumption | Reality | Required change |
|---|---|---|
| Data fits in viewport at default zoom | Data may be orders of magnitude larger | crossSectionScale is mandatory, not optional |
| Fixed 1280×720 viewport (16:9) | Square viewport eliminates directional bias | Fixed 1024×1024 square viewport |
| Single-pass view planning | Large data requires multi-scale exploration | Survey → ROI → detail loop |
| Model doesn't need to know data extent | Model must understand FOV at each zoom level | Volume metadata in planning prompt |
| Position is the only spatial parameter | Position + scale together define visible window | FOV computation and feedback to model |
| Isotropic voxels assumed | Anisotropic voxels distort views | Canonical voxel factors in metadata |

The core architecture (Playwright + NeuroglancerState + canvas screenshots) remains valid. The changes are about **what gets computed before the screenshot** (viewport, scale, position validation) and **what the model knows** (volume extent, FOV at each zoom, anisotropy).

---

*Generated 2026-04-01. Addendum to VISUAL_INFO_PROPOSAL.md.*
