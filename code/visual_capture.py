"""
visual_capture — Playwright-based clean state builder and screenshot capture.

Handles:
- Building clean NG states (overlay hiding, view spec application)
- CSS injection to hide remaining UI chrome
- Canvas-only screenshot capture with readiness polling
- Scan frame generation (video sweeps)
"""

import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

VIEWPORT_SIZE = 1024
RESULTS_DIR = Path("/results")

# CSS to hide all Neuroglancer UI chrome
NG_HIDE_CSS = """
    .neuroglancer-viewer-top-row { display: none !important; }
    .neuroglancer-layer-panel { display: none !important; }
    .neuroglancer-layer-side-panel { display: none !important; }
    .neuroglancer-statistics-panel { display: none !important; }
    .neuroglancer-layer-group-viewer-top-row { display: none !important; }
    #neuroglancer-container > div > div:first-child { display: none !important; }
"""

# JS for readiness polling
NG_READY_JS = """() => {
    const v = window.viewer;
    return v && typeof v.isReady === 'function' && v.isReady();
}"""


def build_clean_state(base_state, view_spec: dict, volume_info=None):
    """Apply view spec + overlay hiding to an NG state. Returns a new NeuroglancerState.

    view_spec keys:
        x, y, z                   — position
        layout                    — 'xy', 'xz', 'yz', '3d', '4panel', etc.
        crossSectionScale         — 2D zoom (<1 = zoom in, >1 = zoom out)
        projectionScale           — 3D zoom
        projectionOrientation     — [x, y, z, w] quaternion
        crossSectionOrientation   — oblique slice quaternion
        layerVisibility           — {"layer_name": bool, ...}
        shaderRange               — [vmin, vmax] for image layers
    """
    state = base_state.clone()
    d = state.data

    # ── Overlay hiding ──────────────────────────────────────────────────
    d["showAxisLines"] = False
    d["showScaleBar"] = True             # keep scale bar for physical size reference
    d["showDefaultAnnotations"] = True   # yellow bounding box — shows data extent
    d["crossSectionBackgroundColor"] = "#000000"
    d["selectedLayer"] = {"visible": False}
    d["statistics"] = {"visible": False}

    # ── Position ────────────────────────────────────────────────────────
    num_dims = len(d.get("dimensions", {}))
    if "position" not in d or not d["position"]:
        d["position"] = [0.0] * num_dims
    while len(d["position"]) < num_dims:
        d["position"].append(0.0)

    if "x" in view_spec:
        d["position"][0] = float(view_spec["x"])
    if "y" in view_spec:
        d["position"][1] = float(view_spec["y"])
    if "z" in view_spec:
        d["position"][2] = float(view_spec["z"])

    # Clamp position to volume bounds if volume_info is available
    if volume_info is not None:
        for i in range(min(3, len(d["position"]))):
            d["position"][i] = max(0, min(d["position"][i], volume_info.shape[i] - 1))

    # ── Layout ──────────────────────────────────────────────────────────
    if "layout" in view_spec:
        d["layout"] = view_spec["layout"]

    # ── Zoom ────────────────────────────────────────────────────────────
    if "crossSectionScale" in view_spec:
        d["crossSectionScale"] = float(view_spec["crossSectionScale"])
    elif volume_info is not None:
        # Default: 2x fit — data fills ~half the viewport, shows some surrounding context
        fit_scale = max(volume_info.shape[0], volume_info.shape[1]) / VIEWPORT_SIZE
        d["crossSectionScale"] = fit_scale * 2
    if "projectionScale" in view_spec:
        d["projectionScale"] = float(view_spec["projectionScale"])

    # ── Orientation ─────────────────────────────────────────────────────
    if "projectionOrientation" in view_spec:
        d["projectionOrientation"] = [float(v) for v in view_spec["projectionOrientation"]]
    if "crossSectionOrientation" in view_spec:
        d["crossSectionOrientation"] = [float(v) for v in view_spec["crossSectionOrientation"]]

    # ── Layer visibility ────────────────────────────────────────────────
    if "layerVisibility" in view_spec:
        for layer in d.get("layers", []):
            name = layer.get("name", "")
            if name in view_spec["layerVisibility"]:
                layer["visible"] = view_spec["layerVisibility"][name]

    # ── Shader range (contrast) ─────────────────────────────────────────
    if "shaderRange" in view_spec:
        vmin, vmax = view_spec["shaderRange"]
        for layer in d.get("layers", []):
            if layer.get("type") == "image":
                sc = layer.setdefault("shaderControls", {})
                sc.setdefault("normalized", {})["range"] = [vmin, vmax]

    return state


def capture_screenshot(page, state, config: dict, screenshot_id: int) -> Image.Image:
    """Navigate to an NG state URL, wait for data, and capture a clean canvas screenshot.

    Args:
        page: Playwright page object (already created with correct viewport).
        state: NeuroglancerState object with clean view settings.
        config: GPU profile config dict (for max_image_side).
        screenshot_id: Sequential ID for saving the PNG.

    Returns:
        PIL Image of the canvas.
    """
    url = state.to_url()
    print(f"  Navigating to NG URL ({len(url)} chars) ...")

    page.goto(url, wait_until="networkidle")

    # Inject CSS to hide UI chrome
    page.add_style_tag(content=NG_HIDE_CSS)

    # Wait for viewer to signal data is rendered
    try:
        page.wait_for_function(NG_READY_JS, timeout=15000)
        print("  viewer.isReady() = true")
    except Exception:
        print("  WARNING: viewer.isReady() timed out after 15s, falling back to sleep")
        time.sleep(3)

    # Capture canvas only
    canvas = page.locator("canvas").first
    png_bytes = canvas.screenshot()
    img = Image.open(BytesIO(png_bytes)).convert("RGB")

    # Downscale on T4 if needed
    max_side = config.get("max_image_side")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)

    # Save to results
    screenshot_dir = RESULTS_DIR / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    png_path = screenshot_dir / f"view_{screenshot_id:03d}.png"
    img.save(png_path)
    print(f"  Screenshot saved: {png_path} ({img.size[0]}x{img.size[1]})")

    return img


def create_browser(playwright):
    """Create a Playwright browser + page with 1024x1024 viewport."""
    browser = playwright.chromium.launch(
        headless=True,
        args=["--disable-blink-features=AutomationControlled"],
    )
    context = browser.new_context(
        viewport={"width": VIEWPORT_SIZE, "height": VIEWPORT_SIZE},
    )
    page = context.new_page()
    return browser, page


# ── Scan Frame Generation ───────────────────────────────────────────────────

def execute_scan(page, base_state, scan_spec: dict, volume_info, config: dict, scan_id: int) -> list[Image.Image]:
    """Execute a scan: generate frames by sweeping through positions.

    Args:
        page: Playwright page (reused from screenshot capture).
        base_state: NeuroglancerState to use as template.
        scan_spec: Dict with scan_type, start, end, frames, layout, crossSectionScale, etc.
        volume_info: VolumeInfo for bounds clamping.
        config: GPU profile config.
        scan_id: Sequential ID for naming the video file.

    Returns:
        List of PIL Images (one per frame).
    """
    scan_type = scan_spec.get("scan_type", "z_sweep")
    num_frames = min(scan_spec.get("frames", 20), config["max_scan_frames"])
    layout = scan_spec.get("layout", "xy")
    cross_section_scale = scan_spec.get("crossSectionScale", 1.0)

    positions = generate_scan_positions(scan_spec, volume_info, num_frames)
    print(f"  Scan {scan_id}: {scan_type}, {len(positions)} frames, layout={layout}")

    frames = []
    for i, pos in enumerate(positions):
        view_spec = {
            "x": pos[0], "y": pos[1], "z": pos[2],
            "layout": layout,
            "crossSectionScale": cross_section_scale,
        }
        # Carry through optional orientation
        if "projectionOrientation" in scan_spec:
            # For rotation scans, interpolate per frame
            if scan_type == "rotation" and "orientations" in scan_spec:
                view_spec["projectionOrientation"] = scan_spec["orientations"][i]
            else:
                view_spec["projectionOrientation"] = scan_spec["projectionOrientation"]

        state = build_clean_state(base_state, view_spec, volume_info)

        if i == 0:
            # First frame: full navigation
            url = state.to_url()
            page.goto(url, wait_until="networkidle")
            page.add_style_tag(content=NG_HIDE_CSS)
            try:
                page.wait_for_function(NG_READY_JS, timeout=10000)
            except Exception:
                time.sleep(2)
        else:
            # Subsequent frames: hash-fragment update (faster, data cached)
            state_json = json.dumps(state.data, separators=(",", ":"))
            page.evaluate("(h) => { location.hash = '!' + h }", state_json)
            try:
                page.wait_for_function(NG_READY_JS, timeout=3000)
            except Exception:
                time.sleep(0.5)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"    frame {i+1}/{len(positions)}")

        canvas = page.locator("canvas").first
        png_bytes = canvas.screenshot()
        img = Image.open(BytesIO(png_bytes)).convert("RGB")

        # Downscale on T4 if needed
        max_side = config.get("max_image_side")
        if max_side and max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)

        frames.append(img)

    # Save video artifact
    save_scan_video(frames, scan_id)

    return frames


def generate_scan_positions(scan_spec: dict, volume_info, num_frames: int) -> np.ndarray:
    """Generate interpolated positions for a scan.

    Returns array of shape (num_frames, 3) with [x, y, z] per frame.
    """
    shape = volume_info.shape

    start = scan_spec.get("start", {})
    end = scan_spec.get("end", {})

    # Default center
    cx, cy, cz = shape[0] / 2, shape[1] / 2, shape[2] / 2

    start_pos = np.array([
        start.get("x", cx), start.get("y", cy), start.get("z", cz)
    ], dtype=float)
    end_pos = np.array([
        end.get("x", cx), end.get("y", cy), end.get("z", cz)
    ], dtype=float)

    # Clamp to volume bounds
    bounds = np.array(shape[:3], dtype=float) - 1
    start_pos = np.clip(start_pos, 0, bounds)
    end_pos = np.clip(end_pos, 0, bounds)

    return np.linspace(start_pos, end_pos, num_frames)


def save_scan_video(frames: list[Image.Image], scan_id: int, fps: int = 4):
    """Save scan frames as a video artifact. Tries mp4, falls back to gif."""
    video_dir = RESULTS_DIR / "scans"
    video_dir.mkdir(parents=True, exist_ok=True)

    frame_arrays = [np.array(f) for f in frames]

    # Try mp4 with explicit codec, fall back to gif
    video_path = video_dir / f"scan_{scan_id:03d}.mp4"
    try:
        import imageio.v3 as iio
        iio.imwrite(video_path, frame_arrays, fps=fps, codec="libx264")
        print(f"  Scan video saved: {video_path} ({len(frames)} frames)")
        return
    except Exception as e:
        print(f"  WARNING: mp4 save failed ({e}), trying gif ...")

    # Fallback: save as gif
    video_path = video_dir / f"scan_{scan_id:03d}.gif"
    try:
        import imageio.v3 as iio
        iio.imwrite(video_path, frame_arrays, duration=int(1000 / fps), loop=0)
        print(f"  Scan video saved: {video_path} ({len(frames)} frames, gif)")
    except Exception as e2:
        # Last resort: just save individual frames as PNGs
        print(f"  WARNING: gif save also failed ({e2}), saving individual frames")
        for i, frame in enumerate(frames):
            frame_path = video_dir / f"scan_{scan_id:03d}_frame_{i:03d}.png"
            Image.fromarray(frame_arrays[i]).save(frame_path)
        print(f"  Saved {len(frames)} frames as PNGs in {video_dir}")
