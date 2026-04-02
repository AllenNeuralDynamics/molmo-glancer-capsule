"""
visual_capture — Playwright-based clean state builder and screenshot capture.

Handles:
- Building clean NG states (overlay hiding, view spec application)
- CSS injection to hide remaining UI chrome
- Canvas-only screenshot capture with readiness polling
- Scan frame generation (video sweeps)
"""

import asyncio
import hashlib
import json
import threading
import time
from io import BytesIO
from pathlib import Path

import os

import numpy as np
from PIL import Image, ImageDraw

VIEWPORT_SIZE = 1024
SCRATCH_TMP = "/scratch/tmp"
os.makedirs(SCRATCH_TMP, exist_ok=True)
os.environ.setdefault("TMPDIR", SCRATCH_TMP)

_CHROMIUM_ARGS_BASE = [
    "--disable-blink-features=AutomationControlled",
    f"--disk-cache-dir={SCRATCH_TMP}/chromium-cache",
    f"--crash-dumps-dir={SCRATCH_TMP}/chromium-crashes",
]

# Hardware GPU rendering via EGL — only for L40S (full profile).
# T4 reserves its GPU entirely for model inference.
_CHROMIUM_ARGS_GPU = _CHROMIUM_ARGS_BASE + ["--use-gl=egl"]


def _chromium_args(config: dict = None) -> list[str]:
    """Return Chromium launch args, with GPU acceleration for full profile."""
    if config and config.get("quantization") is None:  # full profile = no quantization
        return _CHROMIUM_ARGS_GPU
    return _CHROMIUM_ARGS_BASE
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


def _canvas_has_data(png_bytes: bytes, threshold: float = 0.02) -> bool:
    """Check if more than `threshold` fraction of canvas pixels are non-black.

    NG UI chrome (axis labels, crosshairs, scale bar) covers <1% of pixels.
    Actual volume data fills much more. Default threshold of 2% distinguishes
    empty-with-chrome from data-loaded.
    """
    img = Image.open(BytesIO(png_bytes))
    arr = np.array(img)
    non_black = np.any(arr > 10, axis=-1).mean()
    return non_black > threshold


def _wait_for_canvas_stable(page, interval_ms: int = 150, max_attempts: int = 10,
                            wait_for_data: bool = False):
    """Poll until the canvas pixels stop changing between consecutive snapshots (sync).

    If wait_for_data=True, first waits until the canvas has meaningful pixel
    content (>2% non-black), then waits for stability. Use for cold loads.
    """
    canvas = page.locator("canvas").first
    if wait_for_data:
        for _ in range(60):  # up to ~30s for data to arrive
            png = canvas.screenshot()
            if _canvas_has_data(png):
                break
            time.sleep(0.5)
    prev_hash = None
    for _ in range(max_attempts):
        png = canvas.screenshot()
        h = hashlib.md5(png).hexdigest()
        if h == prev_hash:
            return
        prev_hash = h
        time.sleep(interval_ms / 1000)


async def _async_wait_for_canvas_stable(page, interval_ms: int = 150, max_attempts: int = 10,
                                        wait_for_data: bool = False):
    """Poll until the canvas pixels stop changing between consecutive snapshots (async).

    If wait_for_data=True, first waits until the canvas has meaningful pixel
    content (>2% non-black), then waits for stability. Use for cold loads.
    """
    canvas = page.locator("canvas").first
    if wait_for_data:
        for _ in range(60):
            png = await canvas.screenshot()
            if _canvas_has_data(png):
                break
            await asyncio.sleep(0.5)
    prev_hash = None
    for _ in range(max_attempts):
        png = await canvas.screenshot()
        h = hashlib.md5(png).hexdigest()
        if h == prev_hash:
            return
        prev_hash = h
        await asyncio.sleep(interval_ms / 1000)


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

    page.goto(url, wait_until="domcontentloaded", timeout=10000)

    # Inject CSS to hide UI chrome
    page.add_style_tag(content=NG_HIDE_CSS)

    # Wait for data to arrive (canvas changes from blank), then stabilize
    _wait_for_canvas_stable(page, wait_for_data=True)

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


def create_browser(playwright, config: dict = None):
    """Create a Playwright browser + page with 1024x1024 viewport."""
    browser = playwright.chromium.launch(
        headless=True,
        args=_chromium_args(config),
    )
    context = browser.new_context(
        viewport={"width": VIEWPORT_SIZE, "height": VIEWPORT_SIZE},
    )
    page = context.new_page()
    return browser, page


# ── Scan Frame Generation ───────────────────────────────────────────────────


def execute_scan(base_state, scan_spec: dict, volume_info, config: dict, scan_id: int) -> list[Image.Image]:
    """Execute a scan: single-page sequential capture via async Playwright.

    Uses one page with hash-fragment updates so adjacent frames share cached
    zarr chunks (~90% overlap for Z-sweeps). Runs in a separate thread with
    its own async event loop to avoid conflicts with the sync Playwright
    instance in the main thread.

    Args:
        base_state: NeuroglancerState to use as template.
        scan_spec: Dict with scan_type, start, end, frames, layout, crossSectionScale, etc.
        volume_info: VolumeInfo for bounds clamping.
        config: GPU profile config.
        scan_id: Sequential ID for naming the video file.

    Returns:
        List of PIL Images (one per frame).
    """
    scan_type = scan_spec.get("scan_type", "z_sweep")
    num_frames = min(scan_spec.get("frames", config["max_scan_frames"]), config["max_scan_frames"])
    layout = scan_spec.get("layout", "xy")
    cross_section_scale = scan_spec.get("crossSectionScale", 1.0)

    positions = generate_scan_positions(scan_spec, volume_info, num_frames)
    print(f"  Scan {scan_id}: {scan_type}, {len(positions)} frames, layout={layout}")

    # Build all states up front
    states = []
    for i, pos in enumerate(positions):
        view_spec = {
            "x": pos[0], "y": pos[1], "z": pos[2],
            "layout": layout,
            "crossSectionScale": cross_section_scale,
        }
        if "projectionOrientation" in scan_spec:
            if scan_type == "rotation" and "orientations" in scan_spec:
                view_spec["projectionOrientation"] = scan_spec["orientations"][i]
            else:
                view_spec["projectionOrientation"] = scan_spec["projectionOrientation"]
        states.append(build_clean_state(base_state, view_spec, volume_info))

    max_side = config.get("max_image_side")

    async def _run_sequential():
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=_chromium_args(config),
            )
            ctx = await browser.new_context(
                viewport={"width": VIEWPORT_SIZE, "height": VIEWPORT_SIZE},
            )
            page = await ctx.new_page()
            try:
                # First frame: full navigation, wait for data to load
                await page.goto(states[0].to_url(), wait_until="domcontentloaded", timeout=15000)
                await page.add_style_tag(content=NG_HIDE_CSS)
                await _async_wait_for_canvas_stable(page, wait_for_data=True)

                frames = []
                for i, state in enumerate(states):
                    if i > 0:
                        # Hash-fragment update — adjacent slices share ~90% of chunks
                        state_json = json.dumps(state.data, separators=(",", ":"))
                        await page.evaluate("(h) => { location.hash = '!' + h }", state_json)
                        await _async_wait_for_canvas_stable(page)

                    canvas = page.locator("canvas").first
                    png_bytes = await canvas.screenshot()
                    img = Image.open(BytesIO(png_bytes)).convert("RGB")
                    if max_side and max(img.size) > max_side:
                        img.thumbnail((max_side, max_side), Image.LANCZOS)
                    frames.append(img)

                    if (i + 1) % 10 == 0 or i == 0:
                        print(f"    frame {i+1}/{len(states)}")

                return frames
            finally:
                await ctx.close()
                await browser.close()

    # Run in a separate thread to avoid conflict with the sync Playwright
    # event loop already running in the main thread.
    result_holder = {}

    def _run_in_thread():
        loop = asyncio.new_event_loop()
        try:
            result_holder["frames"] = loop.run_until_complete(_run_sequential())
        finally:
            loop.close()

    t = threading.Thread(target=_run_in_thread)
    t.start()
    t.join()

    if "frames" not in result_holder:
        raise RuntimeError("Scan capture failed")

    frames = list(result_holder["frames"])
    print(f"    captured {len(frames)} frames")

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


def save_scan_video(frames: list[Image.Image], scan_id: int,
                    target_duration: float = 5.0, suffix: str = ""):
    """Save scan frames as a video artifact. Tries mp4, falls back to gif.

    fps is computed so that the video is always `target_duration` seconds long.
    suffix is appended to the filename (e.g. "_annotated" for annotated versions).
    """
    fps = max(1, len(frames) / target_duration)
    video_dir = RESULTS_DIR / "scans"
    video_dir.mkdir(parents=True, exist_ok=True)

    frame_arrays = [np.array(f) for f in frames]

    # Try mp4 with explicit codec, fall back to gif
    video_path = video_dir / f"scan_{scan_id:03d}{suffix}.mp4"
    try:
        import imageio.v3 as iio
        iio.imwrite(video_path, frame_arrays, fps=fps, codec="libx264",
                    plugin="pyav")
        print(f"  Scan video saved: {video_path} ({len(frames)} frames, {fps:.1f} fps, {target_duration}s)")
        return
    except Exception as e:
        print(f"  WARNING: mp4 save failed ({e}), trying gif ...")

    # Fallback: save as gif
    video_path = video_dir / f"scan_{scan_id:03d}{suffix}.gif"
    try:
        import imageio.v3 as iio
        iio.imwrite(video_path, frame_arrays, duration=int(1000 / fps), loop=0)
        print(f"  Scan video saved: {video_path} ({len(frames)} frames, {fps:.1f} fps, gif)")
    except Exception as e2:
        # Last resort: just save individual frames as PNGs
        print(f"  WARNING: gif save also failed ({e2}), saving individual frames")
        for i, frame in enumerate(frames):
            frame_path = video_dir / f"scan_{scan_id:03d}{suffix}_frame_{i:03d}.png"
            Image.fromarray(frame_arrays[i]).save(frame_path)
        print(f"  Saved {len(frames)} frames as PNGs in {video_dir}")


# ── Point Annotation ────────────────────────────────────────────────────────

MARKER_RADIUS = 8
MARKER_COLOR = (0, 255, 0)       # green fill
MARKER_OUTLINE = (255, 255, 255)  # white border


def _draw_markers(img: Image.Image, points: list[tuple[float, float]]) -> Image.Image:
    """Draw circle markers on a copy of the image at each (x, y) point."""
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    r = MARKER_RADIUS
    for x, y in points:
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=MARKER_COLOR, outline=MARKER_OUTLINE, width=2,
        )
    return annotated


def annotate_screenshot(img: Image.Image, points: list[tuple[float, float]],
                        screenshot_id: int) -> Image.Image:
    """Draw point markers on a screenshot and save the annotated version.

    Args:
        img: Original screenshot PIL Image.
        points: List of (x, y) pixel coordinates.
        screenshot_id: ID for filename.

    Returns:
        Annotated PIL Image.
    """
    annotated = _draw_markers(img, points)

    out_dir = RESULTS_DIR / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"view_{screenshot_id:03d}_annotated.png"
    annotated.save(out_path)
    print(f"  Annotated screenshot saved: {out_path} ({len(points)} markers)")

    return annotated


def annotate_scan_frames(frames: list[Image.Image],
                         points: list[tuple[float, float, float]],
                         scan_id: int) -> list[Image.Image]:
    """Draw point markers on scan frames and save as annotated video.

    Args:
        frames: Original scan frame PIL Images.
        points: List of (frame_idx, x, y) tuples. frame_idx is the 0-based
                frame index from per-keyframe image pointing.
        scan_id: ID for filename.

    Returns:
        List of annotated PIL Images.
    """
    points_by_frame: dict[int, list[tuple[float, float]]] = {}
    for frame_id, x, y in points:
        idx = min(round(frame_id), len(frames) - 1)
        idx = max(0, idx)
        points_by_frame.setdefault(idx, []).append((x, y))

    annotated_frames = []
    for i, frame in enumerate(frames):
        if i in points_by_frame:
            annotated_frames.append(_draw_markers(frame, points_by_frame[i]))
        else:
            annotated_frames.append(frame.copy())

    # Save annotated video
    save_scan_video(annotated_frames, scan_id, suffix="_annotated")

    frames_with_markers = sum(1 for i in points_by_frame if i < len(frames))
    print(f"  Annotated scan saved: {len(points)} markers across {frames_with_markers} frames")

    return annotated_frames
