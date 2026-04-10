"""
visual_capture — Playwright-based clean state builder and screenshot capture.

Handles:
- Building clean NG states (overlay hiding, view spec application)
- CSS injection to hide remaining UI chrome
- Canvas-only screenshot capture via chunk-stability readiness + JS toDataURL
- Scan frame generation (video sweeps)

Targets L40S exclusively — always uses hardware GPU rendering via EGL.
"""

import asyncio
import base64
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

CHROMIUM_ARGS = [
    "--disable-blink-features=AutomationControlled",
    f"--disk-cache-dir={SCRATCH_TMP}/chromium-cache",
    f"--crash-dumps-dir={SCRATCH_TMP}/chromium-crashes",
    "--use-gl=egl",
]

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

# Patch WebGL context creation so that preserveDrawingBuffer=true is set.
# Without this, the framebuffer is cleared after each composite and
# toDataURL() returns a black image. Must be applied via add_init_script
# BEFORE page navigation.
PRESERVE_DRAWING_BUFFER_JS = """
    const _origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(type, attrs) {
        if (type === 'webgl' || type === 'webgl2') {
            attrs = Object.assign({}, attrs || {}, {preserveDrawingBuffer: true});
        }
        return _origGetContext.call(this, type, attrs);
    };
"""

# JS to query total (needed, available) across all visible render layers.
_CHUNK_COUNTS_JS = """(() => {
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
})()"""

_CANVAS_TO_DATA_URL_JS = """() => {
    const canvas = document.querySelector('canvas');
    return canvas ? canvas.toDataURL('image/png') : null;
}"""

# Readiness constants (matching validated probe: _data_ready_simple.py)
STABLE_POLLS = 4       # consecutive unchanged (available, needed) polls → ready
POLL_S = 0.5           # seconds between stability polls
TIMEOUT_S = 60.0       # max wait for chunk stability
WARMUP_S = 10.0        # max wait for viewer to initialise (needed > 0)
WARMUP_POLL_S = 0.2    # seconds between warmup polls


# ── Chunk-Stability Readiness ───────────────────────────────────────────────

def _get_chunk_counts(page) -> tuple[int, int]:
    """Query total (needed, available) across all visible render layers."""
    result = page.evaluate(_CHUNK_COUNTS_JS)
    if result is None:
        return (0, 0)
    return (result["needed"], result["available"])


def _wait_for_data_loaded(page, timeout_s=TIMEOUT_S, stable_polls=STABLE_POLLS,
                          poll_s=POLL_S, warmup=True):
    """Wait until chunk counts (available, needed) are stable.

    Handles both 100%-loaded and plateau cases (e.g. links where available
    stabilises below needed). Declares ready when (available, needed) is
    unchanged for `stable_polls` consecutive reads.

    Parameters
    ----------
    page : playwright page
    timeout_s : float
        Max wait time for stability after warmup.
    stable_polls : int
        Consecutive unchanged polls required.
    poll_s : float
        Seconds between stability polls.
    warmup : bool
        If True, first wait up to WARMUP_S for viewer to initialise (needed > 0).
    """
    if warmup:
        t_start = time.time()
        viewer_ready = False
        while time.time() - t_start < WARMUP_S:
            needed, available = _get_chunk_counts(page)
            if needed > 0:
                viewer_ready = True
                print(f"    Viewer ready at {time.time() - t_start:.1f}s "
                      f"(needed={needed}, available={available})")
                break
            time.sleep(WARMUP_POLL_S)
        if not viewer_ready:
            print(f"    WARNING: viewer not ready after {WARMUP_S}s")
            return

    t0 = time.time()
    prev = (-1, -1)
    stable_count = 0

    while time.time() - t0 < timeout_s:
        time.sleep(poll_s)
        needed, available = _get_chunk_counts(page)
        cur = (available, needed)
        if cur == prev and needed > 0 and available > 0:
            stable_count += 1
            if stable_count >= stable_polls:
                pct = available / needed * 100 if needed > 0 else 0
                print(f"    Chunks stable: {available}/{needed} ({pct:.0f}%)")
                return
        else:
            stable_count = 0
        prev = cur

    pct = available / needed * 100 if needed > 0 else 0
    print(f"    WARNING: chunk stability timeout after {timeout_s}s "
          f"({available}/{needed}, {pct:.0f}%)")


async def _async_wait_for_data_loaded(page, timeout_s=TIMEOUT_S,
                                      stable_polls=STABLE_POLLS, poll_s=POLL_S,
                                      warmup=True):
    """Async version of _wait_for_data_loaded for scan frame capture."""
    if warmup:
        t_start = time.time()
        viewer_ready = False
        while time.time() - t_start < WARMUP_S:
            result = await page.evaluate(_CHUNK_COUNTS_JS)
            needed = result["needed"] if result else 0
            if needed > 0:
                viewer_ready = True
                break
            await asyncio.sleep(WARMUP_POLL_S)
        if not viewer_ready:
            print(f"    WARNING: viewer not ready after {WARMUP_S}s")
            return

    t0 = time.time()
    prev = (-1, -1)
    stable_count = 0

    while time.time() - t0 < timeout_s:
        await asyncio.sleep(poll_s)
        result = await page.evaluate(_CHUNK_COUNTS_JS)
        if result is None:
            continue
        cur = (result["available"], result["needed"])
        if cur == prev and result["needed"] > 0 and result["available"] > 0:
            stable_count += 1
            if stable_count >= stable_polls:
                return
        else:
            stable_count = 0
        prev = cur


# ── Canvas Capture ──────────────────────────────────────────────────────────

def _capture_canvas(page) -> bytes:
    """Capture the WebGL canvas via toDataURL(). Returns PNG bytes.

    Requires preserveDrawingBuffer=true to have been set via init script
    before navigation, otherwise the framebuffer is cleared after compositing
    and this returns a black image.
    """
    data_url = page.evaluate(_CANVAS_TO_DATA_URL_JS)
    if not data_url:
        raise RuntimeError("No canvas found on page")
    return base64.b64decode(data_url.split(",", 1)[1])


async def _async_capture_canvas(page) -> bytes:
    """Async version of _capture_canvas for scan frame capture."""
    data_url = await page.evaluate(_CANVAS_TO_DATA_URL_JS)
    if not data_url:
        raise RuntimeError("No canvas found on page")
    return base64.b64decode(data_url.split(",", 1)[1])


def _canvas_has_data(png_bytes: bytes, threshold: float = 0.02) -> bool:
    """Check if more than `threshold` fraction of canvas pixels are non-black.

    Post-capture sanity check. If chunk stability declared ready but the
    canvas is blank, something went wrong (e.g. WebGL context lost).
    """
    img = Image.open(BytesIO(png_bytes))
    arr = np.array(img)
    non_black = np.any(arr > 10, axis=-1).mean()
    return non_black > threshold


# ── Clean State Builder ─────────────────────────────────────────────────────

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
        layerColors               — {"layer_name": "#RRGGBB", ...}
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

    # ── Layer colors ───────────────────────────────────────────────────
    if "layerColors" in view_spec:
        for layer in d.get("layers", []):
            name = layer.get("name", "")
            if name in view_spec["layerColors"]:
                color = view_spec["layerColors"][name]
                sc = layer.setdefault("shaderControls", {})
                sc["color"] = color

    return state


# ── Screenshot Capture ──────────────────────────────────────────────────────

def capture_screenshot(page, state, screenshot_id: int) -> Image.Image:
    """Navigate to an NG state URL, wait for chunk stability, and capture canvas.

    Uses chunk-stability readiness (poll Neuroglancer's layerChunkProgressInfo
    until stable) followed by JS canvas.toDataURL() capture. Bypasses
    Playwright's screenshot machinery which can timeout on font/animation
    stabilisation.

    Parameters
    ----------
    page : playwright page
        Page with preserveDrawingBuffer init script already applied.
    state : NeuroglancerState
        Clean view state to navigate to.
    screenshot_id : int
        Sequential ID for saving the PNG.

    Returns
    -------
    PIL.Image.Image
    """
    url = state.to_url()
    print(f"  Navigating to NG URL ({len(url)} chars) ...")

    page.goto(url, wait_until="domcontentloaded", timeout=15000)

    # Wait for chunk stability
    _wait_for_data_loaded(page)

    # Hide UI chrome, brief delay for CSS to take effect
    page.add_style_tag(content=NG_HIDE_CSS)
    time.sleep(0.1)

    # Capture canvas via JS toDataURL
    png_bytes = _capture_canvas(page)

    # Sanity check — abort if blank
    if not _canvas_has_data(png_bytes):
        print(f"  WARNING: canvas appears blank after chunk stability — possible WebGL issue")

    img = Image.open(BytesIO(png_bytes)).convert("RGB")

    # Save to results
    screenshot_dir = RESULTS_DIR / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    png_path = screenshot_dir / f"view_{screenshot_id:03d}.png"
    img.save(png_path)
    print(f"  Screenshot saved: {png_path} ({img.size[0]}x{img.size[1]})")

    return img


def create_browser(playwright):
    """Create a Playwright browser + page with 1024x1024 viewport.

    Applies the preserveDrawingBuffer WebGL patch via init script so that
    toDataURL() reads the live framebuffer instead of returning black.
    """
    browser = playwright.chromium.launch(
        headless=True,
        args=CHROMIUM_ARGS,
    )
    context = browser.new_context(
        viewport={"width": VIEWPORT_SIZE, "height": VIEWPORT_SIZE},
    )
    page = context.new_page()
    page.add_init_script(PRESERVE_DRAWING_BUFFER_JS)
    return browser, page


# ── Scan Frame Generation ───────────────────────────────────────────────────


def execute_scan(base_state, scan_spec: dict, volume_info, config: dict, scan_id: int) -> list[Image.Image]:
    """Execute a scan: single-page sequential capture via async Playwright.

    Uses one page with hash-fragment updates so adjacent frames share cached
    zarr chunks (~90% overlap for Z-sweeps). Runs in a separate thread with
    its own async event loop to avoid conflicts with the sync Playwright
    instance in the main thread.

    Parameters
    ----------
    base_state : NeuroglancerState
        Template state.
    scan_spec : dict
        Scan parameters (scan_type, start, end, frames, layout, crossSectionScale, etc.).
    volume_info : VolumeInfo
        For bounds clamping.
    config : dict
        Config dict (uses max_scan_frames).
    scan_id : int
        Sequential ID for naming the video file.

    Returns
    -------
    list[PIL.Image.Image]
        One image per frame.
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
        if "layerVisibility" in scan_spec:
            view_spec["layerVisibility"] = scan_spec["layerVisibility"]
        states.append(build_clean_state(base_state, view_spec, volume_info))

    async def _run_sequential():
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=CHROMIUM_ARGS,
            )
            ctx = await browser.new_context(
                viewport={"width": VIEWPORT_SIZE, "height": VIEWPORT_SIZE},
            )
            page = await ctx.new_page()
            await page.add_init_script(PRESERVE_DRAWING_BUFFER_JS)
            try:
                # First frame: full navigation, wait for warmup + chunk stability
                await page.goto(states[0].to_url(), wait_until="domcontentloaded", timeout=15000)
                await page.add_style_tag(content=NG_HIDE_CSS)
                await _async_wait_for_data_loaded(page, warmup=True)

                frames = []
                for i, state in enumerate(states):
                    if i > 0:
                        # Hash-fragment update — adjacent slices share ~90% of chunks
                        state_json = json.dumps(state.data, separators=(",", ":"))
                        await page.evaluate("(h) => { location.hash = '!' + h }", state_json)
                        await _async_wait_for_data_loaded(page, warmup=False)

                    # Brief pause for rendering after stability
                    await asyncio.sleep(0.1)
                    png_bytes = await _async_capture_canvas(page)
                    img = Image.open(BytesIO(png_bytes)).convert("RGB")
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
        for i, arr in enumerate(frame_arrays):
            frame_path = video_dir / f"scan_{scan_id:03d}{suffix}_frame_{i:03d}.png"
            Image.fromarray(arr).save(frame_path)
        print(f"  Saved {len(frames)} frames as PNGs in {video_dir}")


# ── Point Annotation ────────────────────────────────────────────────────────

MARKER_RADIUS = 16
MARKER_COLOR = (255, 0, 0)       # red fill
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

    Parameters
    ----------
    img : PIL.Image.Image
        Original screenshot.
    points : list[tuple[float, float]]
        List of (x, y) pixel coordinates.
    screenshot_id : int
        ID for filename.

    Returns
    -------
    PIL.Image.Image
        Annotated image.
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

    Parameters
    ----------
    frames : list[PIL.Image.Image]
        Original scan frames.
    points : list[tuple[float, float, float]]
        List of (frame_idx, x, y) tuples. frame_idx is the 0-based
        frame index from per-keyframe image pointing.
    scan_id : int
        ID for filename.

    Returns
    -------
    list[PIL.Image.Image]
        Annotated frames.
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
