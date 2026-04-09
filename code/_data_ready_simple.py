#!/usr/bin/env python3
"""Simple chunk-stability readiness probe for Neuroglancer links.

Replaces the CDP two-phase approach with a direct poll of Neuroglancer's own
chunk readiness state. Declares ready when (available, needed) is unchanged for
STABLE_POLLS consecutive polls. Handles both 100%-loaded and plateau cases
(e.g. links where available stabilises below needed).

No CDP, no network event tracking — just ask the viewer directly.

Usage:
    python _data_ready_simple.py               # all links in ng_links/
    python _data_ready_simple.py example_ng_link  # single link by name
"""

import base64
import os
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/scratch/ms-playwright")

STABLE_POLLS = 4      # consecutive unchanged (available, needed) polls → ready
POLL_S = 0.5          # seconds between polls
TIMEOUT_S = 60.0      # max wait per link
WARMUP_S = 10.0       # max wait for viewer to initialise (needed > 0)

SCREENSHOT_DIR = Path(__file__).parent.parent / "results" / "probe_screenshots"

NG_HIDE_CSS = """
    .neuroglancer-viewer-top-row { display: none !important; }
    .neuroglancer-layer-panel { display: none !important; }
    .neuroglancer-layer-side-panel { display: none !important; }
    .neuroglancer-statistics-panel { display: none !important; }
    .neuroglancer-layer-group-viewer-top-row { display: none !important; }
    #neuroglancer-container > div > div:first-child { display: none !important; }
"""


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


def probe(browser, ng_link: str, label: str) -> dict:
    """Wait for chunk stability then screenshot."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Link: {len(ng_link)} chars")
    print(f"{'=' * 70}")

    context = browser.new_context(viewport={"width": 1024, "height": 1024})
    page = context.new_page()

    # Patch WebGL context creation before the page loads so that
    # preserveDrawingBuffer=true is set. Without this, the framebuffer is
    # cleared after each composite and toDataURL() returns a black image.
    page.add_init_script("""
        const _origGetContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(type, attrs) {
            if (type === 'webgl' || type === 'webgl2') {
                attrs = Object.assign({}, attrs || {}, {preserveDrawingBuffer: true});
            }
            return _origGetContext.call(this, type, attrs);
        };
    """)

    try:
        page.goto(ng_link, wait_until="domcontentloaded", timeout=15000)

        # Wait for viewer to initialise (needed > 0)
        t_start = time.time()
        viewer_ready = False
        while time.time() - t_start < WARMUP_S:
            needed, available = _get_chunk_counts(page)
            if needed > 0:
                viewer_ready = True
                print(f"  Viewer ready at {time.time() - t_start:.1f}s "
                      f"(needed={needed}, available={available})")
                break
            time.sleep(0.2)

        if not viewer_ready:
            print(f"  WARNING: viewer not ready after {WARMUP_S}s")
            return {
                "label": label, "result": "NO_VIEWER",
                "ready_time": None, "chunks_available": 0, "chunks_needed": 0,
                "chunk_pct": 0, "screenshot": None,
            }

        # Poll until (available, needed) is stable for STABLE_POLLS consecutive reads
        print(f"\n  Polling for stability ({STABLE_POLLS} × {POLL_S}s unchanged)...")
        print(f"  {'Time':>6}  {'Avail/Needed':>14}  {'%':>6}  Stable")
        print(f"  {'-' * 42}")

        t0 = time.time()
        prev = (-1, -1)
        stable_count = 0
        poll_num = 0
        ready = False
        needed = available = 0

        while time.time() - t0 < TIMEOUT_S:
            time.sleep(POLL_S)
            elapsed = time.time() - t0
            poll_num += 1

            needed, available = _get_chunk_counts(page)
            pct = available / needed * 100 if needed > 0 else 0
            cur = (available, needed)

            if cur == prev and needed > 0:
                stable_count += 1
            else:
                stable_count = 0
            prev = cur

            # Print every ~1s (every 2 polls) and on stability events
            if poll_num % 2 == 0 or stable_count in (1, STABLE_POLLS):
                marker = " ✓" if stable_count >= STABLE_POLLS else ""
                print(f"  {elapsed:6.1f}s  {available:>6}/{needed:<7}  "
                      f"{pct:5.1f}%  {stable_count}{marker}")

            if stable_count >= STABLE_POLLS:
                ready = True
                break

        ready_time = time.time() - t0

        if not ready:
            print(f"  → TIMEOUT at {ready_time:.1f}s")
            return {
                "label": label, "result": "TIMEOUT",
                "ready_time": ready_time,
                "chunks_available": available, "chunks_needed": needed,
                "chunk_pct": available / needed * 100 if needed > 0 else 0,
                "screenshot": None,
            }

        pct = available / needed * 100 if needed > 0 else 0
        print(f"  → Ready at {ready_time:.1f}s ({available}/{needed}, {pct:.0f}%)")

        # Hide UI chrome then read the canvas directly via JS.
        # Bypasses Playwright's screenshot machinery (which blocks on font/animation
        # stabilisation and can timeout even when the canvas is fully rendered).
        page.add_style_tag(content=NG_HIDE_CSS)
        time.sleep(0.1)

        data_url = page.evaluate("""() => {
            const canvas = document.querySelector('canvas');
            return canvas ? canvas.toDataURL('image/png') : null;
        }""")
        if not data_url:
            print("  WARNING: no canvas found, skipping screenshot")
            return {
                "label": label, "result": "NO_CANVAS",
                "ready_time": ready_time,
                "chunks_available": available, "chunks_needed": needed,
                "chunk_pct": pct, "screenshot": None,
            }

        png_bytes = base64.b64decode(data_url.split(",", 1)[1])
        img = Image.open(BytesIO(png_bytes)).convert("RGB")

        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        png_path = SCREENSHOT_DIR / f"{label}.png"
        img.save(png_path)
        print(f"  Screenshot: {png_path} ({img.size[0]}x{img.size[1]})")

        return {
            "label": label, "result": "PASS",
            "ready_time": ready_time,
            "chunks_available": available, "chunks_needed": needed,
            "chunk_pct": pct,
            "screenshot": str(png_path),
        }

    finally:
        context.close()


def main():
    from playwright.sync_api import sync_playwright

    ng_links_dir = Path(__file__).parent / "ng_links"
    link_files = sorted(ng_links_dir.glob("*.txt"))

    if len(sys.argv) > 1:
        pattern = sys.argv[1]
        link_files = [f for f in link_files if pattern in f.stem]
        if not link_files:
            print(f"No link files matching '{pattern}' in {ng_links_dir}")
            sys.exit(1)

    print(f"Chunk-Stability Readiness Probe")
    print(f"Links to test: {len(link_files)}")
    print(f"Stable when: {STABLE_POLLS} × {POLL_S}s unchanged | Timeout: {TIMEOUT_S}s")

    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--use-gl=egl"])
        for link_file in link_files:
            ng_link = link_file.read_text().strip()
            result = probe(browser, ng_link, link_file.stem)
            results.append(result)
        browser.close()

    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Link':<25}  {'Result':>8}  {'Ready(s)':>8}  {'Chunks':>12}  {'%':>5}")
    print("-" * 65)
    for r in results:
        t = f"{r['ready_time']:.1f}" if r['ready_time'] is not None else "---"
        chunks = f"{r['chunks_available']}/{r['chunks_needed']}"
        print(f"{r['label']:<25}  {r['result']:>8}  {t:>8}  {chunks:>12}  {r['chunk_pct']:4.0f}%")

    pass_count = sum(1 for r in results if r["result"] == "PASS")
    print(f"\n{pass_count}/{len(results)} PASS")


if __name__ == "__main__":
    main()
