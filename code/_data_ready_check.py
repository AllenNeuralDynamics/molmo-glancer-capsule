#!/usr/bin/env python3
"""Probe two-phase readiness (CDP network idle → chunk stall) across all NG links.

For each .txt file in ng_links/, navigates to the URL and runs the D5 two-phase
detection from PLAN_v4.md:
  Phase 1: CDP network idle — 0 pending requests for net_idle_s seconds
  Phase 2: Chunk count stall — available unchanged for N consecutive polls

Reports per-link timing and a summary table.

Usage:
    python probe_cdp_readiness.py              # all links
    python probe_cdp_readiness.py example_ng_link  # single link by name
"""

import hashlib
import os
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/scratch/ms-playwright")

NET_IDLE_S = 1.0         # Phase 1: seconds of zero network activity
CHUNK_STABLE_POLLS = 3   # Phase 2: consecutive unchanged polls
CHUNK_POLL_MS = 500      # Phase 2: polling interval
TIMEOUT_S = 60.0         # max total wait per link
WARMUP_S = 10.0          # max wait for viewer to initialize

SCREENSHOT_DIR = Path(__file__).parent.parent / "results" / "probe_screenshots"

# CSS to hide all Neuroglancer UI chrome (same as visual_capture.py)
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


def _get_viewer_state(page) -> dict:
    """Check viewer initialization state for diagnostics."""
    return page.evaluate("""(() => {
        const v = window.viewer;
        if (!v) return {viewer: false};
        const lm = v.layerManager;
        if (!lm) return {viewer: true, layerManager: false};
        const managed = lm.managedLayers || [];
        const layers = [];
        for (const ml of managed) {
            const info = {name: ml.name, hasLayer: !!ml.layer};
            if (ml.layer) {
                const rls = ml.layer.renderLayers;
                info.renderLayerCount = rls ? (rls.length || rls.size || 0) : 0;
            }
            layers.push(info);
        }
        return {viewer: true, layerManager: true, managedCount: managed.length, layers};
    })()""")


def probe_two_phase(browser, ng_link: str, label: str):
    """Run two-phase readiness detection on a single NG link."""

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Link: {len(ng_link)} chars")
    print(f"{'=' * 70}")

    context = browser.new_context(viewport={"width": 1024, "height": 1024})
    page = context.new_page()

    try:
        page.goto(ng_link, wait_until="domcontentloaded", timeout=15000)

        # Phase 0: Wait for viewer to initialize (needed > 0)
        t_start = time.time()
        viewer_ready = False
        while time.time() - t_start < WARMUP_S:
            needed, available = _get_chunk_counts(page)
            if needed > 0:
                viewer_ready = True
                warmup_time = time.time() - t_start
                print(f"  Viewer ready at {warmup_time:.1f}s "
                      f"(needed={needed}, available={available})")
                break
            time.sleep(0.5)

        if not viewer_ready:
            state = _get_viewer_state(page)
            print(f"  WARNING: viewer not ready after {WARMUP_S}s")
            print(f"  Viewer state: {state}")
            return {
                "label": label, "result": "NO_VIEWER",
                "phase1_time": None, "phase2_time": None, "total_time": None,
                "chunks_available": 0, "chunks_needed": 0, "chunk_pct": 0,
                "requests_completed": 0, "requests_failed": 0,
            }

        # ── Phase 1: CDP network idle ──────────────────────────────────
        cdp = page.context.new_cdp_session(page)
        cdp.send("Network.enable")

        pending = set()
        completed_count = [0]
        failed_count = [0]
        last_activity = [time.time()]

        def on_request(params):
            pending.add(params["requestId"])
            last_activity[0] = time.time()

        def on_finished(params):
            pending.discard(params.get("requestId"))
            completed_count[0] += 1
            last_activity[0] = time.time()

        def on_failed(params):
            pending.discard(params.get("requestId"))
            failed_count[0] += 1
            last_activity[0] = time.time()

        cdp.on("Network.requestWillBeSent", on_request)
        cdp.on("Network.loadingFinished", on_finished)
        cdp.on("Network.loadingFailed", on_failed)

        print(f"\n  Phase 1: CDP network idle (waiting for {NET_IDLE_S}s of silence)")
        print(f"  {'Time':>6}  {'Pending':>8}  {'Done':>6}  {'Fail':>5}  "
              f"{'Needed':>7}  {'Avail':>6}  {'%':>6}")
        print(f"  {'-' * 60}")

        t0 = time.time()
        phase1_done = False
        phase1_time = None

        while time.time() - t0 < TIMEOUT_S:
            time.sleep(0.25)
            elapsed = time.time() - t0

            needed, available = _get_chunk_counts(page)
            if available == 0:
                continue

            idle_elapsed = time.time() - last_activity[0]
            pct = available / needed * 100 if needed > 0 else 0

            if int(elapsed * 4) % 4 == 0:
                print(f"  {elapsed:6.1f}  {len(pending):>8}  {completed_count[0]:>6}  "
                      f"{failed_count[0]:>5}  {needed:>7}  {available:>6}  {pct:5.1f}%")

            if (len(pending) == 0
                    and idle_elapsed >= NET_IDLE_S):
                phase1_done = True
                phase1_time = elapsed
                pct = available / needed * 100 if needed > 0 else 0
                print(f"  → Phase 1 DONE at {elapsed:.1f}s: network idle, "
                      f"chunks {available}/{needed} ({pct:.0f}%), "
                      f"{completed_count[0]} reqs done")
                break

        cdp.detach()

        if not phase1_done:
            elapsed = time.time() - t0
            print(f"  → Phase 1 TIMEOUT at {elapsed:.1f}s ({len(pending)} pending)")
            return {
                "label": label, "result": "NET_TIMEOUT",
                "phase1_time": None, "phase2_time": None,
                "total_time": elapsed,
                "chunks_available": available, "chunks_needed": needed,
                "chunk_pct": available / needed * 100 if needed > 0 else 0,
                "requests_completed": completed_count[0],
                "requests_failed": failed_count[0],
            }

        # ── Phase 2: Chunk count stall ─────────────────────────────────
        print(f"\n  Phase 2: Chunk count stall ({CHUNK_STABLE_POLLS} × "
              f"{CHUNK_POLL_MS}ms unchanged)")

        prev_available = -1
        stable_count = 0
        phase2_done = False
        phase2_time = None
        remaining = TIMEOUT_S - (time.time() - t0)

        t1 = time.time()
        while time.time() - t1 < remaining:
            needed, available = _get_chunk_counts(page)
            pct = available / needed * 100 if needed > 0 else 0
            elapsed_p2 = time.time() - t1

            if available == prev_available:
                stable_count += 1
                if stable_count >= CHUNK_STABLE_POLLS:
                    phase2_done = True
                    phase2_time = elapsed_p2
                    print(f"  → Phase 2 DONE at +{elapsed_p2:.1f}s: "
                          f"chunks stable {available}/{needed} ({pct:.0f}%)")
                    break
            else:
                if prev_available >= 0:
                    print(f"    +{elapsed_p2:.1f}s: chunks {prev_available} → {available}"
                          f"/{needed} ({pct:.0f}%)")
                stable_count = 0
            prev_available = available
            time.sleep(CHUNK_POLL_MS / 1000)

        total_time = time.time() - t0

        if not phase2_done:
            print(f"  → Phase 2 TIMEOUT at +{time.time() - t1:.1f}s")
            result_status = "CHUNK_TIMEOUT"
        else:
            result_status = "PASS"

        # ── Phase 3: Final pixel stability + canvas screenshot ─────────
        # Inject CSS to hide UI chrome (same as production capture)
        page.add_style_tag(content=NG_HIDE_CSS)
        time.sleep(0.1)  # let CSS apply

        # Pixel stability check: 2 consecutive identical canvas hashes
        canvas = page.locator("canvas").first
        prev_hash = None
        for _ in range(5):
            png_bytes = canvas.screenshot(timeout=60000)
            h = hashlib.md5(png_bytes).hexdigest()
            if h == prev_hash:
                break
            prev_hash = h
            time.sleep(0.2)

        # Capture final canvas screenshot
        png_bytes = canvas.screenshot(timeout=60000)
        img = Image.open(BytesIO(png_bytes)).convert("RGB")

        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        png_path = SCREENSHOT_DIR / f"{label}.png"
        img.save(png_path)
        print(f"\n  Screenshot saved: {png_path} ({img.size[0]}x{img.size[1]})")

        # Summary
        needed, available = _get_chunk_counts(page)
        pct = available / needed * 100 if needed > 0 else 0
        print(f"\n  Summary for {label}:")
        print(f"    Chunks: {available}/{needed} ({pct:.0f}%)")
        print(f"    Requests: {completed_count[0]} done, {failed_count[0]} failed")
        if phase1_time is not None:
            print(f"    Phase 1 (net idle): {phase1_time:.1f}s")
        if phase2_time is not None:
            print(f"    Phase 2 (chunk stall): +{phase2_time:.1f}s")
        print(f"    Total ready time: {total_time:.1f}s")
        print(f"    Screenshot: {png_path}")

        return {
            "label": label,
            "result": result_status,
            "phase1_time": phase1_time,
            "phase2_time": phase2_time,
            "total_time": total_time,
            "chunks_available": available,
            "chunks_needed": needed,
            "chunk_pct": pct,
            "requests_completed": completed_count[0],
            "requests_failed": failed_count[0],
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

    print(f"Two-Phase Readiness Probe (CDP idle → chunk stall)")
    print(f"Links to test: {len(link_files)}")
    print(f"Phase 1: net idle {NET_IDLE_S}s | Phase 2: {CHUNK_STABLE_POLLS} × "
          f"{CHUNK_POLL_MS}ms stall | Timeout: {TIMEOUT_S}s")

    results = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--use-gl=egl"])

        for link_file in link_files:
            ng_link = link_file.read_text().strip()
            label = link_file.stem
            result = probe_two_phase(browser, ng_link, label)
            results.append(result)

        browser.close()

    # Final summary table
    print(f"\n\n{'=' * 80}")
    print(f"  SUMMARY — Two-Phase Readiness (CDP idle → chunk stall)")
    print(f"{'=' * 80}")
    print(f"\n{'Link':<25}  {'Result':>12}  {'P1(s)':>6}  {'P2(s)':>6}  "
          f"{'Total':>6}  {'Chunks':>12}  {'%':>5}")
    print("-" * 85)
    for r in results:
        p1 = f"{r['phase1_time']:.1f}" if r['phase1_time'] is not None else "---"
        p2 = f"{r['phase2_time']:.1f}" if r['phase2_time'] is not None else "---"
        total = f"{r['total_time']:.1f}" if r['total_time'] is not None else "---"
        chunk_str = f"{r['chunks_available']}/{r['chunks_needed']}"
        print(f"{r['label']:<25}  {r['result']:>12}  {p1:>6}  {p2:>6}  "
              f"{total:>6}  {chunk_str:>12}  {r['chunk_pct']:4.0f}%")

    pass_count = sum(1 for r in results if r["result"] == "PASS")
    print(f"\n{pass_count}/{len(results)} PASS")


if __name__ == "__main__":
    main()
