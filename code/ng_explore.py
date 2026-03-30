"""
MolmoWeb Autonomous Neuroglancer Explorer
==========================================
Gives MolmoWeb one Neuroglancer URL (single XY-panel view) and lets it
autonomously scroll through Z slices, describe the 3D mFISH brain tissue
dataset, and produce a scientific summary.

Usage:
    python /code/ng_explore.py
    python /code/ng_explore.py --max-steps 50
    python /code/ng_explore.py --url "https://neuroglancer-demo.appspot.com/#!{...}"
    python /code/ng_explore.py --no-server                # if MolmoWeb server is already running on port 8001
    python /code/ng_explore.py --record-video             # also save a real-time browser recording (mp4)
    python /code/ng_explore.py --mode registration        # 2-layer alignment assessment mode
    python /code/ng_explore.py --mode registration_large  # neuronal-level registration with zoom/pan

Outputs (in /results/ng_explore/):
    trajectory.html          — interactive replay of every screenshot + agent thought
    trajectory_video.mp4     — slideshow of per-step screenshots (always produced)
    browser_video.webm       — real-time Playwright screen recording (with --record-video)

CTRL+scroll convention (registration_large mode):
    scroll_at with delta_x != 0  →  CTRL+scroll zoom (delta_y ignored)
      delta_x > 0  →  zoom in
      delta_x < 0  →  zoom out
    scroll_at with delta_y != 0  →  Z navigation (normal scroll, delta_x ignored)
"""

import argparse
import json
import os
import urllib.parse
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── path / env setup ────────────────────────────────────────────────────────
sys.path.insert(0, "/code/lib/molmoweb")
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/scratch/ms-playwright")
os.environ.setdefault("HF_HOME", "/scratch/huggingface")
os.environ.setdefault("MOLMO_DATA_DIR", "/scratch/molmo_data")

import httpx

# ── config ───────────────────────────────────────────────────────────────────
MOLMOWEB_CKPT = "/scratch/checkpoints/MolmoWeb-4B"
MOLMOWEB_PORT = 8001
RESULTS_DIR   = Path("/results/ng_explore")

# ── modes ─────────────────────────────────────────────────────────────────────

MODES = {
    "explore": {
        "link_file": Path("/root/capsule/example_ng_link.txt"),
        "task": """\
You are viewing a 3D mFISH (multiplexed fluorescence in situ hybridization) brain tissue \
dataset in Neuroglancer — a scientific 3D brain imaging viewer. The page is already loaded.

Your task: Explore this dataset autonomously and report everything scientifically \
interesting you observe about the brain tissue, cell distributions, and structures.

─── STEP 1: Explore the Z-axis ───
You see a full-screen 2D cross-section (XY plane) through the brain volume.
Scroll the canvas to move through different Z depths (deeper into or out of the tissue).

Use scroll_at at viewport center:
  x=50, y=50, delta_x=0, delta_y=100   (advance Z — go deeper)
  x=50, y=50, delta_x=0, delta_y=-100  (retreat Z — go shallower)

If the canvas is completely black after waiting, this indicates a data or link error — \
send_msg_to_user with [ERROR] describing what you see, then send [EXIT].

After each scroll, observe what changed in the tissue. Use noop if the image is still \
updating. Explore at least 12 different Z positions. You may also navigate in X or Y \
using ArrowLeft, ArrowRight, ArrowUp, ArrowDown if you want to see different in-plane regions.

─── STEP 2: Report findings ───
After exploring, send_msg_to_user with [ANSWER] describing:
  - What tissue structures and cell patterns you observed
  - How the tissue appearance changed across the Z-axis
  - Anything scientifically notable (bright regions, layer boundaries, cell clusters, etc.)
Then send [EXIT].
""",
    },

    "registration": {
        "link_file": Path("/root/capsule/example_r2r_ng_link.txt"),
        "task": """\
You are viewing a 3D brain tissue registration result in Neuroglancer. \
The page is already loaded. There are two image layers overlaid:

  GREEN layer  — "Fixed Round Alignment Res"  : the reference scan (day 1)
  MAGENTA layer — "Loop 1 Aligned Moving"     : the moving scan (day 2), after alignment

Your task: Assess how well the two scans are aligned (registered) by examining \
where the green and magenta structures overlap or diverge across multiple Z slices. \
Report to the user any areas of inaccurate registration in the volume.

─── STEP 1: Explore the Z-axis ───
You see a full-screen 2D cross-section (XY plane) with both layers blended.
Scroll the canvas to move through different Z depths.

Use scroll_at at viewport center:
  x=50, y=50, delta_x=0, delta_y=100   (advance Z — go deeper)
  x=50, y=50, delta_x=0, delta_y=-100  (retreat Z — go shallower)

If any layer is completely black after waiting, this indicates a data or link error — \
send_msg_to_user with [ERROR] describing what you see, then send [EXIT].

After each scroll, look for:
  - Regions where green and magenta overlap well (good alignment → appears white/yellow)
  - Regions where they are offset (misalignment → distinct green and magenta fringes)
  - Whether misalignment is consistent or varies across Z
Use noop if the image is still updating. Explore at least 12 different Z positions.

─── STEP 2: Report findings ───
After exploring, send_msg_to_user with [ANSWER] describing:
  - Overall alignment quality (well-aligned, partially misaligned, or poorly aligned)
  - Which regions or Z depths show the best and worst alignment
  - The nature of any misalignment (shift, rotation, scale, local deformation)
  - Any notable structures where the two rounds agree or disagree
Then send [EXIT].
""",
    },

    "registration_large": {
        "link_file": Path("/root/capsule/thyme_r2r_ng_link.txt"),
        "patch_kwargs": {
            "post_scroll_sleep": 6.0,
            "post_goto_sleep": 15.0,
            "post_drag_sleep": 6.0,
            "scroll_ticks": 20,
            "zoom_ticks": 5,
        },
        "task": """\
You are viewing a large 3D brain tissue registration result in Neuroglancer. \
The page is already loaded. There are two image layers overlaid:

  GREEN layer   — "Fixed Round Alignment Res" : the reference scan (full resolution)
  MAGENTA layer — "Loop 5 Aligned Moving"     : the aligned moving scan

Your task: Assess registration quality at the NEURONAL level — check whether individual \
cells and fine structures are co-localized between the green and magenta channels. \
You will zoom in to see individual neurons, pan across the XY field, and scroll through Z.

─── DATA LOADING ───
This is a large high-resolution dataset streaming from cloud storage. After EVERY action \
check whether the image has finished loading:
  - Blurry, blank, or partially-rendered canvas → use noop to wait (repeat if still loading)
  - Sharp, fully-rendered canvas → proceed with your next action
  - After zooming in or panning, expect 5–15 seconds of loading; use multiple noops if needed
  - Completely black layer (not loading, just black) → data or link error; \
send_msg_to_user with [ERROR] describing what you see, then send [EXIT]

─── STEP 1: Overview assessment ───
At the current zoom level, observe the overall structure:
  - Does green-magenta overlap appear good (white/yellow regions) or poor (distinct fringes)?
  - Note any large-scale misalignment visible at this overview zoom

─── STEP 2: Zoom in for neuronal-level detail ───
Zoom in to see individual cells using scroll_at with delta_x (delta_y must be 0):
  scroll_at(x=50, y=50, delta_x=100, delta_y=0)   → zoom IN  (CTRL+scroll up)
  scroll_at(x=50, y=50, delta_x=-100, delta_y=0)  → zoom OUT (CTRL+scroll down)

Zoom in until individual cell bodies are visible (bright circular spots, ~5–20 µm).
Check whether green and magenta cell bodies co-localize or are offset.
Zoom out and back in to sample multiple areas.

─── STEP 3: Pan to sample the full XY field ───
Pan by dragging the canvas (mouse_drag_and_drop from one pixel to another) to check:
  - Different quadrants: top-left, top-right, bottom-left, bottom-right
  - Any regions with visually interesting structure (bright clusters, tissue edges)

After each pan, wait for data to load (noop if blurry or blank).

─── STEP 4: Explore the Z-axis ───
Scroll through at least 8 different Z depths using scroll_at with delta_y (delta_x must be 0):
  scroll_at(x=50, y=50, delta_x=0, delta_y=100)   → advance Z (go deeper)
  scroll_at(x=50, y=50, delta_x=0, delta_y=-100)  → retreat Z (go shallower)

Check whether alignment quality varies with Z depth.

─── STEP 5: Report findings ───
After exploring, send_msg_to_user with [ANSWER] describing:
  - Overall registration quality at the neuronal level (cell body co-localization)
  - Which XY regions and Z depths show the best and worst alignment
  - The nature of any misalignment (shift, rotation, scale, local deformation)
  - Estimated fraction of cells that appear well-aligned vs. offset
Then send [EXIT].
""",
    },
}


# ── URL helpers ──────────────────────────────────────────────────────────────

def build_ng_url(raw_url: str, layout: str = "xy") -> str:
    """Parse a Neuroglancer URL, set the layout, and re-encode."""
    if not raw_url.startswith(("http://", "https://")):
        raw_url = "https://" + raw_url
    if "#!" not in raw_url:
        return raw_url
    base, fragment = raw_url.split("#!", 1)
    state = json.loads(urllib.parse.unquote(fragment))
    state["layout"] = layout
    return base + "#!" + urllib.parse.quote(json.dumps(state), safe="")


# ── server lifecycle (mirrors run_capsule.py) ────────────────────────────────

def _evict_port(port: int) -> None:
    """Kill any process already listening on the given port."""
    hex_port = f"{port:04X}"
    for tcp_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for line in Path(tcp_file).read_text().splitlines()[1:]:
                parts = line.split()
                if len(parts) < 10:
                    continue
                if parts[1].split(":")[1].upper() == hex_port and parts[3] == "0A":
                    inode = parts[9]
                    for pid_dir in Path("/proc").iterdir():
                        if not pid_dir.name.isdigit():
                            continue
                        try:
                            for fd in (pid_dir / "fd").iterdir():
                                if f"socket:[{inode}]" == os.readlink(fd):
                                    pid = int(pid_dir.name)
                                    print(f"  Evicting stale PID {pid} on port {port}")
                                    os.kill(pid, signal.SIGKILL)
                        except (PermissionError, FileNotFoundError):
                            pass
        except FileNotFoundError:
            pass
    time.sleep(1)


def _wait_for_server(url: str, timeout: int = 300) -> bool:
    """Poll url until any HTTP response — means the server is up."""
    for _ in range(timeout):
        try:
            httpx.get(url, timeout=2)
            return True
        except httpx.ConnectError:
            time.sleep(1)
        except Exception:
            return True
    return False


def start_molmoweb_server() -> subprocess.Popen:
    _evict_port(MOLMOWEB_PORT)
    existing_pypath = os.environ.get("PYTHONPATH", "")
    pypath = "/code/lib/molmoweb:" + existing_pypath if existing_pypath else "/code/lib/molmoweb"
    env = {
        **os.environ,
        "CKPT": MOLMOWEB_CKPT,
        "PREDICTOR_TYPE": "native",
        "PYTHONPATH": pypath,
        "HF_HUB_OFFLINE": "1",
    }
    proc = subprocess.Popen(
        [
            "/opt/conda/bin/uvicorn", "agent.fastapi_model_server:app",
            "--host", "0.0.0.0", "--port", str(MOLMOWEB_PORT),
        ],
        env=env,
    )
    print(f"  Waiting for MolmoWeb server (model load ~60-120s)...")
    if not _wait_for_server(f"http://127.0.0.1:{MOLMOWEB_PORT}/", timeout=300):
        proc.kill()
        raise RuntimeError("MolmoWeb server did not start within 300s")
    print("  MolmoWeb server ready.")
    return proc


def kill_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# ── main ─────────────────────────────────────────────────────────────────────

def patch_env_for_ng(
    env,
    post_scroll_sleep: float = 3.0,
    post_goto_sleep: float = 12.0,
    post_drag_sleep: float = 5.0,
    scroll_ticks: int = 20,
    zoom_ticks: int = 5,
) -> None:
    """Monkey-patch SimpleEnv.step() to sleep after scroll/goto/drag actions and
    fire multiple wheel events per scroll_at to advance more than 1 Z slice.

    Neuroglancer streams zarr data from S3 asynchronously. domcontentloaded
    (MolmoWeb's default wait) fires immediately after JS parses — long before
    chunks arrive. This patch adds explicit sleeps after the actions that
    trigger new data fetches, so the screenshot captures rendered data rather
    than a blank/stale canvas.

    CTRL+scroll convention for zoom (scroll_at with delta_x != 0):
        delta_x > 0  →  CTRL+scroll up   (zoom in)
        delta_x < 0  →  CTRL+scroll down (zoom out)
    Normal Z navigation uses delta_y != 0, delta_x == 0.

    post_scroll_sleep: seconds to wait after scroll_at (Z-navigation or zoom)
    post_goto_sleep:   seconds to wait after goto (full URL navigation)
    post_drag_sleep:   seconds to wait after mouse_drag_and_drop (XY pan)
    scroll_ticks:      number of wheel events per Z scroll_at (1 event = 1 Z slice)
    zoom_ticks:        number of wheel events per CTRL+scroll zoom action
    """
    import time as _time
    from agent.actions import Goto, ScrollAt, KeyboardPress, MouseDragAndDrop

    _orig_step = env.step.__func__  # unbound method

    def _ng_step(self, action):
        if isinstance(action, ScrollAt):
            self.page.mouse.move(action.x, action.y)
            if action.delta_x != 0:
                # CTRL+scroll for zoom: delta_x > 0 = zoom in (wheel up = negative deltaY)
                zoom_delta = -100 if action.delta_x > 0 else 100
                self.page.keyboard.down("Control")
                for _ in range(zoom_ticks):
                    self.page.mouse.wheel(0, zoom_delta)
                self.page.keyboard.up("Control")
            else:
                # Neuroglancer advances exactly 1 Z slice per WheelEvent regardless
                # of deltaY magnitude — fire scroll_ticks events to move N slices.
                for _ in range(scroll_ticks):
                    self.page.mouse.wheel(action.delta_x, action.delta_y)
            _time.sleep(post_scroll_sleep)
            return self._get_obs()

        obs = _orig_step(self, action)
        if isinstance(action, Goto):
            _time.sleep(post_goto_sleep)
        elif isinstance(action, KeyboardPress):
            _time.sleep(post_scroll_sleep)
        elif isinstance(action, MouseDragAndDrop):
            _time.sleep(post_drag_sleep)
        return obs

    import types
    env.step = types.MethodType(_ng_step, env)


def save_trajectory_video(traj, output_path: Path, fps: float = 0.5) -> None:
    """Assemble per-step screenshots into an MP4 (1 frame per agent step)."""
    import imageio
    import numpy as np

    frames = [
        np.array(step.state.img)
        for step in traj.steps
        if step.state and step.state.img is not None
    ]
    if not frames:
        print("  (no frames to save)")
        return

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264", quality=7)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  Trajectory video ({len(frames)} frames @ {fps} fps): {output_path}")


def run(url: str | None, max_steps: int, no_server: bool, record_video: bool, mode: str = "explore") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mode_cfg = MODES[mode]

    # Build NG URL
    raw_url = url or mode_cfg["link_file"].read_text().strip()
    ng_url = build_ng_url(raw_url, layout="xy")
    base, fragment = ng_url.split("#!", 1)
    json_url = base + "#!" + json.dumps(json.loads(urllib.parse.unquote(fragment)))
    print(json_url)
    print(ng_url)

    task = mode_cfg["task"]

    server_proc = None
    if not no_server:
        print("\n[1/3] Starting MolmoWeb server...")
        server_proc = start_molmoweb_server()
    else:
        print("\n[1/3] --no-server: assuming MolmoWeb already running on port 8001")

    try:
        from inference.client import MolmoWeb
        from agent.actions import SendMsgToUser

        print("\n[2/3] Launching MolmoWeb agent...")
        pw_video_dir = str(RESULTS_DIR / "pw_video") if record_video else None
        if pw_video_dir:
            Path(pw_video_dir).mkdir(parents=True, exist_ok=True)

        client = MolmoWeb(
            endpoint=f"http://127.0.0.1:{MOLMOWEB_PORT}",
            local=True,
            headless=True,
            verbose=True,
            record_video_dir=pw_video_dir,
        )

        # Wrap _create_env so every SimpleEnv instance gets the NG-aware
        # post-action sleeps and scroll multipliers injected automatically.
        import types
        _orig_create_env = client._create_env
        patch_kwargs = mode_cfg.get("patch_kwargs", {})
        def _create_env_patched(start_url="about:blank"):
            env = _orig_create_env(ng_url)
            patch_env_for_ng(env, **patch_kwargs)
            return env
        client._create_env = _create_env_patched

        traj = client.run(task, max_steps=max_steps)

        # Finalize Playwright video: must close context before the file is written
        if record_video and client.env is not None:
            with client._pw_context():
                client.env.context.close()
                client.env.context = None

        # ── save results ─────────────────────────────────────────────────────
        print(f"\n[3/3] Saving results ({len(traj.steps)} steps)...")

        html_path = RESULTS_DIR / "trajectory.html"
        traj.save_html(output_path=str(html_path), query="mFISH brain scan exploration")
        print(f"  Trajectory HTML: {html_path}")

        # Trajectory video (always)
        save_trajectory_video(traj, RESULTS_DIR / "trajectory_video.mp4", fps=0.5)

        # Rename Playwright video if recorded
        if record_video and pw_video_dir:
            pw_files = list(Path(pw_video_dir).glob("*.webm"))
            if pw_files:
                dest = RESULTS_DIR / "browser_video.webm"
                pw_files[0].rename(dest)
                print(f"  Browser video (real-time): {dest}")

        # Print agent messages
        msgs = [
            (i + 1, step.prediction.action.msg)
            for i, step in enumerate(traj.steps)
            if step.prediction and isinstance(step.prediction.action, SendMsgToUser)
        ]
        if msgs:
            print("\n" + "=" * 60)
            print("AGENT MESSAGES:")
            print("=" * 60)
            for step_num, msg in msgs:
                print(f"\n[Step {step_num}]\n{msg}")
            print("=" * 60)
        else:
            print("\n(No send_msg_to_user actions recorded)")

        # Action type summary
        from collections import Counter
        action_types = Counter(
            type(step.prediction.action).__name__
            for step in traj.steps
            if step.prediction
        )
        print("\nAction summary:")
        for action, count in action_types.most_common():
            print(f"  {action}: {count}")

    finally:
        try:
            resp = httpx.get(f"http://127.0.0.1:{MOLMOWEB_PORT}/stats", timeout=5)
            s = resp.json()
            print(f"\nToken usage ({s['calls']} calls):")
            print(f"  Input  — text: {s['tokens_in_text']:,}  image: {s['tokens_in_image']:,}  total: {s['tokens_in_text'] + s['tokens_in_image']:,}")
            print(f"  Output — text: {s['tokens_out']:,}")
        except Exception:
            pass

        if server_proc is not None:
            print("\nStopping MolmoWeb server...")
            kill_server(server_proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MolmoWeb Neuroglancer Explorer")
    parser.add_argument(
        "--url",
        default=None,
        help="Neuroglancer URL to explore (default: example_ng_link.txt)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum agent steps (default: 5)",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Skip server startup (use if MolmoWeb server is already running on port 8001)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Also record a real-time browser video via Playwright (saved as browser_video.webm)",
    )
    parser.add_argument(
        "--mode",
        default="explore",
        choices=list(MODES.keys()),
        help="Task mode: 'explore' (single-channel mFISH exploration), "
             "'registration' (two-channel alignment assessment), or "
             "'registration_large' (neuronal-level registration with zoom/pan on large dataset). "
             "Default: explore",
    )
    args = parser.parse_args()
    run(url=args.url, max_steps=args.max_steps, no_server=args.no_server, record_video=args.record_video, mode=args.mode)
