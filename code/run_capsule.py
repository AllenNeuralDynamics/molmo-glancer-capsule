"""
Neuroglancer Alignment Grid Search — MVP
-----------------------------------------
Pipeline:
  1. Load NG URL from example_ng_link.txt
  2. Grid-search z-positions using NeuroglancerState
  3. Screenshot each with Playwright (networkidle wait, browser reused)
  4. CV-rank by sharpness → keep top 5
  5. Start MolmoWeb model server → visual interpretation of top 5
  6. Start OLMo via vllm → synthesize final recommendation
  7. Save all outputs to /results/grid_search/

Assumes:
  /scratch/checkpoints/MolmoWeb-4B
  /scratch/checkpoints/OLMo-3-7B-Instruct
  PLAYWRIGHT_BROWSERS_PATH=/scratch/ms-playwright (set below)
"""

import json, os, sys, time, subprocess, urllib.parse
from pathlib import Path

# molmoweb editable install only registers 'inference'; add root to path
# so that 'utils' and 'agent' packages are importable directly.
sys.path.insert(0, "/code/lib/molmoweb")

import httpx
import numpy as np
from PIL import Image
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from neuroglancer_chat.backend.tools.neuroglancer_state import NeuroglancerState
from utils.envs.browser_env import SimpleEnv
from agent.model_backends import FastApiActionPredictor
from skimage.filters import laplace
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MOLMOWEB_CKPT = "/scratch/checkpoints/MolmoWeb-4B"
OLMO_CKPT     = "/scratch/checkpoints/OLMo-3-7B-Instruct"
MOLMOWEB_PORT = 8001
OLMO_PORT     = 8002
RESULTS_DIR   = Path("/results/grid_search")
NG_LINK_FILE      = Path("/root/capsule/example_ng_link.txt")
NG_STATE_TEMPLATE = Path("/root/capsule/ground_truth_ng_link_z_16.json")  # full ground-truth state

TOP_N   = 5                    # top CV-ranked positions sent to MolmoWeb
N_STEPS = 10                   # number of z-positions to sample

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/scratch/ms-playwright")
os.environ.setdefault("MOLMO_DATA_DIR", "/scratch/molmo_data")
os.environ.setdefault("HF_HOME", "/scratch/huggingface")

DESCRIBE_PROMPT = (
    "Describe what you see in this fluorescence microscopy image. "
    "What structures are visible? Is the tissue in focus and well-resolved? "
    "Rate the image clarity on a scale of 1-10."
)

OLMO_SYSTEM = (
    "You are analyzing descriptions of fluorescence microscopy images taken at "
    "different z-positions in a 3D light-sheet volume. "
    "Your job is to rank them by image quality and tissue focus."
)


# ---------------------------------------------------------------------------
# Step 1 — Grid URL generation
# ---------------------------------------------------------------------------
def get_z_extent() -> int:
    """Read the zarr source from the ground-truth NG state and return the z-axis voxel count."""
    import zarr, s3fs
    template = json.loads(NG_STATE_TEMPLATE.read_text())
    layers = template.get("layers", [])
    source_url = None
    for layer in layers:
        src = layer.get("source", "")
        if isinstance(src, dict):
            src = src.get("url", "")
        if src:
            source_url = src
            break
    if source_url is None:
        raise ValueError("No layer source found in NG state template")
    path = source_url.removeprefix("zarr://")
    if path.startswith("s3://"):
        bucket_key = path[len("s3://"):]
        fs = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(bucket_key, s3=fs)
    else:
        store = path
    z_arr = zarr.open(store, mode="r")
    # Resolution level "0" is full-res; shape is (t, c, z, y, x)
    shape = z_arr["0"].shape
    z_size = shape[2]
    print(f"  Zarr shape (full res): {shape}  →  z extent = {z_size} voxels")
    return z_size


def make_grid_urls(z_extent: int) -> list[tuple[int, str]]:
    """Build grid URLs by cloning the ground-truth state template and varying z."""
    template = json.loads(NG_STATE_TEMPLATE.read_text())
    # Keep x, y, t from template; only vary z (index 2)
    pos = template.get("position", [0, 0, 0, 0])
    cx, cy, ct = pos[0], pos[1], pos[3] if len(pos) > 3 else 0
    step = max(1, z_extent // N_STEPS)
    z_positions = list(range(0, z_extent, step))[:N_STEPS]
    ng_base = "https://neuroglancer-demo.appspot.com"
    results = []
    for z in z_positions:
        state = json.loads(json.dumps(template))  # deep copy
        state["position"] = [cx, cy, float(z), ct]
        encoded = urllib.parse.quote(json.dumps(state, separators=(",", ":")), safe="")
        raw_url = ng_base + "/#!" + encoded
        results.append((z, raw_url))
    print(f"  z positions: {z_positions}")
    return results


# ---------------------------------------------------------------------------
# Step 2 — Screenshot all URLs (browser kept alive between navigations)
# ---------------------------------------------------------------------------
def screenshot_all(grid: list[tuple]) -> list[tuple]:
    results = []
    env = SimpleEnv(headless=True, viewport_width=1280, viewport_height=720)
    try:
        # First URL: reset() launches Chromium and waits for networkidle internally
        first_z, first_url = grid[0]
        obs, _ = env.reset(start_url=first_url)
        time.sleep(5)  # allow Neuroglancer to fetch and render zarr chunks after networkidle
        obs = env._get_obs()
        results.append((first_z, first_url, obs["screenshot"]))
        print(f"  z={first_z:5}  ✓")

        # Remaining URLs: reuse the open browser, navigate + networkidle wait
        for z, url in grid[1:]:
            env.page.goto(url, wait_until="domcontentloaded")
            try:
                env.page.wait_for_load_state("networkidle", timeout=20_000)
            except PlaywrightTimeoutError:
                pass  # timeout — capture whatever has rendered
            time.sleep(5)  # allow Neuroglancer to fetch and render zarr chunks after networkidle
            obs = env._get_obs()
            results.append((z, url, obs["screenshot"]))
            print(f"  z={z:5}  ✓")
    finally:
        env.close()
    return results


# ---------------------------------------------------------------------------
# Step 3 — CV scoring
# ---------------------------------------------------------------------------
def cv_score(img: np.ndarray) -> dict:
    g = rgb2gray(img.astype(float) / 255.0)
    return {
        "sharpness": float(laplace(g).var()),
        "contrast":  float(g.std()),
        "entropy":   float(shannon_entropy(g)),
    }


# ---------------------------------------------------------------------------
# Step 4 — Server lifecycle helpers
# ---------------------------------------------------------------------------
def _evict_port(port: int):
    """Kill any process already listening on port so our server can bind."""
    import signal
    hex_port = f"{port:04X}"
    killed = []
    for tcp_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            for line in Path(tcp_file).read_text().splitlines()[1:]:
                parts = line.split()
                if parts[1].split(":")[1].upper() == hex_port and parts[3] == "0A":
                    inode = parts[9]
                    # find pid owning this inode
                    for pid_dir in Path("/proc").iterdir():
                        if not pid_dir.name.isdigit():
                            continue
                        try:
                            for fd in (pid_dir / "fd").iterdir():
                                if f"socket:[{inode}]" == os.readlink(fd):
                                    pid = int(pid_dir.name)
                                    print(f"  Evicting stale PID {pid} on port {port}")
                                    os.kill(pid, signal.SIGKILL)
                                    killed.append(pid)
                        except (PermissionError, FileNotFoundError):
                            pass
        except FileNotFoundError:
            pass
    if killed:
        time.sleep(2)


def _wait_any_response(url: str, timeout: int = 180) -> bool:
    """Poll url until any HTTP response (including 4xx) — means server is up."""
    for _ in range(timeout):
        try:
            httpx.get(url, timeout=2)
            return True
        except httpx.ConnectError:
            time.sleep(1)
        except Exception:
            return True  # non-connection error still means server responded
    return False


def start_molmoweb_server() -> subprocess.Popen:
    _evict_port(MOLMOWEB_PORT)
    # PYTHONPATH ensures 'agent' and 'utils' packages are findable in the subprocess
    existing_pypath = os.environ.get("PYTHONPATH", "")
    pypath = "/code/lib/molmoweb:" + existing_pypath if existing_pypath else "/code/lib/molmoweb"
    env = {
        **os.environ,
        "CKPT": MOLMOWEB_CKPT,
        "PREDICTOR_TYPE": "native",
        "PYTHONPATH": pypath,
    }
    proc = subprocess.Popen(
        [
            "/opt/conda/bin/uvicorn", "agent.fastapi_model_server:app",
            "--host", "0.0.0.0", "--port", str(MOLMOWEB_PORT),
        ],
        env=env,
    )
    print("  Waiting for MolmoWeb server (model load ~60-120s)...")
    if not _wait_any_response(f"http://127.0.0.1:{MOLMOWEB_PORT}/", timeout=300):
        proc.kill()
        raise RuntimeError("MolmoWeb server did not start within 300s")
    return proc


def start_olmo_server() -> subprocess.Popen:
    _evict_port(OLMO_PORT)
    # vllm is installed in its own venv at /scratch/vllm-venv (see _dev_startup2.sh)
    vllm_python = "/scratch/vllm-venv/bin/python"
    # FlashInfer default workspace is 394 MiB which doesn't fit on T4 after model load.
    # Reduce to 32 MiB via the env var vllm 0.18 actually reads.
    env = {**os.environ, "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE": str(32 * 1024 * 1024)}
    proc = subprocess.Popen([
        vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", OLMO_CKPT,
        "--served-model-name", "OLMo-3-7B-Instruct",
        "--port", str(OLMO_PORT),
        "--host", "0.0.0.0",
        "--enforce-eager",             # skip torch.compile — T4 has no VRAM headroom
        "--max-model-len", "512",      # minimise profile-run activation memory on T4
        "--gpu-memory-utilization", "0.99",  # 1.0 fails if parent holds ~100 MiB CUDA context
    ], env=env)
    print("  Waiting for OLMo/vllm server (model load ~60-120s)...")
    if not _wait_any_response(f"http://127.0.0.1:{OLMO_PORT}/health", timeout=180):
        proc.kill()
        raise RuntimeError("OLMo server did not start within 180s")
    return proc


def kill_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)  # ensure SIGKILL is reaped before continuing


# ---------------------------------------------------------------------------
# Step 5 — MolmoWeb visual interpretation
# ---------------------------------------------------------------------------
def _extract_thought(result) -> str:
    """Pull the 'thought' string out of whatever the predictor returns."""
    if result is None:
        return "(no response)"
    if isinstance(result, dict):
        return result.get("thought", str(result))
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                return parsed.get("thought", result)
        except json.JSONDecodeError:
            pass
        return result
    return str(result)


def interpret_with_molmoweb(top_shots: list[tuple]) -> dict[int, str]:
    predictor = FastApiActionPredictor(endpoint=f"http://127.0.0.1:{MOLMOWEB_PORT}")
    interpretations = {}
    for z, _url, img in top_shots:
        result = predictor.predict(DESCRIBE_PROMPT, img)
        thought = _extract_thought(result)
        interpretations[int(z)] = thought
        print(f"  z={z:5}: {thought[:100]}...")
    return interpretations


# ---------------------------------------------------------------------------
# Step 6 — OLMo synthesis
# ---------------------------------------------------------------------------
def synthesize_with_olmo(interpretations: dict, scored: list) -> str:
    cv_lines = "\n".join(
        f"  z={z}: sharpness={s['sharpness']:.5f}, contrast={s['contrast']:.3f}"
        for z, _, _, s in scored[:TOP_N]
    )
    desc_lines = "\n".join(
        f"  z={z}: {desc}"
        for z, desc in interpretations.items()
    )
    prompt = (
        f"CV metrics for top-{TOP_N} positions (by sharpness):\n{cv_lines}\n\n"
        f"MolmoWeb visual descriptions:\n{desc_lines}\n\n"
        "Which z-position shows the best image quality and tissue structure? "
        "Provide a ranked recommendation with brief reasoning."
    )
    resp = httpx.post(
        f"http://127.0.0.1:{OLMO_PORT}/v1/chat/completions",
        json={
            "model": "OLMo-3-7B-Instruct",
            "messages": [
                {"role": "system", "content": OLMO_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": 512,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_url = NG_LINK_FILE.read_text().strip()
    print(f"Base URL: {base_url[:100]}...")

    # --- 1. Grid ---
    print(f"\n[1/6] Reading zarr volume shape and generating grid URLs...")
    z_extent = get_z_extent()
    grid = make_grid_urls(z_extent)

    # Write grid URLs to file so they can be inspected / opened in a browser
    grid_urls_path = RESULTS_DIR / "grid_urls.txt"
    with open(grid_urls_path, "w") as f:
        for z, url in grid:
            f.write(f"z={z:5}  {url}\n")
    print(f"  Grid URLs saved to {grid_urls_path}")

    # --- 2. Screenshots ---
    print(f"\n[2/6] Taking {len(grid)} screenshots (networkidle wait)...")
    screenshots = screenshot_all(grid)

    # --- 3. CV rank ---
    print(f"\n[3/6] CV scoring...")
    scored = sorted(
        [(z, url, img, cv_score(img)) for z, url, img in screenshots],
        key=lambda x: x[3]["sharpness"],
        reverse=True,
    )

    print("\n  CV Rankings:")
    for rank, (z, _, _, s) in enumerate(scored):
        print(f"    {rank+1:2}. z={z:5}  "
              f"sharpness={s['sharpness']:.5f}  "
              f"contrast={s['contrast']:.3f}  "
              f"entropy={s['entropy']:.2f}")

    # Save all PNGs in CV-ranked order
    for rank, (z, url, img, _) in enumerate(scored):
        Image.fromarray(img).save(RESULTS_DIR / f"rank{rank+1:02d}_z{int(z):04d}.png")

    top5 = scored[:TOP_N]

    # --- 4+5. MolmoWeb ---
    print(f"\n[4/6] Starting MolmoWeb server ({MOLMOWEB_CKPT})...")
    molmoweb_proc = start_molmoweb_server()
    try:
        print(f"\n[5/6] Interpreting top-{TOP_N} screenshots with MolmoWeb...")
        interpretations = interpret_with_molmoweb(
            [(z, url, img) for z, url, img, _ in top5]
        )
    finally:
        print("\n  Stopping MolmoWeb server...")
        kill_server(molmoweb_proc)

    (RESULTS_DIR / "molmo_interpretations.json").write_text(
        json.dumps({str(k): v for k, v in interpretations.items()}, indent=2)
    )

    # --- 6. OLMo ---
    print(f"\n[6/6] Starting OLMo server ({OLMO_CKPT})...")
    olmo_proc = start_olmo_server()
    try:
        print("  Synthesizing recommendation with OLMo...")
        recommendation = synthesize_with_olmo(interpretations, scored)
    finally:
        print("\n  Stopping OLMo server...")
        kill_server(olmo_proc)

    (RESULTS_DIR / "olmo_recommendation.txt").write_text(recommendation)

    # Full results table
    results_table = [
        {
            "rank": rank + 1,
            "z": z,
            "url": url,
            "molmo_description": interpretations.get(int(z), ""),
            **score,
        }
        for rank, (z, url, _, score) in enumerate(scored)
    ]
    (RESULTS_DIR / "results.json").write_text(json.dumps(results_table, indent=2))

    print(f"\n{'='*60}")
    print("OLMo Recommendation:")
    print(recommendation)
    print(f"\nOutputs saved to {RESULTS_DIR}/")
    print("  rank01_z*.png ... rank10_z*.png  (CV-ranked screenshots)")
    print("  molmo_interpretations.json")
    print("  olmo_recommendation.txt")
    print("  results.json")


if __name__ == "__main__":
    run()
