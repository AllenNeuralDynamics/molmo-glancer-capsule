# Demo Plan: Neuron Counting in HCR Brain Tissue via MolmoWeb + neuroglancer-chat

## What the data is

The example link points to a single-tile AIND HCR (Hybridization Chain Reaction) light-sheet scan:

```
s3://aind-open-data/HCR_772643-3a-1_2025-03-19_10-00-00/
  SPIM.ome.zarr/Tile_X_0000_Y_0000_Z_0000_ch_405.zarr/
```

- **Modality**: SPIM (Single Plane Illumination / light-sheet) fluorescence
- **Channel 405nm**: Nuclear stain (DAPI-equivalent) — cell nuclei appear as discrete bright blobs
- **Voxel resolution**: 1 µm isotropic (x, y, z)
- **Format**: OME-Zarr on a public S3 bucket (`aind-open-data`) — no credentials needed
- **Layer type in NG**: `image` (raw fluorescence, not segmentation)

The data contains cell nuclei visible as bright puncta. "Counting neurons" here means **detecting nuclei in the fluorescence volume** — this is a classical blob-detection / connected-components problem, not an LLM task.

---

## Do we need external LLM access?

**Short answer: Yes, but only for one optional component (neuroglancer-chat).**

| Component | LLM required? | Which one? |
|-----------|--------------|-----------|
| MolmoWeb model server | No — local | `allenai/MolmoWeb-4B` on GPU |
| Cell counting pipeline | No | Pure image analysis (scipy/skimage) |
| `NeuroglancerState` library (URL generation) | No | Direct Python import |
| neuroglancer-chat **chat interface** | **Yes** | OpenAI API key (`OPENAI_API_KEY`) |
| neuroglancer-chat **as a library** | No | `NeuroglancerState` can be imported directly without the chat backend |

The OpenAI dependency in neuroglancer-chat is confined to the chat loop in `backend/main.py`. The `NeuroglancerState` class (`backend/tools/neuroglancer_state.py`) is a standalone URL parser/builder with no LLM dependency. We can use it directly in a script to generate annotated NG URLs without ever starting the FastAPI backend or needing an API key.

**If `OPENAI_API_KEY` is unavailable**: run the pipeline as a standalone Python script using `NeuroglancerState` directly. Skip the chat UI entirely. MolmoWeb still works — it only uses the local model.

**If `OPENAI_API_KEY` is available**: use the full neuroglancer-chat backend for richer interaction (e.g., natural language queries like "show me the 10 densest regions").

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: Parse NG URL → extract S3 data path                        │
│  Tool: NeuroglancerState.from_url()  (neuroglancer-chat library)    │
│  Output: s3://aind-open-data/.../ch_405.zarr  +  voxel resolution   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: Load a representative volume tile from S3                  │
│  Tool: zarr + s3fs (Python, no GPU, no LLM)                         │
│  Strategy: read a sub-region (e.g., 512×512×100 voxels) at          │
│            the coarsest resolution level of the OME-Zarr pyramid    │
│  Output: 3D numpy array                                             │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: Detect nuclei (blob detection)                             │
│  Tool: skimage.feature.blob_dog  OR  scipy connected-components     │
│  Method: threshold (Otsu) → label connected regions → filter by     │
│          size (nuclei ~5–15 µm diameter → 5–15 voxels radius)       │
│  Output: list of (x, y, z) centroids in voxel coordinates + count  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: Generate annotated Neuroglancer URL                        │
│  Tool: NeuroglancerState  (neuroglancer-chat library)               │
│  Method: start from the original state, call ng_annotations_add()  │
│          for each detected centroid, serialize to URL               │
│  Output: new NG URL with a point-annotation layer overlaid          │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5: Visual verification via MolmoWeb                           │
│  Tool: MolmoWeb inference client + Playwright/Chromium              │
│  Method: client.run(task, max_steps=5) where task is:               │
│    "Navigate to <annotated_url>, wait for the volume to render,     │
│     take a screenshot showing the annotation dots on the tissue,    │
│     then report how many annotations are visible."                  │
│  Output: trajectory HTML + screenshots  →  /results/               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Roles

### neuroglancer-chat (`NeuroglancerState`)
- **Role**: URL plumbing — parse the input link, add the annotation layer, regenerate a valid NG URL
- **Used as**: a Python library (no FastAPI server required for the core demo)
- **Key calls**:
  ```python
  from neuroglancer_chat.backend.tools.neuroglancer_state import NeuroglancerState
  state = NeuroglancerState.from_url(open("example_ng_link.txt").read().strip())
  # ... add annotations ...
  annotated_url = state.to_url()
  ```

### Image analysis (standalone Python)
- **Role**: the actual cell counting — this is where "neurons" come from
- **No LLM, no GPU required** — just zarr, s3fs, numpy, skimage
- This step is deliberately kept simple for the demo; a real pipeline would use a segmentation model (e.g., Cellpose, StarDist)

### MolmoWeb
- **Role**: visual verification and screenshot capture
- **Why needed**: NG renders via WebGL — you cannot verify the annotation overlay without actually rendering it in a browser
- **What it does**: navigates to the annotated URL, waits for render, screenshots, reports visible annotation count
- **Uses**: local MolmoWeb-4B on Tesla T4 (no external API)

### neuroglancer-chat FastAPI backend (optional for demo)
- **Role**: natural-language interface for interactive exploration after the count
- **When useful**: "show me the 10 densest 50-µm cubes" or "add a scale bar annotation at the center"
- **Requires**: `OPENAI_API_KEY`

---

## Implementation Steps

### Prerequisites

```bash
# Install deps
cd /code/neuroglancer-chat && uv sync
cd /code/molmoweb && uv sync && uv run playwright install --with-deps chromium

# Download model weights (4B only — 8B exceeds T4 VRAM)
cd /code/molmoweb
bash scripts/download_weights.sh allenai/MolmoWeb-4B
mv checkpoints/MolmoWeb-4B /scratch/checkpoints/MolmoWeb-4B

# Install Zarr + S3 access
pip install zarr s3fs scikit-image   # or add to uv env
```

### Step 1–4: Counting script

A single Python script (`/code/count_neurons.py`) handles everything up to URL generation:

```python
# Pseudocode outline — implement in /code/count_neurons.py
import zarr, s3fs, numpy as np
from skimage.feature import blob_dog
from skimage.filters import threshold_otsu
from neuroglancer_chat.backend.tools.neuroglancer_state import NeuroglancerState

# 1. Parse URL
state = NeuroglancerState.from_url(open("/root/capsule/example_ng_link.txt").read().strip())
s3_path = state.layers[0]["source"].replace("zarr://", "")  # s3://aind-open-data/...

# 2. Load volume (coarsest pyramid level for speed)
fs = s3fs.S3FileSystem(anon=True)
store = s3fs.S3Map(root=s3_path, s3=fs)
z = zarr.open(store, mode="r")
# OME-Zarr: z[0] is full res, z[1] is 2x downsampled, etc.
# Load a manageable tile at low resolution
vol = np.array(z["1"][0, 0, :100, :512, :512])  # (z, y, x) chunk

# 3. Detect blobs
thresh = threshold_otsu(vol)
blobs = blob_dog(vol.astype(float), min_sigma=3, max_sigma=8, threshold=thresh * 0.5)
# blobs shape: (N, 4) → (z, y, x, sigma)
centroids_vox = blobs[:, :3]
count = len(centroids_vox)
print(f"Detected {count} nuclei")

# 4. Add annotation layer
# (convert voxel coords to physical coords using state dimensions)
# state.ng_annotations_add(name="detected_nuclei", points=centroids_physical, color="#ff0000")
annotated_url = state.to_url()
with open("/results/annotated_ng_url.txt", "w") as f:
    f.write(annotated_url)
```

> **Note**: The exact zarr array path structure (`z["1"][0, 0, ...]`) depends on the OME-Zarr layout of this specific dataset. Inspect `z.tree()` first to see the array hierarchy and dimension order.

### Step 5: MolmoWeb verification

```python
# Run after the MolmoWeb model server is up (Terminal: bash scripts/start_server.sh)
from inference import MolmoWeb

annotated_url = open("/results/annotated_ng_url.txt").read().strip()
client = MolmoWeb(endpoint="http://127.0.0.1:8001", local=True, headless=True)

task = f"""
Navigate to this Neuroglancer URL: {annotated_url}
Wait up to 10 seconds for the 3D volume to render (it loads from S3).
Take a screenshot once you can see the grayscale tissue volume with colored dot annotations on it.
Report the approximate number of annotation dots visible in the current view.
"""

traj = client.run(task, max_steps=8)
traj.save_html(query="neuron_count_verification")
# Output: /code/molmoweb/inference/htmls/neuron_count_verification_*.html
```

---

## Expected Output

| Output | Location | Description |
|--------|----------|-------------|
| Cell count (int) | stdout / log | Number of detected nuclei in the sampled volume |
| Annotated NG URL | `/results/annotated_ng_url.txt` | NG link with point annotations at each detected cell |
| Verification trajectory | `/code/molmoweb/inference/htmls/` | HTML with screenshots showing annotations on tissue |
| PNG screenshots | `/results/screenshots/` | Individual step screenshots from MolmoWeb |

---

## Caveats and Scope Limitations

**This counts nuclei in a sub-region, not the full tile.** The full tile at 1 µm/voxel is likely multi-GB. The demo samples a manageable crop. A production count would tile across the full volume.

**405nm channel = all nuclei, not just neurons.** HCR brain tissue contains neurons, glia, endothelial cells, etc. — all have nuclei that stain with this channel. To isolate neurons specifically you'd need either a neuron-specific marker channel (e.g., NeuN) or a segmentation model trained on cell types. This demo counts all nuclei as a proxy.

**Blob detection ≠ segmentation.** `blob_dog` is a good first-pass detector but will over-count in dense regions and miss touching nuclei. For publication-quality counts, use Cellpose or StarDist. The demo uses blob detection for simplicity and zero additional model downloads.

**MolmoWeb's visual count is a sanity check, not the ground truth.** The programmatic count (Step 3) is the authoritative number. MolmoWeb verifies that annotations actually rendered in the browser and are spatially coherent with the tissue — it catches pipeline bugs (wrong coordinate space, wrong URL encoding) that wouldn't show up in Python alone.

---

## Startup Sequence for Full Demo

```bash
# Terminal 1 — MolmoWeb model server
cd /code/molmoweb
export CKPT=/scratch/checkpoints/MolmoWeb-4B
bash scripts/start_server.sh           # PREDICTOR_TYPE=native, PORT=8001

# Terminal 2 — Run the counting + annotation script
cd /code
python count_neurons.py

# Terminal 3 — MolmoWeb visual verification (once model server is ready)
cd /code/molmoweb
uv run python verify_with_molmoweb.py
```

If using the neuroglancer-chat **interactive backend** for exploration after the count:

```bash
# Terminal 4 — neuroglancer-chat backend (requires OPENAI_API_KEY)
cd /code/neuroglancer-chat/src/neuroglancer_chat
export OPENAI_API_KEY="sk-..."
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 5 — Panel UI
export BACKEND="http://127.0.0.1:8000"
uv run python -m panel serve panel/panel_app.py --port 8006 --address 127.0.0.1 \
  --allow-websocket-origin=127.0.0.1:8006
```
