# CLAUDE.md

## Session onboarding — read these first

| File | What it contains |
|------|-----------------|
| `CLAUDE.md` | This file — workspace layout, hardware, commands |
| `v3_plan.md` | **v3 design: agent loop, visual input, GPU config, implementation plan** |
| `TASKS.md` | Implementation task list with checkboxes — track progress here |
| `REFERENCES.md` | Research sources: Molmo2, Neuroglancer, VRAM math |
| `SETUP_NOTES.md` | Known env issues and fixes |

## Workspace Layout

This git repo (`molmo-glancer-capsule`) lives at `/root/capsule/`.

| Path | Purpose |
|------|---------|
| `code/molmo_glancer.py` | **Main v3 pipeline** — agent loop, actions, prompt construction |
| `code/gpu_config.py` | GPU detection, profiles, model loading |
| `code/visual_capture.py` | Playwright: clean state, CSS injection, canvas screenshot, scans |
| `code/volume_info.py` | Volume metadata discovery from zarr sources |
| `code/run_v3` | Shell entry point (env vars, logging) |
| `code/_dev_startup.sh` | Dep install script (pip-based; run once at capsule startup) |
| `code/_download_weights.sh` | Weight download script (requires `HF_TOKEN`) |
| `code/ng_links/` | Test NG links (example_ng_link.txt, etc.) |
| `code/lib/neuroglancer-chat/` | NeuroglancerState — used as library import only |
| `code/run_capsule.py` | v2 pipeline (reference only, being replaced) |
| `data/` | Immutable input data |
| `scratch/` → `/scratch/` | Model weights, caches |
| `results/` → `/results/` | Pipeline outputs |
| `_archive/` | Old planning docs (v2_plan, proposals, drafts) |

**Dev hardware:** 16 CPU cores, 64 GB RAM, **Tesla T4 GPU (15 GB VRAM)**
**Prod hardware:** 16 CPU cores, 64 GB RAM, **NVIDIA L40S GPU (45 GB VRAM)**

---

## Project Goal

**molmo-glancer v3** — autonomous Neuroglancer visual analysis powered by Molmo2-O-7B.

Given an NG link and a question about 3D data, the system runs a free-form agent loop: the model decides what to do (screenshot, scan, think, answer), the system executes it (Playwright + NeuroglancerState), and the model interprets the result. Iterates until confident.

---

## Architecture (v3)

```
User Question + NG Link
        │
        ▼
  Volume Metadata Discovery (zarr shape, voxel scales, layers)
        │
        ▼
  ┌─────────────────────────────────────┐
  │           AGENT LOOP                │
  │                                     │
  │  Model decides action (text-only)   │
  │    ├── screenshot → Playwright      │
  │    ├── scan → Playwright (video)    │
  │    ├── think → reasoning only       │
  │    └── answer → final synthesis     │
  │                                     │
  │  System executes, model interprets  │
  │  Context accumulates, loop repeats  │
  └─────────────────────────────────────┘
        │
        ▼
  Final Answer + all artifacts (screenshots, scan videos, findings)
```

### Key design decisions

- **Fixed 1024×1024 square viewport** — no directional bias, clean crop tiling for Molmo2
- **crossSectionScale is the zoom knob** — data fills the viewport via zoom, not viewport reshaping
- **Clean screenshots** — NG overlay hiding (showAxisLines/showScaleBar/showDefaultAnnotations=false) + CSS injection + canvas-only capture
- **Scans as video** — frame sequence fed to Molmo2 video mode (81 tokens/frame via 3×3 pooling)
- **No system role** — Molmo2 chat template only supports user/assistant. All instructions in user message.
- **Same code, two GPU configs** — 4-bit on T4 (dev), fp16 on L40S (prod). Auto-detected.

---

## Model: Molmo2-O-7B

| Aspect | Detail |
|---|---|
| Checkpoint | `/scratch/checkpoints/Molmo2-O-7B` |
| Backbone | OLMo3-7B-Instruct + SigLIP 2 vision encoder |
| Params | ~7.76B total |
| Precision (L40S) | fp16, no quantization — 14.5 GB weights |
| Precision (T4) | 4-bit NF4 — 3.6 GB weights |
| KV cache | 0.5 MB/token (full MHA, 32 KV heads) |
| Max context | 65,536 tokens |
| Image crops | 378×378, 2×2 pooling, ~169–196 tokens/crop, max_crops=8 (24 for detail) |
| Video frames | 378×378, 3×3 pooling, 81 tokens/frame, max 384 frames |
| Chat template | `<\|im_start\|>user/assistant` — **no system role** |

### NeuroglancerState (from neuroglancer-chat)

Library import only: `from neuroglancer_chat.backend.tools.neuroglancer_state import NeuroglancerState`

**Critical:** Do NOT use `sort_keys=True` on NG state JSON — NG depends on dimension order (x,y,z,t).

---

## Key Commands

```bash
# Install dependencies
bash /code/_dev_startup.sh

# Download model weights
export HF_TOKEN=hf_...
bash /code/_download_weights.sh

# Run v3 pipeline
bash /code/run_v3

# Direct invocation
PLAYWRIGHT_BROWSERS_PATH=/scratch/ms-playwright python3 -u /code/molmo_glancer.py

# neuroglancer-chat tests
cd /root/capsule/code/lib/neuroglancer-chat && python -m pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | (required for download) | HuggingFace access token |
| `PLAYWRIGHT_BROWSERS_PATH` | `/scratch/ms-playwright` | Chromium install location |
| `HF_HOME` | `/scratch/huggingface` | HF cache (keep off root overlay) |
| `NEUROGLANCER_BASE` | `https://neuroglancer-demo.appspot.com` | NG viewer base URL |

---

## Dependencies (v3)

| Package | Purpose |
|---------|---------|
| `transformers>=4.57` | Molmo2 inference |
| `bitsandbytes` | 4-bit quantization (T4 only) |
| `accelerate` | device_map="auto" |
| `molmo_utils` | Molmo2 processor helpers |
| `decord2` | Video frame decoding (Molmo2 requirement) |
| `playwright` | Headless browser screenshots |
| `neuroglancer-chat` | NeuroglancerState (in lib/) |
| `zarr` / `s3fs` | Volume metadata from zarr sources |
| `imageio` | Save scan videos as mp4 |

---

## Known Issues (see SETUP_NOTES.md)

- **torchvision/torch mismatch**: Fixed in `_dev_startup.sh`
