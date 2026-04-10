# molmo-glancer

Autonomous Neuroglancer visual question answering powered by [Molmo2-O-7B](https://huggingface.co/allenai/Molmo2-O-7B).

## What it does

Given an open-ended question about 3D neuroscience data in [Neuroglancer](https://github.com/google/neuroglancer), an agent loop iteratively:

1. **Discovers** volume metadata (shape, layers, voxel scales) from the NG state and zarr sources
2. **Takes a first look** at the center of the volume and describes what it sees
3. **Plans** a strategy to answer the question
4. **Executes actions** chosen by the model — screenshots, video scans, object counting, or reasoning
5. **Interprets** each visual result, then decides the next step based on findings
6. **Answers** when confident or when the iteration budget is reached

All steps are driven by a single resident model — Molmo2-O-7B (7.76B parameters) — loaded once and used for text reasoning, visual interpretation, video understanding, and object pointing/counting.

## Examples:

See full output, transcripts, and artifacts from running the 'neurons' and 'alignment' presets here: 

https://open.quiltdata.com/b/aind-scratch-data/tree/molmo-glancer/example_data/ 

## Architecture

```
User Question + NG Link
        │
        ▼
Volume Metadata Discovery (zarr .zattrs → shape, layers, scales)
        │
        ▼
Agent Loop (iterative, max 8-20 iterations)
  ┌─────────────────────────────────────────────┐
  │  Decision (text-only)                       │
  │    → model picks: screenshot | scan |       │
  │      count | reason | answer                │
  │                                             │
  │  Execution                                  │
  │    → NeuroglancerState builds URL           │
  │    → Playwright captures canvas             │
  │                                             │
  │  Interpretation (image/video + text)        │
  │    → model describes findings               │
  │    → appended to history for next iteration │
  └─────────────────────────────────────────────┘
        │
        ▼
Final Answer + Artifacts (screenshots, videos, transcript, token usage)
```

### Components

| Component | Role |
|-----------|------|
| **Molmo2-O-7B** | Single VLM for all reasoning, vision, video, and pointing tasks |
| **NeuroglancerState** | Deterministic URL builder (inlined from neuroglancer-chat) |
| **Playwright** | Headless Chromium for screenshot and scan frame capture |
| **zarr / s3fs** | Read volume shape and scale metadata from S3 zarr sources |

### Agent actions

| Action | Description |
|--------|-------------|
| `screenshot` | Capture a 2D cross-section at a specified position, layout, and zoom |
| `scan` | Video sweep along an axis — qualitative description of structure |
| `count` | Automated object detection via per-keyframe pointing — quantitative counts |
| `reason` | Text-only synthesis of findings so far |
| `answer` | Final answer (ends the session) |

## Hardware profiles

The capsule auto-detects GPU hardware and selects the appropriate profile:

| | Compact (T4, 15 GB) | Full (L40S+, 45 GB) |
|---|---|---|
| **Quantization** | 4-bit NF4 (~3.6 GB weights) | fp16 (~14.5 GB weights) |
| **Image resolution** | 512px max side | Full resolution |
| **Max scan frames** | 50 | 50 |
| **Max iterations** | 8 | 20 |
| **Max context** | 16K tokens | 55K tokens |

## Quick start

```bash
# Run with a preset (dependencies are baked into the Docker image)
bash /code/run.sh --preset neurons

# Run all presets sequentially with stashing
bash /code/run.sh --preset all
```

### Presets

| Preset | NG link | Question |
|--------|---------|----------|
| `neurons` | Single-channel mFISH brain | "How many neurons can you count?" |
| `alignment` | Two-channel round-to-round registration | "How well are the neurons aligned between fixed and moving?" |
| `neurons_large` | Large-volume mFISH | "How many neurons can you count?" |
| `alignment_loop` | Multi-loop iterative alignment | "How does alignment change across loops?" |

### Custom inputs

```bash
# Via environment variables
export NG_LINK_FILE=/path/to/ng_link.txt
export QUESTION="Describe the structures visible in this volume."
bash /code/run.sh

# Via direct invocation
python3 -u /code/molmo_glancer.py --preset neurons
```

## Repository structure

```
code/
├── molmo_glancer.py       # Agent loop + CLI entry point
├── inference.py           # Molmo2 + OLMo model inference wrappers
├── prompts.py             # Prompt construction for all agent steps
├── actions.py             # Action JSON parsing, validation, dedup
├── gpu_config.py          # GPU validation, config, model loading
├── visual_capture.py      # Playwright screenshots, scans, point annotation
├── volume_info.py         # Volume metadata discovery, FOV computation
├── neuroglancer_state.py  # NeuroglancerState URL builder (inlined)
├── run.sh                 # Shell entry point (sets env, tees log)
├── presets/               # Run presets (JSON: ng_link + question)
│   ├── neurons.json
│   ├── alignment.json
│   ├── neurons_large.json
│   ├── alignment_loop.json
│   └── segmentation.json
└── ng_links/              # Neuroglancer state URLs
    ├── example_ng_link.txt
    ├── example_r2r_ng_link.txt
    ├── large_ng_link.txt
    └── alignment_loop.txt

environment/
├── Dockerfile             # Base: pytorch 2.4 + CUDA 12.4 + Python 3.12
└── postInstall            # torchvision upgrade + Playwright browser install
```

## Output artifacts

All results are saved to `/results/`:

| File | Contents |
|------|----------|
| `answer.txt` | Final answer text |
| `findings.json` | Per-iteration action data and findings |
| `token_usage.json` | Token counts by step and totals |
| `transcript.md` | Full prompts and responses for inspection |
| `prompts.md` | Prompt templates used (for design review) |
| `screenshots/` | `view_001.png`, `view_002.png`, ... (+ `_annotated` variants) |
| `scans/` | `scan_001.mp4` (+ `_annotated` variants with point markers) |
| `output.log` | Full pipeline log (when run via `run.sh`) |

## Project progression

This capsule was built during a hackathon and went through three major iterations:

1. **MolmoWeb agent** — First attempt used MolmoWeb (the interactive web agent bundled with Molmo2) to directly browse Neuroglancer, clicking through the UI and taking screenshots autonomously. This proved impractical within the hackathon's time constraint, and controlling Neuroglancer's complex UI via click-based automation was unreliable.

2. **Fixed pipeline** — Replaced the web agent with direct Molmo2 inference + Playwright screenshots. A rigid 9-step pipeline: plan all views up front, screenshot all, interpret all, synthesize. This worked but was inflexible — the model couldn't react to what it saw. In practice it produced near-identical views (only Z varied) with hallucinated findings, because the plan was committed before seeing any data.

3. **Reactive agent loop** (current) — Replaced the fixed pipeline with a free-form agent loop. The model sees visual data and decides what to do next based on findings. Added video scans (z-sweep, pan), automated object counting via Molmo2's pointing capability, named zoom levels, duplicate detection, frame caching, and dual GPU profile support (4-bit on T4, fp16 on L40S). This is the architecture described in this README.

## License

[MIT](LICENSE) - Allen Institute for Neural Dynamics
