# molmo-glancer v2

Autonomous Neuroglancer question answering powered by Molmo2-O-7B.

## What it does

Given an open-ended question about 3D neuroscience data displayed in [Neuroglancer](https://github.com/google/neuroglancer), the system iteratively:

1. **Plans** which views to examine (positions, zoom levels, layer visibility)
2. **Generates** Neuroglancer URLs via deterministic state manipulation
3. **Screenshots** each view using a headless browser
4. **Interprets** the screenshots with a vision-language model
5. **Decides** whether to explore more or synthesize a final answer

All steps are driven by a single model — [Molmo2-O-7B](https://huggingface.co/allenai/Molmo2-O-7B) — loaded once and used for both text reasoning and visual understanding.

## Architecture

```
User Question → Orchestration Loop (Molmo2-O-7B, text-only)
                    ↓ view specs          ↑ visual findings
            NeuroglancerState        Molmo2-O-7B (image+text)
                    ↓ NG URLs             ↑ screenshots
                Playwright (headless Chromium)
```

- **NeuroglancerState** (from [neuroglancer-chat](code/lib/neuroglancer-chat/)) constructs Neuroglancer URLs by directly manipulating the JSON state — no browser interaction needed for navigation.
- **Molmo2-O-7B** runs in 8-bit quantization (~8 GB VRAM), fitting on a Tesla T4 (15 GB).
- **Playwright** handles headless Chromium for screenshots only.

## Hardware Requirements

- GPU with ≥ 15 GB VRAM (tested on Tesla T4)
- 64 GB RAM recommended
- CUDA 13.0+

## Quick Start

```bash
# Install dependencies
bash /code/_dev_startup.sh

# Download Molmo2-O-7B weights (~14-16 GB, requires HF_TOKEN)
export HF_TOKEN=hf_...
bash /code/_download_weights.sh

# Run the pipeline
bash /code/run_v2
```

## Repository Structure

```
code/
├── molmo_glancer.py       # Main pipeline
├── run_v2                 # Shell entry point
├── _dev_startup.sh        # Dependency installation
├── _download_weights.sh   # Model weight download
└── lib/
    └── neuroglancer-chat/ # NeuroglancerState URL builder
```

## Test Data

- `example_ng_link.txt` — single-channel mFISH brain dataset
- `example_r2r_ng_link.txt` — two-channel round-to-round registration
- `thyme_r2r_ng_link.txt` — large multi-loop registration (full resolution)

## License

See [LICENSE](LICENSE).
