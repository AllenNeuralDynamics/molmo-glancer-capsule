# Reference Index

Sources used during v3 planning and Molmo2 capability audit. Grouped by topic.

---

## Molmo2 Model

- [allenai/Molmo2-O-7B — HuggingFace Model Card](https://huggingface.co/allenai/Molmo2-O-7B)
  Usage examples (image, video, multi-image, pointing), benchmark scores, chat template, config files.

- [Molmo2 Blog Post (Allen AI)](https://allenai.org/blog/molmo2)
  Architecture overview, video capability, benchmark comparisons, training details.

- [Molmo2 Technical Report (arXiv 2601.10611)](https://arxiv.org/abs/2601.10611)
  Full architecture: crop/tile strategy, pooling sizes, training data, long-context training, SlowFast inference.

- [Molmo2 GitHub (allenai/molmo2)](https://github.com/allenai/molmo2)
  Source code, issues, training scripts.

- [molmo-utils GitHub (allenai/molmo-utils)](https://github.com/allenai/molmo-utils)
  `process_vision_info` helper, video/image preprocessing utilities.

## Molmo2 Config Files (on HuggingFace)

- [preprocessor_config.json](https://huggingface.co/allenai/Molmo2-O-7B/raw/main/preprocessor_config.json)
  Image processor: `max_crops=8`, `pooling_size=[2,2]`, `overlap_margins=[4,4]`, `size=378x378`, `patch_size=14`.

- [video_preprocessor_config.json](https://huggingface.co/allenai/Molmo2-O-7B/raw/main/video_preprocessor_config.json)
  Video processor: `pooling_size=[3,3]`, `num_frames=384`, `max_fps=2.0`, `do_sample_frames=true`.

- [config.json](https://huggingface.co/allenai/Molmo2-O-7B/raw/main/config.json)
  Model architecture: 32 layers, 32 KV heads (full MHA), head_dim 128, max 65536 tokens, YaRN RoPE.

- [chat_template.jinja](https://huggingface.co/allenai/Molmo2-O-7B/blob/main/chat_template.jinja)
  Chat format: `<|im_start|>user/assistant`, no system role. Images/videos prepended before messages.

- [image_processing_molmo2.py](https://huggingface.co/allenai/Molmo2-O-7B/blob/main/image_processing_molmo2.py)
  Image processor source: `select_tiling()`, `preprocess()` with `max_crops` override, overlap margin masking.

- [video_processing_molmo2.py](https://huggingface.co/allenai/Molmo2-O-7B/blob/main/video_processing_molmo2.py)
  Video processor source: `load_video()` accepts non-string inputs, frame sampling logic, `do_sample_frames`.

## OLMo3 Backbone

- [OLMo3 Blog Post (Allen AI)](https://allenai.org/blog/olmo3)
  OLMo3-7B-Instruct benchmarks: IFEval 85.6, MATH 87.3, MMLU 69.1, AIME 44.3.

- [allenai/OLMo-3-7B-Instruct — HuggingFace](https://huggingface.co/allenai/Olmo-3-7B-Instruct)
  Model card, architecture details.

## Vision Encoder

- [google/siglip2-so400m-patch14-384 — HuggingFace](https://huggingface.co/google/siglip2-so400m-patch14-384)
  SigLIP 2 vision encoder: ~400M params, 27 layers, 16 heads, patch_size=14, 384x384 input.

- [SigLIP 2 — HuggingFace Blog](https://huggingface.co/blog/siglip2)
  Architecture, training, multi-resolution support.

## HuggingFace API

- [Multimodal Chat Templates — HuggingFace Docs](https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal)
  Standard format for passing images/videos via `apply_chat_template`. Numpy array video input documented here.

## LLM Inference & VRAM

- [Mastering LLM Inference Optimization — NVIDIA Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
  KV cache formula, memory-bandwidth bottleneck, quantization tradeoffs.

- [Transformer Inference Arithmetic — kipply](https://kipp.ly/p/transformer-inference-arithmetic)
  Per-token KV cache cost derivation, memory bandwidth vs compute bound analysis.

- [Model Quantization Explained — AI Competence](https://aicompetence.org/model-quantization-explained-4-bit-vs-8-bit-vs-fp16/)
  4-bit vs 8-bit vs fp16 tradeoffs, practical VRAM estimates.

## Neuroglancer

- [Neuroglancer GitHub (google/neuroglancer)](https://github.com/google/neuroglancer)
  Source code. Key files: `src/navigation_state.ts` (crossSectionScale definition), `src/sliceview/frontend.ts` (FOV calculation), `src/perspective_view/panel.ts` (projectionScale).

- [neuroglancer Python screenshot tool](https://github.com/google/neuroglancer/blob/master/python/neuroglancer/tool/screenshot.py)
  CLI screenshot tool: `--hide-axis-lines`, `--hide-default-annotations`, `selectedLayer.visible=false` trick.

- [Neuroglancer JSON schema — viewer_state.yml](https://github.com/google/neuroglancer/blob/master/docs/json_schema/viewer_state.yml)
  Full state JSON schema: `showAxisLines`, `showScaleBar`, `showDefaultAnnotations`, `crossSectionScale`, `projectionScale`, layout options.

## Playwright + Neuroglancer Screenshot Capture (v3 optimization research)

- [Neuroglancer screenshot.py — chunk statistics callback](https://github.com/google/neuroglancer/blob/master/python/neuroglancer/tool/screenshot.py)
  `visible_chunks_gpu_memory / visible_chunks_total` — the official readiness metric. Requires local NG Python server (not remote demo).

- [Neuroglancer video_tool.py — PrefetchState](https://github.com/google/neuroglancer/blob/master/python/neuroglancer/tool/video_tool.py)
  Shows how NG pre-loads chunks for upcoming states with priority ordering. Concept applies to scan frame sequencing.

- [Neuroglancer webdriver_example.py — layerChunkProgressInfo](https://github.com/google/neuroglancer/blob/master/python/examples/webdriver_example.py)
  Accesses `viewer.layerManager.getLayerByName(...).layer.renderLayers.map(x => x.layerChunkProgressInfo)` via JS eval. Potential readiness signal if `window.viewer` is accessible on remote demo.

- [Neuroglancer chunk_manager/frontend.ts](https://github.com/google/neuroglancer/blob/master/src/chunk_manager/frontend.ts)
  Chunk loading internals — how NG tracks which chunks are visible, requested, and loaded to GPU.

- [Neuroglancer state update discussion #268](https://github.com/google/neuroglancer/discussions/268)
  Community discussion on detecting when NG has finished processing a state change.

- [Enable GPU for slow Playwright tests — Michel Kraemer](https://michelkraemer.com/enable-gpu-for-slow-playwright-tests-in-headless-mode/)
  Key insight: `--use-gl=egl` flag enables hardware GPU rendering in headless Chromium (~40% speedup over SwiftShader software rendering).

- [Headless Chrome WebGL testing with Playwright — CreateIT](https://www.createit.com/blog/headless-chrome-testing-webgl-using-playwright/)
  Practical guide for WebGL canvas capture in headless mode.

- [Testing 3D applications with Playwright on GPU — Promaton](https://blog.promaton.com/testing-3d-applications-with-playwright-on-gpu-1e9cfc8b54a9)
  End-to-end testing patterns for GPU-rendered 3D apps via Playwright.

- [Playwright CDPSession API](https://playwright.dev/python/docs/api/class-cdpsession)
  `Page.captureScreenshot` via CDP bypasses Playwright abstraction for faster captures.

- [Closer to the Metal: Leaving Playwright for CDP — browser-use](https://browser-use.com/posts/playwright-to-cdp)
  Performance comparison: CDP direct calls vs Playwright wrapper overhead.

- [Playwright parallel execution docs](https://playwright.dev/docs/test-parallel)
  Official guidance on running multiple pages concurrently.

- [Playwright performance with many workers — Issue #26739](https://github.com/microsoft/playwright/issues/26739)
  Discussion of resource contention with many concurrent pages.

### Key findings (2026-04-02)

1. **Single-page sequential > multi-page parallel for NG** — adjacent Z-slices share ~90% chunk data. Hash-fragment updates on one page reuse cached chunks; separate pages/contexts each fetch independently from S3.
2. **Shared browser context** — `context.new_page()` shares HTTP cache vs `browser.new_context()` which isolates it. Critical for zarr chunk reuse across workers.
3. **`--use-gl=egl`** — enables real GPU for WebGL rendering in headless Chromium. We have T4/L40S GPUs available but weren't using them for rendering.
4. **`networkidle` is unreliable for NG** — NG keeps streaming connections alive. `domcontentloaded` + data-aware readiness check is more robust.
5. **`window.viewer` JS access** — if accessible on remote demo, enables chunk-loading-based readiness detection (`layerChunkProgressInfo`) instead of pixel polling.

---

*Compiled 2026-04-01 during v3 planning. Updated 2026-04-02 with Playwright/NG capture research.*
