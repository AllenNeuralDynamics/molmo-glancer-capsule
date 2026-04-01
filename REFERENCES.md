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

---

*Compiled 2026-04-01 during v3 planning.*
