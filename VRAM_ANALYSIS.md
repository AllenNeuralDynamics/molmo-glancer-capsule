# Molmo2-O-7B VRAM Analysis for molmo-glancer v2

> **Hardware baseline:** Tesla T4 — 15 GB VRAM, CUDA 13.0
> **Current config:** 4-bit NF4 quantization, images downscaled to 512px max side

---

## 1. Model Architecture Breakdown

| Component | Parameters | Notes |
|-----------|-----------|-------|
| OLMo3-7B-Instruct (LLM backbone) | ~7.3B | 32 layers, 32 attention heads (full MHA), head dim 128 |
| SigLIP 2 vision encoder | ~413M | So400m-patch14-384, 27 layers, 16 heads |
| Adapter / image projector | ~96M | SwiGLU projector + multi-head attention pooling |
| **Total** | **~7.76B** | |

**Critical detail:** OLMo3-7B uses **full multi-head attention** (32 KV heads = 32 Q heads), *not* grouped-query attention (GQA). This means KV cache is **4x larger** than comparable models like Llama 3 (which uses 8 KV heads). This is the dominant factor constraining multi-turn / multi-image workloads.

---

## 2. Weight VRAM at Each Precision

| Precision | Bytes/param | Weight VRAM | Fits T4? | Headroom for KV cache |
|-----------|-------------|-------------|----------|-----------------------|
| fp32 | 4 | **~29 GB** | No | — |
| fp16 / bf16 | 2 | **~14.5 GB** | Barely | ~0.5 GB — unusable |
| 8-bit (INT8) | 1 | **~7.2 GB** | Yes | ~6.8 GB |
| 4-bit (NF4) ← current | 0.5 | **~3.6 GB** | Yes | ~10.4 GB |

Add ~1 GB for CUDA context, framework overhead, and activations on top of these figures.

### What GPU for full precision?

| Target | Minimum GPU | Examples |
|--------|-------------|---------|
| fp16, single image | 24 GB | RTX 3090, RTX 4090, A5000, L4 |
| fp16, multi-image (10 views) | 48 GB | A6000, A40 |
| fp32, single image | 48 GB | A6000, A40 |
| fp32, multi-image (10 views) | 80 GB | A100-80GB, H100 |

---

## 3. Vision Token Budget — How Images Consume VRAM

Molmo2 uses a **crop/tile** strategy for images:

1. A **global low-res crop** at 378×378 is always produced (1 crop)
2. The image is then tiled into up to K **overlapping high-res crops**, each 378×378
3. After attention pooling (2×2 for images), each crop produces **169 tokens**
4. Default `max_crops=8` (training) or `max_crops=24` (inference override)

| Image config | Crops | Vision tokens (approx) |
|-------------|-------|----------------------|
| Tiny image (1 crop only) | 1 | ~169 |
| Default, max_crops=8 | 1 global + up to 8 | **~1,521** |
| Override, max_crops=24 | 1 global + up to 24 | **~4,225** |

### Current pipeline constraint

The current code downscales screenshots from 1280×720 to **512×512 max** (`max_side=512` in `ask_vision()`). At 512px, the processor likely fits the image into 2–4 crops, producing roughly **340–680 vision tokens** per image — well below the model's full capability.

### Full 1280×720 screenshots (no downscale)

A 1280×720 image with `max_crops=8` would tile into approximately 6–8 crops, producing **~1,000–1,520 vision tokens**. This is what the model was trained for and would yield significantly better visual understanding — at the cost of ~2–4x more vision tokens per image.

---

## 4. KV Cache — The Hidden VRAM Consumer

### Per-token KV cache cost

```
KV cache per token = 2 × num_layers × num_kv_heads × head_dim × precision_bytes
                   = 2 × 32 × 32 × 128 × 2 (fp16)
                   = 524,288 bytes
                   ≈ 0.5 MB per token
```

This is **4x larger** than Llama-3-7B (which has 8 KV heads), because OLMo3 uses full MHA. KV cache grows **linearly** with sequence length.

### KV cache at various context lengths (fp16)

| Context length | KV cache VRAM |
|---------------|--------------|
| 1,000 tokens | 0.49 GB |
| 2,000 tokens | 0.98 GB |
| 4,000 tokens | 1.95 GB |
| 8,192 tokens | 4.00 GB |
| 16,384 tokens | 8.00 GB |
| 32,768 tokens | 16.00 GB |
| 65,536 tokens (model max) | 32.00 GB |

### Does using KV chat history affect VRAM?

**Yes, substantially.** If the pipeline accumulates context across turns (e.g., keeping prior screenshots + interpretations in a conversation), KV cache grows with every turn.

#### Current pipeline: independent calls (no shared history)

Each `ask_text()` / `ask_vision()` call in `run_capsule.py` constructs messages from scratch — there is **no KV cache reuse** between steps. Each call's KV cache is allocated, used, and freed independently.

**Peak VRAM per call (current, 4-bit model):**

| Step | Est. tokens | KV cache | Total (4-bit weights + KV + overhead) |
|------|------------|----------|--------------------------------------|
| Text planning | ~500–1,500 | 0.25–0.73 GB | ~4.8–5.3 GB |
| Vision (512px image) | ~800–1,200 | 0.39–0.59 GB | ~5.0–5.2 GB |
| Final synthesis (all findings) | ~2,000–4,000 | 0.98–1.95 GB | ~5.6–6.6 GB |

Comfortably within T4's 15 GB.

#### Hypothetical: multi-turn chat with KV history

If you instead kept a running conversation (each view's screenshot + interpretation appended to history), the context would accumulate:

| Scenario (8-crop images, max_crops=8) | Cumulative tokens | KV cache | Total w/ 4-bit | Total w/ 8-bit | Total w/ fp16 |
|---------------------------------------|-------------------|----------|----------------|----------------|---------------|
| 1 image + interpretation | ~2,000 | 0.98 GB | 5.6 GB | 9.2 GB | 16.5 GB |
| 5 images + interpretations | ~9,000 | 4.39 GB | 9.0 GB | 12.6 GB | 19.9 GB |
| 10 images + interpretations | ~17,000 | 8.30 GB | 12.9 GB | 16.5 GB | 23.8 GB |
| 15 images + interpretations | ~25,500 | 12.45 GB | 17.1 GB | 20.7 GB | 27.9 GB |

**Key takeaway:** With KV chat history and full-resolution images, 10 views on a T4 at 4-bit quantization uses ~12.9 GB — it fits, but barely. At 8-bit or fp16, it does not.

---

## 5. Video Mode — Stitched Screenshots as Video

Molmo2-O-7B **natively supports video input**. Video frames are processed differently from images:

| Aspect | Image mode | Video mode |
|--------|-----------|-----------|
| Pooling | 2×2 → 169 tokens/crop | 3×3 → **81 tokens/frame** |
| Multi-crop tiling | Yes (up to 8–24 crops) | **No** (1 crop per frame) |
| Tokens per input | ~1,521 (8 crops) | **81 per frame** |
| Sampling | N/A | Up to 2 fps, max 384 frames |

### Token comparison: N separate images vs. N video frames

| N views | As separate images (8 crops each) | As video frames | Token savings |
|---------|----------------------------------|-----------------|---------------|
| 4 | ~6,084 | ~324 | **18.8x fewer** |
| 8 | ~12,168 | ~648 | **18.8x fewer** |
| 16 | ~24,336 | ~1,296 | **18.8x fewer** |
| 10 | ~15,210 | ~810 | **18.8x fewer** |

### VRAM for video processing

| Scenario | Tokens (text + video) | KV cache | Total w/ 4-bit | Total w/ fp16 |
|----------|-----------------------|----------|----------------|---------------|
| 10 frames + prompts | ~1,500 | 0.73 GB | 5.3 GB | 16.2 GB |
| 50 frames + prompts | ~5,000 | 2.44 GB | 7.0 GB | 17.9 GB |
| 128 frames + prompts | ~11,000 | 5.37 GB | 10.0 GB | 20.9 GB |
| 384 frames + prompts | ~32,000 | 15.63 GB | 20.2 GB | 31.1 GB |

### Trade-off: image quality vs. token efficiency

Video mode is **~19x more token-efficient** but each frame gets only **81 tokens of visual representation** (vs. ~1,521 for a properly-tiled image). This means:

- **Fine details** (small neurons, thin dendrites, subtle boundaries) may be lost in video mode
- Video mode is best for **spatial orientation, coarse structure, navigation planning**
- For **detailed visual interpretation**, individual images with full cropping are superior
- A **hybrid approach** — video for survey/orientation, then full images for detailed analysis — could be optimal

### No cross-frame compression

Each video frame is processed independently through the vision encoder. There is **no temporal compression or token sharing** between frames. The efficiency gain comes purely from using 3×3 pooling (vs 2×2) and single-crop (vs multi-crop).

---

## 6. GPU Upgrade Scenarios

### What each GPU tier unlocks

| GPU | VRAM | Precision | Max context (KV cache) | Multi-image capacity |
|-----|------|-----------|----------------------|---------------------|
| **T4** (current) | 15 GB | 4-bit only | ~21K tokens | ~12 images (downscaled) |
| **L4** | 24 GB | 8-bit comfortable | ~34K tokens | ~15 full-res images |
| **RTX 4090** | 24 GB | fp16 tight, 8-bit comfortable | ~34K tokens (8-bit) | ~15 full-res images |
| **A10G** | 24 GB | fp16 tight, 8-bit comfortable | ~34K tokens (8-bit) | ~15 full-res images |
| **A6000** | 48 GB | fp16 comfortable | ~67K tokens (fp16) | 30+ full-res images |
| **A100-40GB** | 40 GB | fp16 comfortable | ~51K tokens (fp16) | 25+ full-res images |
| **A100-80GB** | 80 GB | fp32 possible | ~93K tokens (fp16) | 50+ full-res images |
| **H100** | 80 GB | fp32 possible, fastest | ~93K tokens (fp16) | 50+ full-res images |

### Recommended upgrade for this project

**For full-resolution images without quantization:**
- **Minimum: 24 GB GPU (L4, RTX 4090, A10G)** — runs fp16 with headroom for ~5 full-res images per context
- **Recommended: 48 GB GPU (A6000, A40)** — runs fp16 comfortably with 20+ images or 128-frame video
- **Ideal: 80 GB GPU (A100-80GB, H100)** — fp16 with full 65K context, or fp32 for maximum quality

---

## 7. Summary & Recommendations

### Current bottlenecks on T4

1. **Image downscaling** (1280×720 → 512px) loses fine detail — driven by VRAM limits
2. **4-bit quantization** trades model quality for memory — noticeable on reasoning tasks
3. **No KV reuse** between pipeline steps — each call starts fresh (actually saves VRAM)
4. **Full MHA** (32 KV heads) makes KV cache 4x more expensive than GQA models

### Quick wins (no hardware change)

| Change | Impact |
|--------|--------|
| Use video mode for survey views, full images for detail views | ~10x token reduction for survey phase |
| Increase `max_side` to 720px (from 512) | Better detail, ~2x more vision tokens, still fits T4 |
| Process views in batches (free KV between batches) | Already doing this — no change needed |

### With a 24 GB GPU

| Change | Impact |
|--------|--------|
| Switch to 8-bit quantization | Better model quality, ~7.2 GB weights |
| Full 1280×720 screenshots (no downscale) | Full visual detail, ~1,500 tokens/image |
| KV chat history for 5–8 images | Better context-aware interpretation |

### With a 48+ GB GPU

| Change | Impact |
|--------|--------|
| fp16 inference (no quantization) | Best model quality |
| Full resolution + full crop budget (max_crops=24) | Maximum visual detail |
| Full KV chat history across all views | Coherent multi-view reasoning |
| 128-frame video for volumetric flythrough | Comprehensive spatial survey |

---

*Generated 2026-04-01. Based on Molmo2-O-7B architecture (config.json), Molmo2 tech report, and HuggingFace model card.*
