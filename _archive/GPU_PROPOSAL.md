# GPU Proposal: L40S Configuration for v3

> Target GPU: **NVIDIA L40S — 45 GB VRAM**, Ada Lovelace, CUDA 13.0
> Dev/test GPU: **Tesla T4 — 15 GB VRAM**, Turing, CUDA 13.0
> Same CUDA version on both — same driver stack, same code.

---

## 1. What the L40S Changes

### VRAM budget comparison

| | T4 (dev) | L40S (prod) | Ratio |
|---|---|---|---|
| Total VRAM | 15 GB | 45 GB | **3×** |
| Weights (fp16) | Won't fit | 14.5 GB | — |
| Weights (8-bit) | 7.2 GB | 7.2 GB | — |
| Weights (4-bit) | 3.6 GB | 3.6 GB | — |
| KV headroom (fp16 weights) | — | **~29.5 GB** | — |
| KV headroom (4-bit weights) | ~10.4 GB | ~40.4 GB | **3.9×** |
| Max context (fp16 weights) | — | **~59,000 tokens** | — |
| Max context (4-bit weights) | ~21,000 tokens | ~80,000 tokens | 3.8× |

The L40S runs fp16 weights with more KV headroom than the T4 has with 4-bit quantization.

### What each configuration unlocks

| Capability | T4 (4-bit) | L40S (fp16) |
|---|---|---|
| Model quality | Degraded (4-bit quantization) | **Full quality (no quantization)** |
| Image resolution | Downscaled to 512px | **Full 1024×1024** |
| Vision tokens per image | ~340–680 (2–4 crops) | **~1,521 (8 crops)** |
| Max crops per image | 4 (VRAM limited) | **8 default, up to 24 for detail** |
| Agent loop iterations | ~5–8 (tight VRAM) | **15–20+ comfortably** |
| Video scan frames | ~30–50 | **128–200** |
| KV context for synthesis | ~4,000 tokens | **~30,000+ tokens** |
| Concurrent images in context | ~3 downscaled | **10+ full resolution** |

---

## 2. Recommended L40S Configuration

### Model loading

```python
# L40S: fp16, no quantization
model = AutoModelForImageTextToText.from_pretrained(
    checkpoint, trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

No `BitsAndBytesConfig`, no `bitsandbytes` dependency on the hot path. The full 7.76B parameters load in fp16 at ~14.5 GB. Vision backbone stays in fp16 naturally (no special casting needed).

### Image processing

```python
# L40S: no downscaling — feed full-resolution screenshots
# max_crops=8 for standard views, max_crops=24 for detail views
def ask_vision(model, processor, image, prompt, max_new_tokens=1024, detail=False):
    # No image.thumbnail() — pass full resolution
    # Processor handles cropping automatically
    max_crops = 24 if detail else 8
    # ... (pass max_crops to processor if configurable, else rely on default=8)
```

### VRAM budget for a typical v3 session

A realistic agent session on L40S with fp16 weights:

| Action | Vision tokens | Text tokens | Cumulative KV | VRAM (weights + KV + overhead) |
|--------|-------------|-------------|--------------|-------------------------------|
| *Model loaded* | — | — | — | 15.5 GB |
| Scan: 50-frame z-sweep | 4,050 | 500 | 4,550 | 17.7 GB |
| Screenshot: detail view 1 | 1,521 | 400 | 6,471 | 18.7 GB |
| Screenshot: detail view 2 | 1,521 | 400 | 8,392 | 19.6 GB |
| Think: reasoning step | — | 800 | 9,192 | 20.0 GB |
| Scan: 30-frame pan | 2,430 | 400 | 12,022 | 21.4 GB |
| Screenshot: detail view 3 | 1,521 | 400 | 13,943 | 22.3 GB |
| Screenshot: detail view 4 | 1,521 | 400 | 15,864 | 23.3 GB |
| Think: synthesis prep | — | 600 | 16,464 | 23.6 GB |
| Answer: final synthesis | — | 2,000 | 18,464 | 24.5 GB |

**Total: ~24.5 GB of 45 GB — 54% utilization.** Massive headroom.

The same session could continue for another 10+ actions before approaching limits. Or the model could request a 200-frame scan (16,200 video tokens) and still have room for detailed follow-up screenshots.

### The same session on T4 (4-bit, downscaled)

| Action | Vision tokens | Text tokens | Cumulative KV | VRAM (weights + KV + overhead) |
|--------|-------------|-------------|--------------|-------------------------------|
| *Model loaded* | — | — | — | 4.6 GB |
| Scan: 50-frame z-sweep | 4,050 | 500 | 4,550 | 6.8 GB |
| Screenshot: detail (512px) | ~500 | 400 | 5,450 | 7.3 GB |
| Screenshot: detail (512px) | ~500 | 400 | 6,350 | 7.7 GB |
| Think | — | 800 | 7,150 | 8.1 GB |
| Scan: 30-frame pan | 2,430 | 400 | 9,980 | 9.5 GB |
| Screenshot: detail (512px) | ~500 | 400 | 10,880 | 9.9 GB |
| Screenshot: detail (512px) | ~500 | 400 | 11,780 | 10.4 GB |
| Think | — | 600 | 12,380 | 10.7 GB |
| Answer | — | 2,000 | 14,380 | 11.7 GB |

**Total: ~11.7 GB of 15 GB — 78% utilization.** Tight but works. The model sees lower-quality images and reasons with degraded 4-bit weights, but the code path is identical.

---

## 3. Dual-GPU Strategy: T4 Dev / L40S Prod

### Principle: same code, different config

The pipeline code is identical on both GPUs. A single configuration layer adapts to detected hardware:

```python
import torch

def detect_gpu_profile():
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected")

    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    gpu_name = torch.cuda.get_device_properties(0).name

    if vram_gb >= 40:
        return "full"     # L40S, A6000, A100, etc.
    elif vram_gb >= 20:
        return "medium"   # L4, A10G, RTX 4090
    else:
        return "compact"  # T4, RTX 3090
```

### Configuration per profile

```python
GPU_PROFILES = {
    "compact": {      # T4 (15 GB)
        "quantization": "4bit",       # BitsAndBytesConfig(load_in_4bit=True, ...)
        "torch_dtype": torch.float16,
        "max_image_side": 512,        # downscale screenshots
        "max_crops": 4,               # limit vision tokens per image
        "max_scan_frames": 50,        # shorter scans
        "max_agent_iterations": 8,    # fewer loop cycles
        "max_context_tokens": 16000,  # conservative KV budget
        "viewport": (1024, 1024),    # fixed square
    },
    "medium": {       # L4, A10G (24 GB)
        "quantization": "8bit",
        "torch_dtype": torch.float16,
        "max_image_side": None,       # no downscale
        "max_crops": 8,
        "max_scan_frames": 80,
        "max_agent_iterations": 12,
        "max_context_tokens": 28000,
        "viewport": (1024, 1024),     # fixed square
    },
    "full": {         # L40S (45 GB)
        "quantization": None,         # no quantization — pure fp16
        "torch_dtype": torch.float16,
        "max_image_side": None,       # no downscale
        "max_crops": 8,               # 8 default, 24 for detail action
        "max_scan_frames": 200,       # long scans
        "max_agent_iterations": 20,   # long sessions
        "max_context_tokens": 55000,  # near model max
        "viewport": (1024, 1024),     # fixed square
    },
}
```

### What's tested on T4 vs what's different on L40S

| Aspect | Tested on T4? | Different on L40S? |
|--------|--------------|-------------------|
| Agent loop logic | Yes | No — same loop |
| Action parsing (screenshot, scan, think, answer) | Yes | No — same parser |
| NG state construction | Yes | No — same state builder |
| Playwright capture + CSS injection + canvas screenshot | Yes | No — same browser code |
| Volume metadata discovery | Yes | No — same zarr reader |
| FOV computation / position validation | Yes | No — same math |
| Model loading | Yes (4-bit path) | Yes — fp16 path, no BnB config |
| Image preprocessing | Yes (downscaled) | Yes — full resolution |
| Scan frame count | Yes (fewer frames) | Yes — more frames |
| Vision token count per image | Yes (fewer) | Yes — more tokens, more detail |
| Agent iteration count | Yes (fewer) | Yes — longer sessions possible |
| Output quality | Degraded (4-bit + downscaled) | Better (fp16 + full res) |

**The logic is identical. Only numerical parameters change.** A bug caught on T4 is a bug fixed on L40S. The only thing T4 can't test is whether the model produces *better answers* at fp16 — that's a quality difference, not a code difference.

### Development workflow

```
1. Write/modify code on T4 (free)
2. Run pipeline on T4 — verify no crashes, actions parse correctly,
   screenshots are clean, loop terminates, answer is produced
3. Deploy to L40S — same code, "full" profile auto-detected
4. Run pipeline on L40S — verify answer quality with full precision + resolution
```

---

## 4. What the L40S Budget Unlocks for v3

### 4.1 Full-resolution screenshots with no quality loss

| | T4 | L40S |
|---|---|---|
| Screenshot captured at | 1024×1024 | 1024×1024 |
| Fed to model at | 512×512 (downscaled) | 1024×1024 (native) |
| Vision tokens | ~340–680 | ~1,521 (8 crops) |
| Fine detail (dendrites, synapses) | Lost in downscale | **Preserved** |

The model was trained on images tiled into 8 crops at 378×378. Feeding it full-resolution screenshots means it operates in its training distribution, not a degraded version.

### 4.2 Detail mode: max_crops=24

For critical views where the model needs to see fine structure, we can override to 24 crops:

```python
# ~4,225 vision tokens — 3× more detail than 8 crops
# Use sparingly: 1-3 detail views per session
```

On L40S, a single max_crops=24 image costs ~4,225 tokens → ~2.1 GB KV cache. The budget easily supports 3-4 such detail images alongside regular 8-crop images and scans.

On T4, this would be prohibitive (4,225 tokens × 0.5 MB = 2.1 GB KV on top of limited headroom).

### 4.3 Long video scans (128–200 frames)

| Scan length | Video tokens | KV cache | Fits T4 (4-bit)? | Fits L40S (fp16)? |
|-------------|-------------|----------|-------------------|-------------------|
| 30 frames | 2,430 | 1.2 GB | Yes | Yes |
| 50 frames | 4,050 | 2.0 GB | Yes | Yes |
| 100 frames | 8,100 | 4.0 GB | Tight | Yes |
| 128 frames | 10,368 | 5.1 GB | No (with other context) | Yes |
| 200 frames | 16,200 | 7.9 GB | No | Yes |
| 384 frames | 31,104 | 15.2 GB | No | Tight (~30 GB total) |

On L40S, a 200-frame scan is comfortable. This means:
- A Z-sweep through 200 slices of a large volume
- A full 360° rotation in 200 steps
- A pan across a 200-frame-wide region

The model watches a comprehensive flythrough and identifies exactly where to zoom in.

### 4.4 Deep agent sessions (15–20 iterations)

With ~55,000 tokens of context budget, the model can:

- Perform 2–3 scans (survey, follow-up, confirmation) = ~10,000–20,000 video tokens
- Take 5–8 full-resolution screenshots = ~7,500–12,000 image tokens
- Think/reason 3–5 times = ~2,000–4,000 text tokens
- Synthesize a detailed answer = ~2,000 text tokens
- **Total: ~21,500–38,000 tokens — well within budget**

For comparison, the v2 run used 23,931 total tokens across 15 views but with degraded quality. v3 on L40S uses a similar token budget but with dramatically better visual input per token.

### 4.5 Mixed modality in a single session

The L40S budget supports combining scans and screenshots freely:

```
Example deep session (1024×1024 square viewport):

[scan]   Z-sweep, 100 frames, overview          → 8,100 video tokens
[image]  Detail at z=500, xy, 8 crops           → 1,521 image tokens
[image]  Detail at z=500, xz, 8 crops           → 1,521 image tokens
[scan]   X-pan at z=500, 60 frames              → 4,860 video tokens
[image]  Detail at x=12000, 24 crops (max)      → 4,225 image tokens
[image]  Detail at x=38000, 24 crops (max)      → 4,225 image tokens
[scan]   Rotation, 30 frames, 3D overview       → 2,430 video tokens
[text]   All prompts, reasoning, synthesis       → ~5,000 text tokens

Total: ~31,882 tokens → ~15.6 GB KV cache
Total VRAM: 14.5 (weights) + 15.6 (KV) + 1 (overhead) = ~31.1 GB
Headroom remaining: ~14 GB
```

This is a rich, thorough exploration that would be impossible on T4.

---

## 5. Why Not fp32?

fp32 weights = ~29 GB. That leaves only ~15 GB for KV cache (~30,000 tokens). This is workable but offers no practical advantage over fp16 for inference:

| | fp16 | fp32 |
|---|---|---|
| Weight VRAM | 14.5 GB | 29 GB |
| KV headroom | 29.5 GB | 15 GB |
| Max context | ~59K tokens | ~30K tokens |
| Inference quality | Excellent | Negligibly better |
| Speed | Fast | ~2× slower |

fp16 is the standard for inference. fp32 is only useful for training stability. **Use fp16.**

---

## 6. Why Not bf16?

The L40S supports bf16 natively. bf16 has wider dynamic range than fp16 (same exponent range as fp32, but fewer mantissa bits):

| | fp16 | bf16 |
|---|---|---|
| Exponent bits | 5 | 8 |
| Mantissa bits | 10 | 7 |
| Range | ±65,504 | ±3.4×10³⁸ |
| Precision | Higher | Lower |
| Training stability | Can overflow | More stable |
| Inference quality | Fine | Fine |

For inference, both work. The Molmo2-O-7B weights were saved in fp32 and can be loaded in either. **fp16 is the safe default** — slightly more numerical precision, widely tested. If we encounter any overflow issues (unlikely for inference), bf16 is the fallback.

---

## 7. Dynamic VRAM Management

The agentic loop runs an unpredictable number of iterations. We should monitor and adapt:

```python
def get_vram_usage():
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    return {"allocated_gb": allocated, "reserved_gb": reserved,
            "total_gb": total, "free_gb": total - reserved}

def check_budget(profile, action_type, estimated_tokens):
    """Check if we can afford this action. Returns (ok, warning_msg)."""
    usage = get_vram_usage()
    estimated_kv = estimated_tokens * 0.5 / 1024  # GB

    if usage["free_gb"] - estimated_kv < 2.0:  # keep 2 GB safety margin
        return False, f"Insufficient VRAM: {usage['free_gb']:.1f} GB free, need {estimated_kv:.1f} GB"

    return True, None
```

### Context compression at budget limits

If the agent approaches the context limit mid-session, the system can:

1. **Summarize older findings** — replace full scan/screenshot interpretations with 1-line summaries
2. **Drop vision tokens from history** — keep text findings, discard the image/video tokens
3. **Force answer** — tell the model to synthesize with what it has

```python
def compress_context(context, target_tokens):
    """Reduce context size by summarizing old findings."""
    while estimate_tokens(context) > target_tokens:
        oldest = context["findings"][0]
        context["findings"][0] = {
            "summary": oldest["response"][:200] + "...",
            "action": oldest["action"],
            # Drop vision tokens, keep text summary
        }
```

This graceful degradation means the loop never crashes from OOM — it adapts.

---

## 8. L40S vs T4: Performance Characteristics

Beyond VRAM, the L40S is a faster GPU:

| Spec | T4 | L40S | Speedup |
|------|-----|------|---------|
| Architecture | Turing (2018) | Ada Lovelace (2022) | — |
| FP16 TFLOPS | 65 | 362 | **5.6×** |
| Memory bandwidth | 320 GB/s | 864 GB/s | **2.7×** |
| TDP | 70W | 350W | — |
| Tensor cores | Gen 2 | Gen 4 | — |

**Practical impact on inference speed:**

LLM inference is memory-bandwidth bound (loading weights from VRAM). The L40S has 2.7× the bandwidth → roughly **2–3× faster token generation**. Combined with no-quantization overhead (4-bit dequantization has a compute cost), the L40S should produce tokens noticeably faster.

For the agent loop, this means:
- Each model call returns faster → more iterations in the same wall time
- Scan generation is faster (faster model interpretation per frame)
- Total pipeline time for a 10-action session: ~5–8 min on L40S vs ~15–25 min on T4 (estimate)

---

## 9. Could We Use a Larger Model?

With 45 GB, could we fit a larger VLM?

| Model | Params | fp16 VRAM | KV headroom on L40S | Viable? |
|-------|--------|-----------|---------------------|---------|
| Molmo2-O-7B (current) | 7.76B | 14.5 GB | 29.5 GB | **Yes — optimal** |
| Molmo2-O-7B (8-bit) | 7.76B | 7.2 GB | 36.8 GB | Yes but unnecessary |
| Hypothetical 13B VLM (fp16) | ~13B | ~26 GB | ~18 GB | Tight — limited context |
| Hypothetical 13B VLM (8-bit) | ~13B | ~13 GB | ~31 GB | Yes — but is it better? |
| Hypothetical 70B VLM (4-bit) | ~70B | ~35 GB | ~9 GB | Barely — very limited context |

**Molmo2-O-7B at fp16 is the sweet spot for 45 GB.** It leaves maximal KV headroom for long agent sessions with rich visual input. A 13B model would eat into the KV budget, reducing the number of scans and screenshots per session — trading exploration depth for per-token reasoning quality. For this project, **breadth of exploration matters more than marginal reasoning improvement**, because the bottleneck has been visual input quality and view diversity, not the model's ability to reason about what it sees.

If a better/larger Molmo variant is released in the future, 8-bit quantization of a 13B model would fit and might be worth testing. But right now, Molmo2-O-7B at full fp16 is the right call.

---

## 10. Summary: Recommended Configuration

### L40S production config

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Precision | **fp16** (no quantization) | Full model quality, 14.5 GB weights |
| Image resolution | **Full (1024×1024 viewport)** | No downscaling, square, 8 crops default |
| Detail mode | **max_crops=24** (on demand) | For critical detail views |
| Max scan frames | **200** | Long survey sweeps |
| Max agent iterations | **20** | Deep exploration |
| Max context tokens | **55,000** | Near model max, ~27 GB KV budget |
| Safety margin | **2 GB** | OOM protection |

### T4 dev config

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Precision | **4-bit NF4** | Fits in 15 GB |
| Image resolution | **512px max side** | Limit vision tokens |
| Detail mode | **Disabled** | KV budget too tight |
| Max scan frames | **50** | Shorter scans |
| Max agent iterations | **8** | Fewer cycles |
| Max context tokens | **16,000** | Conservative |
| Safety margin | **1.5 GB** | Tight budget |

### Code structure

```python
profile = detect_gpu_profile()  # "compact", "medium", or "full"
config = GPU_PROFILES[profile]

model = load_model(checkpoint, config)  # quantized or fp16
# ... rest of pipeline uses config for all numeric parameters
```

**One codebase, one pipeline, two configurations. Debug on T4, deliver on L40S.**

---

*Generated 2026-04-01. Based on NVIDIA L40S specs (45 GB VRAM, Ada Lovelace), Molmo2-O-7B architecture, and v3 agent loop design.*
