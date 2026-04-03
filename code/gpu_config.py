"""
gpu_config — GPU detection, profiles, and model loading for molmo-glancer v3.

Auto-detects hardware (T4 vs L40S) and selects quantization, resolution,
and budget parameters accordingly. Same code, different config.
"""

import torch

# ── GPU Profiles ────────────────────────────────────────────────────────────

GPU_PROFILES = {
    "compact": {          # T4 (15 GB) — dev/test
        "quantization": "4bit",
        "torch_dtype": torch.float16,
        "max_image_side": 512,
        "max_crops": 4,
        "max_scan_frames": 50,
        "max_agent_iterations": 8,
        "max_context_tokens": 16000,
    },
    "full": {             # L40S (45 GB) — production
        "quantization": None,           # pure fp16
        "torch_dtype": torch.float16,
        "max_image_side": None,         # no downscale
        "max_crops": 8,                 # 24 on demand for detail
        "max_scan_frames": 50,
        "max_agent_iterations": 20,
        "max_context_tokens": 55000,
    },
}

CHECKPOINT = "/scratch/checkpoints/Molmo2-O-7B"


def detect_gpu_profile() -> str:
    """Read GPU VRAM and return profile name: 'compact' or 'full'."""
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected, defaulting to 'compact' profile (CPU).")
        return "compact"

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")

    if vram_gb >= 40:
        profile = "full"
    else:
        profile = "compact"

    print(f"Selected profile: {profile}")
    return profile


def load_model(checkpoint: str = CHECKPOINT, profile: str | None = None):
    """Load Molmo2-O-7B with profile-appropriate quantization.

    Returns (model, processor, config_dict).
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    if profile is None:
        profile = detect_gpu_profile()
    config = GPU_PROFILES[profile]

    print(f"Loading processor from {checkpoint} ...")
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True,
    )

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if config["quantization"] == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["vision_backbone"],
        )
        print(f"Loading model (4-bit NF4, vision in fp16) from {checkpoint} ...")
    else:
        model_kwargs["torch_dtype"] = config["torch_dtype"]
        print(f"Loading model (fp16) from {checkpoint} ...")

    model = AutoModelForImageTextToText.from_pretrained(checkpoint, **model_kwargs)

    # Keep vision backbone in fp16 to avoid LayerNorm/cuBLAS issues on T4
    if config["quantization"] == "4bit" and hasattr(model.model, "vision_backbone"):
        model.model.vision_backbone.to(torch.float16)
        print("  Vision backbone cast to fp16.")

    print(f"Model loaded. Profile: {profile}")
    return model, processor, config


def get_vram_usage() -> dict:
    """Return current VRAM usage in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return {
        "allocated": round(allocated, 2),
        "reserved": round(reserved, 2),
        "free": round(total - allocated, 2),
        "total": round(total, 2),
    }
