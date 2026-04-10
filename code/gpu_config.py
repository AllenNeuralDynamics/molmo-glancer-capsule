"""
gpu_config — GPU validation, config, model loading, and swap management.

Targets L40S (45 GB VRAM) exclusively. No T4/compact fallback.

ModelManager handles symmetric swapping: both Molmo2 and OLMo are loaded
from disk each time they're needed and deleted when done. Simpler than
asymmetric (CPU-parking) at the cost of ~30s extra per Molmo load.
"""

import gc
import time

import torch

# ── Checkpoints ──────────────────────────────────────────────────────────────

MOLMO_CHECKPOINT = "/scratch/checkpoints/Molmo2-O-7B"
OLMO_CHECKPOINT = "/scratch/checkpoints/Olmo-3.1-32B-Think"

# ── Config ───────────────────────────────────────────────────────────────────

CONFIG = {
    # Molmo2-O-7B (vision)
    "torch_dtype": torch.float16,
    "max_scan_frames": 50,
    "max_context_tokens": 55000,
    # OLMo 3.1 32B Think (text reasoning)
    "max_olmo_context_tokens": 32000,
    # OLMo generation budgets — moderate debug (model needs room to think + answer)
    "olmo_max_new_tokens_plan": 1536,
    "olmo_max_new_tokens_decision": 1024,
    "olmo_max_new_tokens_vision_instr": 768,
    "olmo_max_new_tokens_reasoning": 2048,
    "olmo_max_new_tokens_synthesis": 2048,
    "olmo_max_new_tokens_retry": 768,
    "olmo_max_new_tokens_hard_cap": 4096,
    # OLMo sampling — per HF model card: temp=0.6, top_p=0.95
    "olmo_sampling_structured": {
        "temperature": 0.6, "top_p": 0.95,
        "repetition_penalty": 1.1, "do_sample": True,
    },
    "olmo_sampling_synthesis": {
        "temperature": 0.6, "top_p": 0.95,
        "repetition_penalty": 1.1, "do_sample": True,
    },
    "olmo_sampling_retry": {
        "temperature": 0.3, "top_p": 0.95,
        "repetition_penalty": 1.1, "do_sample": True,
    },
    # Agent loop
    "max_agent_iterations": 3,  # debug — restore to 20 for prod
}

MIN_VRAM_GB = 40


# ── GPU validation ──────────────────────────────────────────────────────────

def assert_gpu():
    """Verify CUDA GPU with >=40 GB VRAM is available, or abort."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. molmo-glancer requires an L40S (45 GB).")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")

    if vram_gb < MIN_VRAM_GB:
        raise RuntimeError(
            f"Insufficient VRAM: {vram_gb:.1f} GB (need >={MIN_VRAM_GB} GB). "
            f"molmo-glancer requires an L40S or equivalent."
        )


# ── VRAM monitoring ─────────────────────────────────────────────────────────

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


def _log_vram(label: str):
    """Print VRAM usage with a label."""
    v = get_vram_usage()
    print(f"  VRAM [{label}]: {v['allocated']:.1f} GB allocated, "
          f"{v['free']:.1f} GB free / {v['total']:.1f} GB total")


def _clear_gpu():
    """Delete all GPU tensors and reclaim VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ── ModelManager ────────────────────────────────────────────────────────────

class ModelManager:
    """Symmetric swap manager for Molmo2 and OLMo.

    Only one model on GPU at a time. Both are loaded from /scratch SSD
    and deleted when swapped out. No CPU parking — simpler at the cost
    of reload time (~30s Molmo, ~30-60s OLMo).

    Usage::

        mgr = ModelManager()
        mgr.swap_to_molmo()    # loads Molmo2 fp16
        # ... use mgr.molmo_model, mgr.molmo_processor ...
        mgr.swap_to_olmo()     # deletes Molmo, loads OLMo INT8
        # ... use mgr.olmo_model, mgr.olmo_tokenizer ...
        mgr.swap_to_molmo()    # deletes OLMo, loads Molmo2 again
    """

    def __init__(self):
        assert_gpu()
        self.active: str | None = None   # "molmo" | "olmo" | None

        # Molmo2 state
        self.molmo_model = None
        self.molmo_processor = None

        # OLMo state
        self.olmo_model = None
        self.olmo_tokenizer = None

    def swap_to_molmo(self):
        """Delete any active model, load Molmo2-O-7B fp16 from disk."""
        from transformers import AutoProcessor, AutoModelForImageTextToText

        if self.active == "molmo":
            return  # already loaded

        t0 = time.time()
        print(f"\n[ModelManager] swap_to_molmo ...")

        self._unload_all()

        self.molmo_processor = AutoProcessor.from_pretrained(
            MOLMO_CHECKPOINT, trust_remote_code=True,
        )
        self.molmo_model = AutoModelForImageTextToText.from_pretrained(
            MOLMO_CHECKPOINT,
            trust_remote_code=True,
            torch_dtype=CONFIG["torch_dtype"],
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.active = "molmo"

        elapsed = time.time() - t0
        _log_vram("molmo loaded")
        print(f"[ModelManager] Molmo2 ready ({elapsed:.1f}s)")

    def swap_to_olmo(self):
        """Delete any active model, load OLMo 3.1 32B Think INT8 from disk."""
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
        )

        if self.active == "olmo":
            return  # already loaded

        t0 = time.time()
        print(f"\n[ModelManager] swap_to_olmo ...")

        self._unload_all()

        self.olmo_tokenizer = AutoTokenizer.from_pretrained(
            OLMO_CHECKPOINT, trust_remote_code=True,
        )
        self.olmo_model = AutoModelForCausalLM.from_pretrained(
            OLMO_CHECKPOINT,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        self.active = "olmo"

        elapsed = time.time() - t0
        _log_vram("olmo loaded")
        print(f"[ModelManager] OLMo ready ({elapsed:.1f}s)")

    def _unload_all(self):
        """Delete whichever model is currently loaded and reclaim VRAM."""
        if self.active == "molmo":
            del self.molmo_model
            del self.molmo_processor
            self.molmo_model = None
            self.molmo_processor = None
        elif self.active == "olmo":
            del self.olmo_model
            del self.olmo_tokenizer
            self.olmo_model = None
            self.olmo_tokenizer = None

        self.active = None
        _clear_gpu()
