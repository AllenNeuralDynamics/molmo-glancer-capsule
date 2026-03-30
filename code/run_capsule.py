"""
molmo-glancer v2 — Neuron Counting MVP
======================================
Given a Neuroglancer link to a 3D fluorescence microscopy dataset,
uses Molmo2-O-7B to plan Z-depth views, screenshot each via Playwright,
visually interpret them, and synthesize a neuron count estimate.

Usage:
    python3 -u /code/run_capsule.py
    bash /code/run.sh              # preferred (sets env vars, logs output)

Outputs (in /results/):
    output.log        — full pipeline log (via run.sh tee)
    screenshots/      — one PNG per Z-slice
    findings.json     — per-slice model responses
    answer.txt        — final synthesized answer
"""

import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────

CHECKPOINT = "/scratch/checkpoints/Molmo2-O-7B"
NG_LINK_FILE = "/root/capsule/example_ng_link.txt"
RESULTS_DIR = Path("/results")
SCREENSHOT_DIR = RESULTS_DIR / "screenshots"
VIEWPORT = {"width": 1280, "height": 720}
DATA_LOAD_WAIT = 12          # seconds to wait for NG async data streaming
DATA_SHAPE = (495, 495, 215) # hardcoded XYZ shape for this dataset
FALLBACK_Z_POSITIONS = [20, 60, 100, 140, 180]


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(checkpoint: str):
    """Load Molmo2-O-7B with 4-bit NF4 quantization. Returns (model, processor)."""
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    print(f"Loading processor from {checkpoint} ...")
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True,
    )
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["vision_backbone"],
    )
    print(f"Loading model (4-bit NF4, vision in fp16) from {checkpoint} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, trust_remote_code=True,
        quantization_config=quant_config, device_map="auto",
    )
    # Keep vision backbone in fp16 to avoid LayerNorm/cuBLAS issues
    if hasattr(model.model, "vision_backbone"):
        model.model.vision_backbone.to(torch.float16)
        print("  Vision backbone cast to fp16.")
    print("Model loaded.")
    return model, processor


def ask_text(model, processor, prompt: str, max_new_tokens: int = 512) -> str:
    """Text-only call to Molmo2 (no image). Returns generated text."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def ask_vision(model, processor, image: Image.Image, prompt: str,
               max_new_tokens: int = 512) -> str:
    """Image+text call to Molmo2. Returns generated text."""
    # Resize to avoid cuBLAS OOM/dimension errors in 8-bit vision backbone
    max_side = 512
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── NeuroglancerState URL generation ─────────────────────────────────────────

def build_view_urls(ng_link: str, z_positions: list[float]):
    """For each Z position, clone the NG state, set layout='xy' and the Z
    coordinate, and return a list of (url, metadata_dict)."""
    from neuroglancer_chat.backend.tools.neuroglancer_state import NeuroglancerState

    base_state = NeuroglancerState.from_url(ng_link)

    # Ensure a position exists (the example link has none — default to center)
    # Position array must match the number of dimensions in the state (x,y,z,t = 4)
    if "position" not in base_state.data or not base_state.data["position"]:
        cx, cy = DATA_SHAPE[0] / 2, DATA_SHAPE[1] / 2
        base_state.data["position"] = [cx, cy, 0, 0]
    elif len(base_state.data["position"]) == 3:
        base_state.data["position"].append(0)  # add t=0

    base_state.data["layout"] = "xy"
    base_pos = base_state.data["position"]

    results = []
    view_states = []
    for z in z_positions:
        view = base_state.clone()
        view.data["position"][2] = z
        url = view.to_url()
        meta = {"x": base_pos[0], "y": base_pos[1], "z": z}
        results.append((url, meta))
        view_states.append({"z": z, "url": url, "state": view.data})

    # Dump all NG states for debugging
    states_path = RESULTS_DIR / "ng_states.json"
    states_path.write_text(json.dumps(view_states, indent=2))
    print(f"  NG states saved to {states_path}")

    return results


# ── Playwright screenshots ───────────────────────────────────────────────────

def take_screenshots(urls_with_meta: list[tuple[str, dict]]) -> list[tuple[Image.Image, dict]]:
    """Launch Playwright, navigate to each URL, wait for data, screenshot.
    Saves PNGs to SCREENSHOT_DIR. Returns list of (PIL.Image, metadata)."""
    from playwright.sync_api import sync_playwright

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()

        for i, (url, meta) in enumerate(urls_with_meta):
            print(f"  Screenshotting Z={meta['z']} ({i+1}/{len(urls_with_meta)}) ...")
            page.goto(url, wait_until="domcontentloaded")
            time.sleep(DATA_LOAD_WAIT)

            png_bytes = page.screenshot()
            img = Image.open(BytesIO(png_bytes)).convert("RGB")

            png_path = SCREENSHOT_DIR / f"z_{meta['z']:.0f}.png"
            img.save(png_path)
            results.append((img, meta))

        browser.close()

    print(f"  {len(results)} screenshots saved to {SCREENSHOT_DIR}")
    return results


# ── Pipeline ─────────────────────────────────────────────────────────────────

def parse_z_positions(response: str) -> list[float]:
    """Extract a JSON list of Z positions from model response, with fallback."""
    # Try to find a JSON array anywhere in the response
    match = re.search(r'\[[\d\s,\.]+\]', response)
    if match:
        try:
            positions = json.loads(match.group())
            if isinstance(positions, list) and len(positions) >= 2:
                return [float(z) for z in positions]
        except (json.JSONDecodeError, ValueError):
            pass
    print(f"  WARNING: Could not parse Z positions from model response, using fallback.")
    print(f"  Model said: {response}")
    return FALLBACK_Z_POSITIONS


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────
    print("\n[1/5] Loading Molmo2-O-7B ...")
    model, processor = load_model(CHECKPOINT)

    # ── 2. Parse NG link ─────────────────────────────────────────────────
    print("\n[2/5] Reading Neuroglancer link ...")
    raw_link = Path(NG_LINK_FILE).read_text().strip()
    print(f"  Link file: {NG_LINK_FILE}")
    print(f"  Data shape: {DATA_SHAPE[0]}x{DATA_SHAPE[1]}x{DATA_SHAPE[2]} (XYZ)")

    # ── 3. Plan Z positions ──────────────────────────────────────────────
    print("\n[3/5] Planning Z positions (text-only call) ...")
    plan_prompt = (
        f"You are analyzing a 3D fluorescence microscopy brain tissue dataset.\n"
        f"The data volume has shape X={DATA_SHAPE[0]}, Y={DATA_SHAPE[1]}, Z={DATA_SHAPE[2]} voxels.\n"
        f"The web viewer shows one 2D XY slice at a time.\n"
        f"To count neurons, you need to examine multiple Z-depth slices "
        f"spanning the full Z range (0 to {DATA_SHAPE[2]-1}).\n\n"
        f"Output a JSON list of Z positions to screenshot, e.g. [10, 50, 100, 150, 200].\n"
        f"Choose positions that evenly span the Z range to get a representative sample.\n"
        f"Output ONLY the JSON list, nothing else."
    )
    plan_response = ask_text(model, processor, plan_prompt)
    print(f"  Model response: {plan_response}")

    z_positions = parse_z_positions(plan_response)
    print(f"  Z positions to screenshot: {z_positions}")

    # ── 4. Screenshot each Z position ────────────────────────────────────
    print("\n[4/5] Taking screenshots ...")
    urls_with_meta = build_view_urls(raw_link, z_positions)
    screenshots = take_screenshots(urls_with_meta)

    # ── 5. Interpret each screenshot ─────────────────────────────────────
    print("\n[5/5] Interpreting screenshots ...")
    findings = []
    for i, (img, meta) in enumerate(screenshots):
        print(f"\n  --- Slice {i+1}/{len(screenshots)}: Z={meta['z']} ---")
        interpret_prompt = (
            f"You are viewing a fluorescence microscopy cross-section of brain tissue.\n"
            f"Position: X={meta['x']:.1f}, Y={meta['y']:.1f}, Z={meta['z']:.0f}.\n"
            f"Bright spots are individual neuron cell bodies.\n\n"
            f"Count the number of visible neurons in this image. Report:\n"
            f"1. The count of distinct bright spots (neurons) you can identify\n"
            f"2. A brief description of their distribution "
            f"(clustered, scattered, dense, sparse)\n\n"
            f"Be specific with the count — give a number, not a range."
        )
        response = ask_vision(model, processor, img, interpret_prompt)
        findings.append({"z": meta["z"], "response": response})
        print(f"  {response}")

    # ── 6. Synthesize final answer ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Synthesizing final answer ...")
    print("=" * 60)

    findings_text = "\n".join(
        f"Z={f['z']:.0f}: {f['response']}" for f in findings
    )
    synth_prompt = (
        f"You examined {len(findings)} Z-slices of a 3D fluorescence microscopy "
        f"brain volume (shape {DATA_SHAPE[0]}x{DATA_SHAPE[1]}x{DATA_SHAPE[2]}).\n"
        f"Here are your per-slice neuron count findings:\n\n"
        f"{findings_text}\n\n"
        f"Based on these observations, estimate the total number of unique neurons "
        f"in the volume. Account for the fact that some neurons span multiple "
        f"Z-slices (typical neuron diameter is ~10-20 voxels in Z).\n"
        f"Give a final count estimate and explain your reasoning."
    )
    final_answer = ask_text(model, processor, synth_prompt, max_new_tokens=1024)

    print(f"\n{final_answer}")

    # ── Save outputs ─────────────────────────────────────────────────────
    (RESULTS_DIR / "answer.txt").write_text(final_answer)
    (RESULTS_DIR / "findings.json").write_text(json.dumps(findings, indent=2))
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"  answer.txt     — final neuron count estimate")
    print(f"  findings.json  — per-slice findings")
    print(f"  screenshots/   — {len(screenshots)} PNG files")


if __name__ == "__main__":
    main()
