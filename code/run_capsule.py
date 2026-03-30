"""
molmo-glancer v2 — Neuroglancer Visual QA
==========================================
Given a Neuroglancer link and an open-ended question about 3D data,
uses Molmo2-O-7B to plan views, screenshot each via Playwright,
visually interpret them, and synthesize an answer.

Usage:
    python3 -u /code/run_capsule.py
    bash /code/run.sh              # preferred (sets env vars, logs output)

Outputs (in /results/):
    output.log        — full pipeline log (via run.sh tee)
    screenshots/      — one PNG per planned view
    findings.json     — per-view model responses
    ng_states.json    — NG state used for each screenshot
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
RESULTS_DIR = Path("/results")
SCREENSHOT_DIR = RESULTS_DIR / "screenshots"
VIEWPORT = {"width": 1280, "height": 720}
DATA_LOAD_WAIT = 12          # seconds to wait for NG async data streaming

# ── Inputs — edit these or override via config ──────────────────────────────

NG_LINK_FILE = "/root/capsule/example_ng_link.txt"
QUESTION = "How many neurons can you count in this volume?"
DATA_SHAPE = (495, 495, 215)  # XYZ voxels — TODO: read from zarr metadata


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
    # Resize to limit vision token count for memory stability
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

def build_view_urls(ng_link: str, view_specs: list[dict]):
    """Given a list of view specs from the model, generate NG URLs.

    Each view_spec is a dict with optional keys: x, y, z, layout.
    Missing x/y default to center of DATA_SHAPE. Missing layout defaults to 'xy'.
    Returns list of (url, metadata_dict).
    """
    from neuroglancer_chat.backend.tools.neuroglancer_state import NeuroglancerState

    base_state = NeuroglancerState.from_url(ng_link)

    # Default center position
    cx, cy = DATA_SHAPE[0] / 2, DATA_SHAPE[1] / 2
    cz = DATA_SHAPE[2] / 2

    # Ensure position array exists with correct length for dimensions
    num_dims = len(base_state.data.get("dimensions", {}))
    if "position" not in base_state.data or not base_state.data["position"]:
        base_state.data["position"] = [cx, cy, cz] + [0] * (num_dims - 3)
    while len(base_state.data["position"]) < num_dims:
        base_state.data["position"].append(0)

    results = []
    view_states = []
    for i, spec in enumerate(view_specs):
        view = base_state.clone()
        view.data["layout"] = spec.get("layout", "xy")
        pos = view.data["position"]
        pos[0] = spec.get("x", cx)
        pos[1] = spec.get("y", cy)
        pos[2] = spec.get("z", cz)

        url = view.to_url()
        meta = {"view": i, "x": pos[0], "y": pos[1], "z": pos[2],
                "layout": view.data["layout"]}
        results.append((url, meta))
        view_states.append({"view": i, "url": url, "state": view.data})

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
            label = f"view {meta['view']}"
            print(f"  Screenshotting {label} ({i+1}/{len(urls_with_meta)}) ...")
            page.goto(url, wait_until="domcontentloaded")
            time.sleep(DATA_LOAD_WAIT)

            png_bytes = page.screenshot()
            img = Image.open(BytesIO(png_bytes)).convert("RGB")

            png_path = SCREENSHOT_DIR / f"view_{meta['view']:03d}.png"
            img.save(png_path)
            results.append((img, meta))

        browser.close()

    print(f"  {len(results)} screenshots saved to {SCREENSHOT_DIR}")
    return results


# ── Pipeline ─────────────────────────────────────────────────────────────────

def parse_view_specs(response: str) -> list[dict]:
    """Extract a JSON list of view specs from model response.

    Expects a JSON array of objects, e.g.:
        [{"z": 10}, {"z": 50}, {"z": 100}]
    Falls back to evenly spaced Z slices if parsing fails.
    """
    # Find a JSON array in the response
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        try:
            specs = json.loads(match.group())
            if isinstance(specs, list) and len(specs) >= 2:
                # Normalize: if items are plain numbers, treat as Z positions
                if all(isinstance(s, (int, float)) for s in specs):
                    return [{"z": float(s)} for s in specs]
                if all(isinstance(s, dict) for s in specs):
                    return specs
        except (json.JSONDecodeError, ValueError):
            pass

    print(f"  WARNING: Could not parse view specs from model response, using fallback.")
    print(f"  Model said: {response}")
    # Fallback: 5 evenly spaced Z slices
    z_max = DATA_SHAPE[2] - 1
    return [{"z": round(z_max * i / 4)} for i in range(5)]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────
    print("\n[1/5] Loading Molmo2-O-7B ...")
    model, processor = load_model(CHECKPOINT)

    # ── 2. Read inputs ──────────────────────────────────────────────────
    print("\n[2/5] Reading inputs ...")
    raw_link = Path(NG_LINK_FILE).read_text().strip()
    print(f"  NG link file: {NG_LINK_FILE}")
    print(f"  Data shape: {DATA_SHAPE[0]}x{DATA_SHAPE[1]}x{DATA_SHAPE[2]} (XYZ)")
    print(f"  Question: {QUESTION}")

    # ── 3. Plan views ────────────────────────────────────────────────────
    print("\n[3/5] Planning views (text-only call) ...")
    plan_prompt = (
        f"You are analyzing a 3D dataset displayed in a Neuroglancer web viewer.\n"
        f"The data volume has shape X={DATA_SHAPE[0]}, Y={DATA_SHAPE[1]}, Z={DATA_SHAPE[2]} voxels.\n"
        f"The viewer shows a 2D cross-section at a given position.\n"
        f"You can control which slice to view by setting the position along X, Y, or Z.\n"
        f"The default view is an XY cross-section (looking down the Z axis).\n\n"
        f"Your task is to answer this question about the data:\n"
        f"  \"{QUESTION}\"\n\n"
        f"Plan a set of views (screenshots) you need to examine to answer this question.\n"
        f"Output a JSON list of view specifications. Each view is an object with:\n"
        f"  - \"z\": Z position (0 to {DATA_SHAPE[2]-1})\n"
        f"  - optionally \"x\", \"y\" to shift the XY center\n"
        f"  - optionally \"layout\": \"xy\", \"xz\", or \"yz\" for different cross-sections\n\n"
        f"Example: [{{\"z\": 10}}, {{\"z\": 50}}, {{\"z\": 100}}, {{\"z\": 150}}, {{\"z\": 200}}]\n"
        f"Choose views that will give you the information needed to answer the question.\n"
        f"Output ONLY the JSON list, nothing else."
    )
    plan_response = ask_text(model, processor, plan_prompt)
    print(f"  Model response: {plan_response}")

    view_specs = parse_view_specs(plan_response)
    print(f"  Planned {len(view_specs)} views: {view_specs}")

    # ── 4. Screenshot each view ──────────────────────────────────────────
    print("\n[4/5] Taking screenshots ...")
    urls_with_meta = build_view_urls(raw_link, view_specs)
    screenshots = take_screenshots(urls_with_meta)

    # ── 5. Interpret each screenshot ─────────────────────────────────────
    print("\n[5/5] Interpreting screenshots ...")
    findings = []
    for i, (img, meta) in enumerate(screenshots):
        print(f"\n  --- View {i+1}/{len(screenshots)}: "
              f"{meta['layout']} at ({meta['x']:.0f}, {meta['y']:.0f}, {meta['z']:.0f}) ---")
        interpret_prompt = (
            f"You are viewing a cross-section of a 3D dataset in a Neuroglancer viewer.\n"
            f"View: {meta['layout']} cross-section at position "
            f"X={meta['x']:.0f}, Y={meta['y']:.0f}, Z={meta['z']:.0f}.\n"
            f"Data shape: {DATA_SHAPE[0]}x{DATA_SHAPE[1]}x{DATA_SHAPE[2]} voxels (XYZ).\n\n"
            f"You are trying to answer this question: \"{QUESTION}\"\n\n"
            f"Describe what you see in this view that is relevant to answering the question.\n"
            f"Be specific with any counts or measurements."
        )
        response = ask_vision(model, processor, img, interpret_prompt)
        findings.append({"view": i, "meta": meta, "response": response})
        print(f"  {response}")

    # ── 6. Synthesize final answer ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Synthesizing final answer ...")
    print("=" * 60)

    findings_text = "\n".join(
        f"View {f['view']+1} ({f['meta']['layout']} at Z={f['meta']['z']:.0f}): {f['response']}"
        for f in findings
    )
    synth_prompt = (
        f"You examined {len(findings)} views of a 3D dataset "
        f"(shape {DATA_SHAPE[0]}x{DATA_SHAPE[1]}x{DATA_SHAPE[2]} voxels, XYZ).\n\n"
        f"Question: \"{QUESTION}\"\n\n"
        f"Here are your observations from each view:\n\n"
        f"{findings_text}\n\n"
        f"Based on all your observations, provide a final answer to the question.\n"
        f"Explain your reasoning."
    )
    final_answer = ask_text(model, processor, synth_prompt, max_new_tokens=1024)

    print(f"\n{final_answer}")

    # ── Save outputs ─────────────────────────────────────────────────────
    (RESULTS_DIR / "answer.txt").write_text(final_answer)
    (RESULTS_DIR / "findings.json").write_text(json.dumps(findings, indent=2))
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"  answer.txt     — final answer")
    print(f"  findings.json  — per-view findings")
    print(f"  screenshots/   — {len(screenshots)} PNG files")


if __name__ == "__main__":
    main()
