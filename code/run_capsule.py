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


def ask_text(model, processor, prompt: str, max_new_tokens: int = 512):
    """Text-only call to Molmo2 (no image). Returns (text, token_counts)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[0, input_len:]
    output_len = len(generated)
    text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, {"input_tokens": input_len, "output_tokens": output_len}


def ask_vision(model, processor, image: Image.Image, prompt: str,
               max_new_tokens: int = 512):
    """Image+text call to Molmo2. Returns (text, token_counts)."""
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
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[0, input_len:]
    output_len = len(generated)
    text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, {"input_tokens": input_len, "output_tokens": output_len}


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
    Raises ValueError if parsing fails — no silent fallback.
    """
    match = re.search(r'\[.*\]', response, re.DOTALL)

    # If no closing ']', the model likely hit the token limit mid-array.
    # Try to repair by finding '[' and truncating to the last complete item.
    if not match:
        bracket = response.find('[')
        if bracket == -1:
            raise ValueError(f"No JSON array found in model response:\n  {response}")
        truncated = response[bracket:]
        # Strip any trailing partial object/comma and close the array
        truncated = re.sub(r',\s*(\{[^}]*)?$', '', truncated)
        truncated = truncated.rstrip().rstrip(',') + ']'
        print(f"  WARNING: JSON array was truncated (likely hit token limit), repaired.")
        try:
            specs = json.loads(truncated)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Could not repair truncated JSON: {e}\n  {response}")
    else:
        try:
            specs = json.loads(match.group())
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid JSON in model response: {e}\n  {response}")

    if not isinstance(specs, list) or len(specs) == 0:
        raise ValueError(f"Expected non-empty JSON list, got: {specs}")

    # Normalize: plain numbers → z positions, anything else → must be dicts
    normalized = []
    for item in specs:
        if isinstance(item, (int, float)):
            normalized.append({"z": float(item)})
        elif isinstance(item, dict):
            normalized.append(item)
        else:
            raise ValueError(f"Unexpected item in view specs: {item}")

    # Clamp coordinates to valid data ranges
    x_max, y_max, z_max = DATA_SHAPE[0] - 1, DATA_SHAPE[1] - 1, DATA_SHAPE[2] - 1
    for spec in normalized:
        if "x" in spec:
            spec["x"] = max(0, min(float(spec["x"]), x_max))
        if "y" in spec:
            spec["y"] = max(0, min(float(spec["y"]), y_max))
        if "z" in spec:
            spec["z"] = max(0, min(float(spec["z"]), z_max))

    return normalized


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

    # Token usage tracking
    token_usage = {"steps": [], "totals": {"input_tokens": 0, "output_tokens": 0}}

    def track(step_name, tokens, call_type="text"):
        """Record token counts for a step/sub-step."""
        entry = {"step": step_name, "type": call_type, **tokens}
        token_usage["steps"].append(entry)
        token_usage["totals"]["input_tokens"] += tokens["input_tokens"]
        token_usage["totals"]["output_tokens"] += tokens["output_tokens"]
        print(f"    [{tokens['input_tokens']} in / {tokens['output_tokens']} out tokens]")

    # ── 3. Strategy — what to look for ───────────────────────────────────
    print("\n[3/9] Strategy — what visual features to look for (text-only) ...")
    strategy_prompt = (
        f"You are analyzing a 3D microscopy dataset displayed in a Neuroglancer web viewer.\n"
        f"The data volume has shape X={DATA_SHAPE[0]}, Y={DATA_SHAPE[1]}, Z={DATA_SHAPE[2]} voxels.\n"
        f"Neuroglancer shows 2D cross-sections of 3D data. You can view XY, XZ, or YZ planes\n"
        f"at any position in the volume.\n\n"
        f"You need to answer this question:\n"
        f"  \"{QUESTION}\"\n\n"
        f"What visual features, patterns, or structures should you look for in the images\n"
        f"to answer this question? Think about what would be visible in 2D cross-sections.\n"
        f"Be specific about what to look for and why.\n\n"
        f"Your response will be fed directly into the next step of an automated pipeline.\n"
        f"Write a concise, structured list of features — no preamble or filler."
    )
    strategy, tokens = ask_text(model, processor, strategy_prompt, max_new_tokens=1024)
    track("3_strategy", tokens)
    print(f"  Strategy: {strategy}")

    # ── 4. Approach — how to look for those features ────────────────────
    print("\n[4/9] Approach — how to observe those features in Neuroglancer (text-only) ...")
    approach_prompt = (
        f"You are analyzing a 3D dataset in Neuroglancer.\n"
        f"Volume shape: X={DATA_SHAPE[0]}, Y={DATA_SHAPE[1]}, Z={DATA_SHAPE[2]} voxels.\n"
        f"You can view XY, XZ, or YZ cross-sections at any position in the volume.\n\n"
        f"Question: \"{QUESTION}\"\n\n"
        f"You identified these features to look for:\n"
        f"  {strategy}\n\n"
        f"Now describe HOW to observe those features using 2D cross-sections.\n"
        f"Consider: which orientations (XY, XZ, YZ) reveal the features best?\n"
        f"Which regions of the volume (top, middle, bottom, edges, center) are most\n"
        f"informative? Should you sample densely or sparsely? Do you need multiple\n"
        f"orientations to disambiguate 3D structure from 2D slices?\n\n"
        f"Your response will be fed directly into the next step to produce a concrete\n"
        f"list of view coordinates. Write a concise, actionable viewing plan — no filler."
    )
    approach, tokens = ask_text(model, processor, approach_prompt, max_new_tokens=1024)
    track("4_approach", tokens)
    print(f"  Approach: {approach}")

    # ── 5. Plan views — concrete view list ───────────────────────────────
    print("\n[5/9] Planning views — concrete view specifications (text-only) ...")
    plan_prompt = (
        f"TASK: Output a JSON list of 4-16 views to screenshot in Neuroglancer.\n\n"
        f"CONSTRAINTS:\n"
        f"  - Volume shape: X=0..{DATA_SHAPE[0]-1}, Y=0..{DATA_SHAPE[1]-1}, Z=0..{DATA_SHAPE[2]-1}\n"
        f"  - Each view: {{\"z\": <int>}} with optional \"x\", \"y\", \"layout\" (\"xy\"|\"xz\"|\"yz\")\n"
        f"  - All coordinates must be within the volume bounds above\n"
        f"  - Output 4 to 16 views total\n\n"
        f"CONTEXT:\n"
        f"  Question: \"{QUESTION}\"\n"
        f"  Features: {strategy}\n"
        f"  Approach: {approach}\n\n"
        f"EXAMPLE OUTPUT:\n"
        f"[{{\"z\": 10}}, {{\"z\": 40, \"layout\": \"xz\"}}, {{\"z\": 85}}, {{\"z\": 130, \"x\": 200, \"y\": 300}}, {{\"z\": 190, \"layout\": \"yz\"}}]\n\n"
        f"Output ONLY the JSON list. No text before or after."
    )
    plan_response, tokens = ask_text(model, processor, plan_prompt, max_new_tokens=1024)
    track("5_plan_views", tokens)
    print(f"  Model response: {plan_response}")

    view_specs = parse_view_specs(plan_response)
    print(f"  Planned {len(view_specs)} views: {view_specs}")

    # ── 6. Per-view guidance — what to look for in each view ────────────
    print("\n[6/9] Per-view guidance — what to look for in each view (text-only) ...")
    view_guidances = []
    for i, spec in enumerate(view_specs):
        layout = spec.get("layout", "xy")
        z = spec.get("z", DATA_SHAPE[2] / 2)
        x = spec.get("x", DATA_SHAPE[0] / 2)
        y = spec.get("y", DATA_SHAPE[1] / 2)
        view_label = f"{layout} at ({x:.0f}, {y:.0f}, {z:.0f})"
        print(f"\n  --- View {i+1}/{len(view_specs)}: {view_label} ---")

        guidance_prompt = (
            f"You are about to examine a screenshot of a {layout} cross-section\n"
            f"at position X={x:.0f}, Y={y:.0f}, Z={z:.0f}\n"
            f"in a 3D microscopy dataset ({DATA_SHAPE[0]}x{DATA_SHAPE[1]}x{DATA_SHAPE[2]} voxels).\n\n"
            f"Question: \"{QUESTION}\"\n"
            f"Features to look for: {strategy}\n"
            f"Viewing approach: {approach}\n\n"
            f"Given this specific view's position and orientation, what exactly should\n"
            f"you look for in the image?\n\n"
            f"Your response will be passed directly as instructions to guide visual\n"
            f"interpretation of the screenshot. Write a short, concrete checklist — no filler."
        )
        guidance, tokens = ask_text(model, processor, guidance_prompt, max_new_tokens=512)
        track(f"6_guidance_view_{i}", tokens)
        view_guidances.append(guidance)
        print(f"  Guidance: {guidance}")

    # ── 7. Screenshot each view ──────────────────────────────────────────
    print("\n[7/9] Taking screenshots ...")
    urls_with_meta = build_view_urls(raw_link, view_specs)
    screenshots = take_screenshots(urls_with_meta)

    # ── 8. Visual interpretation — guided by per-view instructions ───────
    print("\n[8/9] Interpreting screenshots ...")
    findings = []
    for i, (img, meta) in enumerate(screenshots):
        view_label = (f"{meta['layout']} at "
                      f"({meta['x']:.0f}, {meta['y']:.0f}, {meta['z']:.0f})")
        print(f"\n  --- View {i+1}/{len(screenshots)}: {view_label} ---")

        interpret_prompt = (
            f"You are viewing a {meta['layout']} cross-section of a 3D microscopy dataset\n"
            f"in Neuroglancer at position X={meta['x']:.0f}, Y={meta['y']:.0f}, Z={meta['z']:.0f}.\n"
            f"Data shape: {DATA_SHAPE[0]}x{DATA_SHAPE[1]}x{DATA_SHAPE[2]} voxels (XYZ).\n\n"
            f"Question: \"{QUESTION}\"\n\n"
            f"What to look for in this view:\n"
            f"  {view_guidances[i]}\n\n"
            f"Describe what you see that is relevant to the question.\n"
            f"Be specific with any counts, measurements, or spatial patterns.\n\n"
            f"Your response will be collected with other views and fed to a final synthesis\n"
            f"step. Write structured observations — no filler or repetition of the instructions."
        )
        response, tokens = ask_vision(model, processor, img, interpret_prompt, max_new_tokens=1024)
        track(f"8_interpret_view_{i}", tokens, call_type="vision")
        findings.append({"view": i, "meta": meta, "guidance": view_guidances[i], "response": response})
        print(f"  Interpretation: {response}")

    # ── 9. Synthesize final answer ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("[9/9] Synthesizing final answer ...")
    print("=" * 60)

    findings_text = "\n".join(
        f"View {f['view']+1} ({f['meta']['layout']} at Z={f['meta']['z']:.0f}): {f['response']}"
        for f in findings
    )
    synth_prompt = (
        f"You examined {len(findings)} views of a 3D microscopy dataset "
        f"(shape {DATA_SHAPE[0]}x{DATA_SHAPE[1]}x{DATA_SHAPE[2]} voxels, XYZ).\n\n"
        f"Question: \"{QUESTION}\"\n\n"
        f"Your strategy was: {strategy}\n\n"
        f"Here are your observations from each view:\n\n"
        f"{findings_text}\n\n"
        f"Based on all your observations and strategy, provide a final answer to the question.\n"
        f"Explain your reasoning."
    )
    final_answer, tokens = ask_text(model, processor, synth_prompt, max_new_tokens=2048)
    track("9_synthesize", tokens)

    print(f"\n{final_answer}")

    # ── Save outputs ─────────────────────────────────────────────────────
    (RESULTS_DIR / "answer.txt").write_text(final_answer)
    (RESULTS_DIR / "findings.json").write_text(json.dumps(findings, indent=2))
    (RESULTS_DIR / "token_usage.json").write_text(json.dumps(token_usage, indent=2))
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"  answer.txt       — final answer")
    print(f"  findings.json    — per-view findings")
    print(f"  token_usage.json — per-step token counts")
    print(f"  screenshots/     — {len(screenshots)} PNG files")
    print(f"\nToken totals: {token_usage['totals']['input_tokens']} input, "
          f"{token_usage['totals']['output_tokens']} output")


if __name__ == "__main__":
    main()
