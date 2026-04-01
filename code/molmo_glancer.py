"""
molmo-glancer v3 — Autonomous Neuroglancer Visual Analysis
==========================================================
Agent loop: model decides actions (screenshot, scan, think, answer),
system executes them (Playwright + NeuroglancerState), model interprets.
Iterates until confident or max iterations reached.

Usage:
    python3 -u /code/molmo_glancer.py
    bash /code/run_v3              # preferred (sets env vars, logs output)
"""

import json
import re
import time
from pathlib import Path

import torch
from PIL import Image

from gpu_config import load_model, get_vram_usage
from volume_info import (
    VolumeInfo, discover_volume, compute_fov,
    compute_visible_window, format_fov_feedback,
)
from visual_capture import (
    build_clean_state, capture_screenshot, execute_scan,
    create_browser,
)

# ── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("/results")

# Inputs — edit these or override via config
NG_LINK_FILE = "/root/capsule/code/ng_links/example_ng_link.txt"
QUESTION = "How many neurons can you count in this volume?"

# ── Model Inference ─────────────────────────────────────────────────────────

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
    text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, {"input_tokens": input_len, "output_tokens": len(generated)}


def ask_vision(model, processor, image: Image.Image, prompt: str,
               max_new_tokens: int = 512, config: dict = None):
    """Image+text call to Molmo2. Returns (text, token_counts)."""
    # Downscale if needed (T4 profile)
    if config and config.get("max_image_side"):
        max_side = config["max_image_side"]
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
    text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, {"input_tokens": input_len, "output_tokens": len(generated)}


def ask_scan(model, processor, frames: list[Image.Image], prompt: str,
             max_new_tokens: int = 1024, config: dict = None):
    """Video (frame sequence) + text call to Molmo2. Returns (text, token_counts)."""
    from transformers.video_utils import VideoMetadata

    # Molmo2's video processor requires FPS metadata for pre-decoded frames.
    # Low fps (0.5) ensures the frame sampler (max_fps=2, step=fps/max_fps)
    # keeps ALL frames. At fps>=max_fps it discards frames — bad for us since
    # each frame is a unique, deliberately-captured position in the volume.
    synthetic_fps = 0.5
    video_metadata = VideoMetadata(
        total_num_frames=len(frames),
        fps=synthetic_fps,
        duration=len(frames) / synthetic_fps,
    )

    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
        video_metadata=video_metadata,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[0, input_len:]
    text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, {"input_tokens": input_len, "output_tokens": len(generated)}


# ── Action Parsing ──────────────────────────────────────────────────────────

def parse_action(model_output: str) -> dict | None:
    """Extract a JSON action object from model text output.

    Returns parsed dict or None if no valid JSON found.
    """
    # Try to find JSON object in the output
    # First try: look for ```json ... ``` blocks
    json_block = re.search(r'```json\s*(\{.*?\})\s*```', model_output, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except json.JSONDecodeError:
            pass

    # Second try: find the outermost { ... }
    brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', model_output, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    # Third try: the entire output might be JSON
    try:
        return json.loads(model_output.strip())
    except json.JSONDecodeError:
        pass

    return None


def validate_action(action: dict, volume_info: VolumeInfo) -> dict:
    """Validate and normalize an action dict. Clamps positions, validates scales."""
    action_type = action.get("action", "")

    if action_type in ("screenshot", "scan"):
        view = action.get("view", {})
        if action_type == "scan":
            # For scans, validate start/end
            for key in ("start", "end"):
                pos = action.get(key, {})
                for i, axis in enumerate(["x", "y", "z"]):
                    if axis in pos:
                        pos[axis] = max(0, min(float(pos[axis]), volume_info.shape[i] - 1))
        else:
            # For screenshots, validate view position
            for i, axis in enumerate(["x", "y", "z"]):
                if axis in view:
                    view[axis] = max(0, min(float(view[axis]), volume_info.shape[i] - 1))

            # Validate scale
            if "crossSectionScale" in view:
                scale = float(view["crossSectionScale"])
                if scale <= 0:
                    view["crossSectionScale"] = 1.0
                else:
                    view["crossSectionScale"] = scale

    return action


# ── Duplicate Detection ─────────────────────────────────────────────────────

def is_duplicate_view(new_action: dict, history: list[dict], volume_info: VolumeInfo, threshold: float = 0.8) -> bool:
    """Check if a new screenshot/scan overlaps >threshold with a prior view."""
    if new_action.get("action") not in ("screenshot",):
        return False

    new_view = new_action.get("view", {})
    new_layout = new_view.get("layout", "xy")
    new_scale = new_view.get("crossSectionScale", 1.0)
    new_pos = [new_view.get("x", 0), new_view.get("y", 0), new_view.get("z", 0)]

    new_window = compute_visible_window(new_pos, new_scale, volume_info.canonical_factors)

    for entry in history:
        prev = entry.get("action_data", {})
        if prev.get("action") != "screenshot":
            continue
        prev_view = prev.get("view", {})
        if prev_view.get("layout", "xy") != new_layout:
            continue

        prev_scale = prev_view.get("crossSectionScale", 1.0)
        # Skip if scales differ by more than 2x
        if max(new_scale, prev_scale) / max(min(new_scale, prev_scale), 0.001) > 2.0:
            continue

        prev_pos = [prev_view.get("x", 0), prev_view.get("y", 0), prev_view.get("z", 0)]
        prev_window = compute_visible_window(prev_pos, prev_scale, volume_info.canonical_factors)

        # Compute overlap fraction
        overlap = 1.0
        for (new_lo, new_hi), (prev_lo, prev_hi) in zip(new_window, prev_window):
            extent = max(new_hi - new_lo, 0.001)
            inter = max(0, min(new_hi, prev_hi) - max(new_lo, prev_lo))
            overlap *= inter / extent

        if overlap > threshold:
            return True

    return False


# ── Prompt Construction ─────────────────────────────────────────────────────

ACTION_SCHEMA = """ACTIONS AVAILABLE:
You must respond with exactly one JSON object. Available actions:

1. screenshot — Take a single high-detail view
   {"action": "screenshot",
    "view": {"x": 25000, "y": 12000, "z": 500, "layout": "xy",
             "crossSectionScale": 0.5},
    "prompt": "What do you see in this zoomed-in XY view?"}

2. scan — Sweep through data as video (for survey/orientation)
   {"action": "scan", "scan_type": "z_sweep",
    "start": {"x": 25000, "y": 12000, "z": 0},
    "end":   {"x": 25000, "y": 12000, "z": 2000},
    "frames": 20, "layout": "xy", "crossSectionScale": 5.0,
    "prompt": "Watch this Z-sweep and identify regions of interest."}
   scan_type options: z_sweep, x_pan, y_pan, zoom_ramp

3. think — Reason about findings so far (no visual input, no cost)
   {"action": "think",
    "reasoning": "I've surveyed the full Z range and found..."}

4. answer — Provide your final answer (terminates the session)
   {"action": "answer",
    "answer": "Based on my analysis of N views..."}

LAYOUT OPTIONS: "xy" (axial), "xz" (coronal), "yz" (sagittal), "3d", "4panel"
crossSectionScale: <1 = zoom in (fewer voxels, more detail), >1 = zoom out (more voxels, less detail)
"""


def build_decision_prompt(question: str, volume_info: VolumeInfo,
                          history: list[dict], config: dict,
                          iteration: int, forced_answer: bool = False) -> str:
    """Build the user message for the agent's next decision."""
    parts = []

    # Role and instructions
    parts.append(
        "You are a neuroglancer data analyst. You explore 3D microscopy data "
        "by taking screenshots and video scans, then synthesize an answer.\n"
    )

    if forced_answer:
        parts.append(
            "YOU MUST ANSWER NOW. This is the final iteration. "
            "Provide your best answer based on all findings so far.\n"
            "Respond with: {\"action\": \"answer\", \"answer\": \"your answer here\"}\n"
        )
    else:
        parts.append(ACTION_SCHEMA)

    # Volume info
    parts.append(f"\nVOLUME INFO:\n  {volume_info.format_for_prompt()}\n")

    # History
    if history:
        max_recent = 10
        if len(history) > max_recent:
            older = history[:-max_recent]
            recent = history[-max_recent:]
            parts.append(f"\nCOVERAGE SUMMARY (iterations 1-{len(older)}):")
            summary_lines = []
            for entry in older:
                a = entry.get("action_data", {})
                atype = a.get("action", "?")
                if atype == "screenshot":
                    v = a.get("view", {})
                    summary_lines.append(
                        f"  [{atype}, {v.get('layout','xy')}, "
                        f"pos=({v.get('x',0):.0f},{v.get('y',0):.0f},{v.get('z',0):.0f}), "
                        f"scale={v.get('crossSectionScale',1.0)}]"
                    )
                elif atype == "scan":
                    summary_lines.append(f"  [{atype}, {a.get('scan_type','?')}, {a.get('frames',0)} frames]")
                elif atype == "think":
                    reasoning = a.get("reasoning", "")[:100]
                    summary_lines.append(f"  [think: {reasoning}...]")
            parts.append("\n".join(summary_lines))

            parts.append(f"\nRECENT FINDINGS (iterations {len(older)+1}-{len(history)}):")
            for entry in recent:
                parts.append(format_history_entry(entry))
        else:
            parts.append(f"\nFINDINGS SO FAR (iterations 1-{len(history)}):")
            for entry in history:
                parts.append(format_history_entry(entry))

    parts.append(f"\nQUESTION: {question}")
    parts.append(f"\nIteration {iteration}/{config['max_agent_iterations']}. What is your next action? Respond with a JSON object.")

    return "\n".join(parts)


def format_history_entry(entry: dict) -> str:
    """Format a single history entry for the prompt."""
    a = entry.get("action_data", {})
    finding = entry.get("finding", "")
    fov = entry.get("fov_feedback", "")
    iteration = entry.get("iteration", "?")
    atype = a.get("action", "?")

    if atype == "screenshot":
        v = a.get("view", {})
        header = (f"  [action {iteration}: screenshot, {v.get('layout','xy')}, "
                  f"pos=({v.get('x',0):.0f},{v.get('y',0):.0f},{v.get('z',0):.0f}), "
                  f"scale={v.get('crossSectionScale',1.0)}]")
    elif atype == "scan":
        header = (f"  [action {iteration}: scan, {a.get('scan_type','?')}, "
                  f"{a.get('frames',0)} frames]")
    elif atype == "think":
        header = f"  [action {iteration}: think]"
    else:
        header = f"  [action {iteration}: {atype}]"

    lines = [header]
    if finding:
        lines.append(f"  [finding {iteration}: \"{finding[:300]}\"]")
    if fov:
        lines.append(f"  {fov}")
    return "\n".join(lines)


# ── Agent Loop ──────────────────────────────────────────────────────────────

def run_agent(model, processor, config: dict, ng_link: str, question: str):
    """Run the v3 agent loop. Returns final answer string."""
    from neuroglancer_chat.backend.tools.neuroglancer_state import NeuroglancerState
    from playwright.sync_api import sync_playwright

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Parse NG state and discover volume ──────────────────────────────
    print("\n[Setup] Parsing NG link and discovering volume metadata ...")
    base_state = NeuroglancerState.from_url(ng_link)
    volume_info = discover_volume(base_state.data)

    # ── Token tracking ──────────────────────────────────────────────────
    token_usage = {"iterations": [], "totals": {"input_tokens": 0, "output_tokens": 0}}

    def track(iteration, step, tokens):
        entry = {"iteration": iteration, "step": step, **tokens}
        token_usage["iterations"].append(entry)
        token_usage["totals"]["input_tokens"] += tokens["input_tokens"]
        token_usage["totals"]["output_tokens"] += tokens["output_tokens"]
        print(f"    [{tokens['input_tokens']} in / {tokens['output_tokens']} out tokens]")

    # ── Agent loop state ────────────────────────────────────────────────
    history = []          # list of {iteration, action_data, finding, fov_feedback}
    screenshot_count = 0
    scan_count = 0
    final_answer = None
    max_iter = config["max_agent_iterations"]

    print(f"\n[Agent] Starting loop (max {max_iter} iterations)")
    print(f"  Question: {question}\n")

    with sync_playwright() as pw:
        browser, page = create_browser(pw)

        for iteration in range(1, max_iter + 1):
            print(f"\n{'='*60}")
            print(f"  Iteration {iteration}/{max_iter}")
            vram = get_vram_usage()
            print(f"  VRAM: {vram['allocated']:.1f} / {vram['total']:.1f} GB")
            print(f"{'='*60}")

            # ── Build decision prompt ───────────────────────────────────
            forced = (iteration == max_iter)
            prompt = build_decision_prompt(
                question, volume_info, history, config, iteration, forced_answer=forced,
            )

            # ── Ask model for next action ───────────────────────────────
            print("\n  [Decision] Asking model for next action ...")
            decision_text, tokens = ask_text(model, processor, prompt, max_new_tokens=512)
            track(iteration, "decision", tokens)
            print(f"  Model output: {decision_text[:200]}...")

            # ── Parse action ────────────────────────────────────────────
            action = parse_action(decision_text)
            if action is None:
                # Retry once with format reminder
                print("  WARNING: Could not parse action JSON. Retrying with format reminder ...")
                retry_prompt = (
                    prompt + "\n\nYour previous response was not valid JSON. "
                    "Please respond with ONLY a JSON object like: "
                    '{"action": "screenshot", "view": {"x": 100, "y": 100, "z": 100, "layout": "xy", "crossSectionScale": 1.0}, "prompt": "describe what you see"}'
                )
                decision_text, tokens = ask_text(model, processor, retry_prompt, max_new_tokens=512)
                track(iteration, "decision_retry", tokens)
                action = parse_action(decision_text)

            if action is None:
                print("  ERROR: Failed to parse action after retry. Forcing think action.")
                action = {"action": "think", "reasoning": f"Failed to produce valid JSON: {decision_text[:200]}"}

            action = validate_action(action, volume_info)
            action_type = action.get("action", "unknown")
            print(f"\n  [Action] {action_type}")

            # ── Execute action ──────────────────────────────────────────
            finding = ""
            fov_feedback = ""

            if action_type == "screenshot":
                view_spec = action.get("view", {})
                interpret_prompt = action.get("prompt", "Describe what you see in this view.")

                # Check for duplicate
                if is_duplicate_view(action, history, volume_info):
                    print("  SKIPPED: duplicate view (>80% overlap with prior view)")
                    finding = "[skipped — duplicate view]"
                else:
                    screenshot_count += 1
                    state = build_clean_state(base_state, view_spec, volume_info)
                    img = capture_screenshot(page, state, config, screenshot_count)

                    # Model interprets the screenshot
                    print(f"  [Interpret] {interpret_prompt[:80]}...")
                    finding, tokens = ask_vision(
                        model, processor, img, interpret_prompt,
                        max_new_tokens=1024, config=config,
                    )
                    track(iteration, "interpret", tokens)
                    print(f"  Finding: {finding[:200]}...")

                    # FOV feedback
                    pos = [view_spec.get("x", 0), view_spec.get("y", 0), view_spec.get("z", 0)]
                    scale = view_spec.get("crossSectionScale", 1.0)
                    layout = view_spec.get("layout", "xy")
                    fov_feedback = format_fov_feedback(pos, scale, layout, volume_info)
                    print(f"  {fov_feedback}")

            elif action_type == "scan":
                scan_count += 1
                interpret_prompt = action.get("prompt", "Describe what you observe in this scan.")

                frames = execute_scan(page, base_state, action, volume_info, config, scan_count)

                # Model interprets the scan frames
                print(f"  [Interpret scan] {len(frames)} frames, {interpret_prompt[:80]}...")
                finding, tokens = ask_scan(
                    model, processor, frames, interpret_prompt,
                    max_new_tokens=1024, config=config,
                )
                track(iteration, "interpret_scan", tokens)
                print(f"  Finding: {finding[:200]}...")

            elif action_type == "think":
                finding = action.get("reasoning", "")
                print(f"  Reasoning: {finding[:200]}...")

            elif action_type == "answer":
                final_answer = action.get("answer", "")
                print(f"\n  [ANSWER] {final_answer}")

                # Save and break
                history.append({
                    "iteration": iteration,
                    "action_data": action,
                    "finding": final_answer,
                    "fov_feedback": "",
                })
                break

            else:
                print(f"  WARNING: Unknown action type '{action_type}'. Treating as think.")
                finding = f"Unknown action: {action_type}"

            # ── Append to history ───────────────────────────────────────
            history.append({
                "iteration": iteration,
                "action_data": action,
                "finding": finding,
                "fov_feedback": fov_feedback,
            })

        browser.close()

    # ── If loop ended without answer, force one ─────────────────────────
    if final_answer is None:
        print("\n  [Forced Answer] Max iterations reached, synthesizing from findings ...")
        synth_prompt = build_decision_prompt(
            question, volume_info, history, config,
            iteration=max_iter, forced_answer=True,
        )
        answer_text, tokens = ask_text(model, processor, synth_prompt, max_new_tokens=2048)
        track(max_iter, "forced_answer", tokens)
        forced_action = parse_action(answer_text)
        if forced_action and "answer" in forced_action:
            final_answer = forced_action["answer"]
        else:
            final_answer = answer_text
        print(f"\n  [ANSWER] {final_answer}")

    # ── Save outputs ────────────────────────────────────────────────────
    save_outputs(final_answer, history, token_usage)

    return final_answer


# ── Output Saving ───────────────────────────────────────────────────────────

def save_outputs(answer: str, history: list[dict], token_usage: dict):
    """Save all pipeline outputs to results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (RESULTS_DIR / "answer.txt").write_text(answer)
    (RESULTS_DIR / "findings.json").write_text(json.dumps(history, indent=2, default=str))
    (RESULTS_DIR / "token_usage.json").write_text(json.dumps(token_usage, indent=2))

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"  answer.txt       — final answer")
    print(f"  findings.json    — per-iteration findings ({len(history)} iterations)")
    print(f"  token_usage.json — token counts")
    print(f"\nToken totals: {token_usage['totals']['input_tokens']} input, "
          f"{token_usage['totals']['output_tokens']} output")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  molmo-glancer v3 — Autonomous Neuroglancer Visual Analysis")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model ...")
    model, processor, config = load_model()

    # Read inputs
    print("\n[2/3] Reading inputs ...")
    ng_link = Path(NG_LINK_FILE).read_text().strip()
    print(f"  NG link file: {NG_LINK_FILE}")
    print(f"  Question: {QUESTION}")

    # Run agent
    print("\n[3/3] Running agent loop ...")
    t0 = time.time()
    answer = run_agent(model, processor, config, ng_link, QUESTION)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
