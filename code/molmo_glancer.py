"""
molmo-glancer v3 — Autonomous Neuroglancer Visual Analysis
==========================================================
Agent loop: model decides actions (screenshot, scan, count, reason, answer),
system executes them (Playwright + NeuroglancerState), model interprets.
Iterates until confident or max iterations reached.

Usage:
    python3 -u /code/molmo_glancer.py
    bash /code/run_v3              # preferred (sets env vars, logs output)
"""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from molmo_utils import process_vision_info

from gpu_config import load_model, get_vram_usage
from volume_info import (
    VolumeInfo, discover_volume, compute_fov,
    compute_visible_window, format_fov_feedback,
    resolve_zoom,
)
from visual_capture import (
    build_clean_state, capture_screenshot, execute_scan,
    create_browser, VIEWPORT_SIZE, save_scan_video,
    annotate_screenshot, annotate_scan_frames,
)

# ── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("/results")

# Named presets: --preset <name> selects an NG link + question pair
PRESETS = {
    "neurons": {
        "ng_link": "/root/capsule/code/ng_links/example_ng_link.txt",
        "question": "How many neurons can you count in this volume?",
    },
    "alignment": {
        "ng_link": "/root/capsule/code/ng_links/example_r2r_ng_link.txt",
        "question": (
            "How well are the neurons aligned between the fixed (green) and moving (magenta) volumes? "
            "Use layerVisibility to compare: show each layer alone, then overlay both. "
            "Look for overlap, shifts, and misregistration between the two."
        ),
    },
    "neurons_large": {
        "ng_link": "/root/capsule/code/ng_links/large_ng_link.txt",
        "question": "How many neurons can you count in this volume?",
    },
    "alignment_loop": {
        "ng_link": "/root/capsule/code/ng_links/alignment_loop.txt",
        "question": (
            "How does neuron alignment change across the iterative alignment loops? "
            "The fixed volume is green, the moving volume is magenta, and the moving "
            "segmentation layers are white/black. Some layers are initially disabled — "
            "the initial view shows the fixed reference and the final aligned moving volume. "
            "Use layerVisibility to enable other loop layers and compare alignment quality "
            "across iterations, including the Raw (unwarped) moving round."
        ),
    },
}

# Defaults — overridden by --preset or env vars
NG_LINK_FILE = os.environ.get("NG_LINK_FILE",
    "/root/capsule/code/ng_links/example_ng_link.txt")
QUESTION = os.environ.get("QUESTION",
    "How many neurons can you count in this volume?")

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


# ── Pointing / Counting ───────────────────────────────────────────────────

# Regexes from Molmo2 model card for parsing point coordinates
_COORD_REGEX = re.compile(r"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
_FRAME_REGEX = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
_POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def extract_video_points(text: str, image_w: int, image_h: int) -> list[tuple]:
    """Extract video pointing coordinates from model output.

    Returns list of (frame_id, x, y) tuples with pixel coordinates.
    """
    all_points = []
    for coord in _COORD_REGEX.finditer(text):
        for point_grp in _FRAME_REGEX.finditer(coord.group(1)):
            frame_id = float(point_grp.group(1))
            for pt in _POINTS_REGEX.finditer(point_grp.group(2)):
                idx, x, y = pt.group(1), pt.group(2), pt.group(3)
                x = float(x) / 1000 * image_w
                y = float(y) / 1000 * image_h
                if 0 <= x <= image_w and 0 <= y <= image_h:
                    all_points.append((frame_id, x, y))
    return all_points


def extract_image_points(text: str, image_w: int, image_h: int) -> list[tuple]:
    """Extract image pointing coordinates from model output.

    Returns list of (x, y) tuples with pixel coordinates.
    """
    all_points = []
    for coord in _COORD_REGEX.finditer(text):
        for point_grp in _FRAME_REGEX.finditer(coord.group(1)):
            for pt in _POINTS_REGEX.finditer(point_grp.group(2)):
                x = float(pt.group(2)) / 1000 * image_w
                y = float(pt.group(3)) / 1000 * image_h
                if 0 <= x <= image_w and 0 <= y <= image_h:
                    all_points.append((x, y))
    return all_points


def ask_vision_pointing(model, processor, image: Image.Image, prompt: str,
                        max_new_tokens: int = 2048, config: dict = None):
    """Image pointing call to Molmo2. Returns (raw_text, points, token_counts).

    Uses same pipeline as ask_vision but returns parsed points too.
    Points are list of (x, y) tuples in pixel coordinates.
    """
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
    points = extract_image_points(text, image.width, image.height)
    return text, points, {"input_tokens": input_len, "output_tokens": len(generated)}


def ask_scan_pointing(model, processor, frames: list[Image.Image], prompt: str,
                      max_new_tokens: int = 2048, config: dict = None):
    """Video pointing call to Molmo2 using process_vision_info pipeline.

    Returns (raw_text, points, token_counts).
    Points are list of (frame_id, x, y) tuples.
    """
    # Build messages with PIL frames as video content
    # process_vision_info expects the video as a list of PIL images with timestamps
    synthetic_fps = 0.5
    timestamps = [i / synthetic_fps for i in range(len(frames))]

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "video", "video": frames, "timestamps": timestamps,
         "max_fps": 2.0, "num_frames": len(frames)},
    ]}]

    # Use process_vision_info to handle video preprocessing (bypasses frame sampling)
    _, videos, video_kwargs = process_vision_info(messages)
    videos_arr, video_metadatas = zip(*videos)
    videos_arr, video_metadatas = list(videos_arr), list(video_metadatas)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        videos=videos_arr,
        video_metadata=video_metadatas,
        text=text,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[0, input_len:]
    text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Extract points using frame dimensions
    img_w = video_metadatas[0]["width"] if isinstance(video_metadatas[0], dict) else video_metadatas[0].width
    img_h = video_metadatas[0]["height"] if isinstance(video_metadatas[0], dict) else video_metadatas[0].height
    points = extract_video_points(text, img_w, img_h)

    return text, points, {"input_tokens": input_len, "output_tokens": len(generated)}


# ── Action Parsing ──────────────────────────────────────────────────────────

def parse_action(model_output: str) -> dict | None:
    """Extract a JSON action object from model text output.

    Returns parsed dict or None if no valid JSON found.
    """
    # First try: the entire output might be JSON (handles nested braces correctly)
    try:
        result = json.loads(model_output.strip())
        if isinstance(result, dict) and "action" in result:
            return result
    except json.JSONDecodeError:
        pass

    # Second try: look for ```json ... ``` blocks
    json_block = re.search(r'```json\s*(\{.*?\})\s*```', model_output, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except json.JSONDecodeError:
            pass

    # Third try: find the outermost { ... } by matching balanced braces
    start = model_output.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(model_output)):
            if model_output[i] == '{':
                depth += 1
            elif model_output[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(model_output[start:i+1])
                    except json.JSONDecodeError:
                        break
        pass

    return None


def _resolve_show(action: dict, volume_info: VolumeInfo):
    """Convert "show": [1, 2] to layerVisibility dict using layer numbers."""
    show = action.pop("show", None)
    if show is None:
        # Also check inside view dict for screenshots
        view = action.get("view", {})
        show = view.pop("show", None)
    if show is None:
        return

    # Build visibility dict: listed numbers are visible, all others hidden
    show_set = set(show) if isinstance(show, list) else {show}
    layer_vis = {}
    for i, layer in enumerate(volume_info.layers, 1):
        layer_vis[layer.name] = (i in show_set)

    # Store in the right place
    if action.get("action") == "screenshot":
        action.setdefault("view", {})["layerVisibility"] = layer_vis
    else:
        action["layerVisibility"] = layer_vis


def validate_action(action: dict, volume_info: VolumeInfo) -> dict:
    """Validate and normalize an action dict. Resolves zoom names, clamps positions."""
    action_type = action.get("action", "")

    if action_type in ("screenshot", "scan", "count"):
        # ── Resolve "show" layer numbers to layerVisibility ────────
        _resolve_show(action, volume_info)

        # ── Resolve zoom name to crossSectionScale ──────────────────
        # The model sends "zoom": "fit" etc., we translate to a float.
        # Check both the action top-level (scans) and view dict (screenshots).
        for container in [action, action.get("view", {})]:
            if "zoom" in container:
                container["crossSectionScale"] = resolve_zoom(
                    container.pop("zoom"), volume_info
                )

        view = action.get("view", {})
        if action_type in ("scan", "count"):
            if action_type == "count" and "keyframe_interval" in action:
                action["keyframe_interval"] = max(1, min(int(action["keyframe_interval"]),
                                                         action.get("frames", 50)))
            for key in ("start", "end"):
                pos = action.get(key, {})
                for i, axis in enumerate(["x", "y", "z"]):
                    if axis in pos:
                        pos[axis] = max(0, min(float(pos[axis]), volume_info.shape[i]))
        else:
            for i, axis in enumerate(["x", "y", "z"]):
                if axis in view:
                    view[axis] = max(0, min(float(view[axis]), volume_info.shape[i]))

            if "crossSectionScale" in view:
                scale = float(view["crossSectionScale"])
                if scale <= 0:
                    view["crossSectionScale"] = resolve_zoom("fit", volume_info)
                else:
                    view["crossSectionScale"] = scale

    return action


# ── Duplicate Detection & Frame Cache ──────────────────────────────────────

def _geometry_fingerprint(action: dict) -> str:
    """Geometric-only fingerprint for frame caching (ignores prompt/target)."""
    atype = action.get("action", "")
    if atype in ("scan", "count"):
        s = action.get("start", {})
        e = action.get("end", {})
        return (f"{action.get('scan_type','')}|{action.get('layout','xy')}|"
                f"{round(s.get('x',0)/5)*5},{round(s.get('y',0)/5)*5},{round(s.get('z',0)/5)*5}|"
                f"{round(e.get('x',0)/5)*5},{round(e.get('y',0)/5)*5},{round(e.get('z',0)/5)*5}|"
                f"{round(action.get('crossSectionScale',1.0), 2)}|"
                f"{action.get('frames', 50)}")
    return ""


def _action_fingerprint(action: dict) -> str:
    """Create a fingerprint string for an action to detect near-duplicates.

    Uses geometry + action type + target (for count). Does NOT include free-form
    prompt text — rephrased prompts on the same view are duplicates.
    A scan and count of the same region are distinct (different atype).
    """
    atype = action.get("action", "")
    # For screenshots, include layerVisibility so toggling layers = different view
    layer_vis = ""
    if atype == "screenshot":
        v = action.get("view", {})
        lv = v.get("layerVisibility", action.get("layerVisibility", {}))
        if lv:
            layer_vis = "|" + ",".join(f"{k}={v}" for k, v in sorted(lv.items()))
        return (f"screenshot|{v.get('layout','xy')}|"
                f"{round(v.get('x',0)/5)*5},{round(v.get('y',0)/5)*5},{round(v.get('z',0)/5)*5}|"
                f"{round(v.get('crossSectionScale',1.0), 2)}{layer_vis}")
    elif atype in ("scan", "count"):
        s = action.get("start", {})
        e = action.get("end", {})
        target = action.get("target", "") if atype == "count" else ""
        lv = action.get("layerVisibility", {})
        if lv:
            layer_vis = "|" + ",".join(f"{k}={v}" for k, v in sorted(lv.items()))
        return (f"{atype}|{action.get('scan_type','')}|{action.get('layout','xy')}|"
                f"{round(s.get('x',0)/5)*5},{round(s.get('y',0)/5)*5},{round(s.get('z',0)/5)*5}|"
                f"{round(e.get('x',0)/5)*5},{round(e.get('y',0)/5)*5},{round(e.get('z',0)/5)*5}|"
                f"{round(action.get('crossSectionScale',1.0), 2)}|{target}{layer_vis}")
    return ""


def count_prior_matches(new_action: dict, history: list[dict]) -> int:
    """Count how many times this action fingerprint appears in history."""
    atype = new_action.get("action", "")
    if atype not in ("screenshot", "scan", "count"):
        return 0

    new_fp = _action_fingerprint(new_action)
    if not new_fp:
        return 0

    return sum(1 for entry in history
               if _action_fingerprint(entry.get("action_data", {})) == new_fp)


# ── Prompt Construction ─────────────────────────────────────────────────────

def build_action_schema(volume_info: VolumeInfo, max_scan_frames: int = 50) -> str:
    """Build the action schema with volume-appropriate example coordinates."""
    from volume_info import format_zoom_table
    cx = volume_info.shape[0] / 2
    cy = volume_info.shape[1] / 2
    cz = volume_info.shape[2] / 2
    zmax = volume_info.shape[2]

    return f"""ACTIONS AVAILABLE:
You must respond with exactly one JSON object. Available actions:

1. screenshot — capture a 2D cross-section of the data
   {{"action": "screenshot",
    "view": {{"x": {cx:.0f}, "y": {cy:.0f}, "z": {cz:.0f}, "layout": "xy",
             "zoom": "full"}},
    "prompt": "<what specifically to look for in this view>"}}

2. scan — sweep through the data as a video and DESCRIBE what you see (qualitative)
   {{"action": "scan", "scan_type": "z_sweep",
    "start": {{"x": {cx:.0f}, "y": {cy:.0f}, "z": 0}},
    "end":   {{"x": {cx:.0f}, "y": {cy:.0f}, "z": {zmax:.0f}}},
    "frames": {max_scan_frames}, "layout": "xy", "zoom": "full",
    "prompt": "<what specifically to look for across this sweep>"}}
   scan_type options: z_sweep, x_pan, y_pan
   scan is for QUALITATIVE description — understanding structure, distribution, and context.
   Do NOT use scan to produce numerical counts; use count for that.

3. count — DETECT + COUNT specific objects via automated pointing on sampled keyframes
   {{"action": "count", "scan_type": "z_sweep",
    "start": {{"x": {cx:.0f}, "y": {cy:.0f}, "z": 0}},
    "end":   {{"x": {cx:.0f}, "y": {cy:.0f}, "z": {zmax:.0f}}},
    "frames": {max_scan_frames}, "layout": "xy", "zoom": "full",
    "target": "neurons", "keyframe_interval": 5}}
   The system automatically detects and marks each instance in sampled keyframes —
   you get back exact per-frame counts. Use count when you need quantitative results.
   A scan first can help you decide what to count, where, and at what zoom.
   keyframe_interval: spacing between sampled frames (2-3 for small/dense, 5-10 for large/sparse).
   "target" should be a short noun describing what to count.

4. reason — reason about findings so far (no visual input, runs a text inference call)
   {{"action": "reason",
    "question": "<what you want to reason about>"}}
   Use reason to synthesize findings, resolve contradictions, or plan next steps
   before committing to a visual action or final answer.

5. answer — final answer (ends the session)
   {{"action": "answer",
    "answer": "<your specific answer to the question>"}}

LAYOUT: "xy", "xz", "yz", "4panel"
NOTE: Do NOT use "3d" layout — it renders only a wireframe bounding box for raw image data, not the actual voxel data.

OPTIONAL KEYS (for screenshot, scan, and count):
  "show": [1, 2]  — which layers to show (by number from the Layers list above). Omit to keep current visibility.
  "shaderRange": [vmin, vmax]  — adjust brightness/contrast for image layers

IMPORTANT: Zooms below "full" CROP the view — you will miss data outside the visible area.

PROMPT: Write a specific "prompt" for each screenshot/scan describing what you want to learn.
Do NOT copy the placeholder — write a prompt specific to your current goal and question.

{format_zoom_table()}
"""


def build_decision_prompt(question: str, volume_info: VolumeInfo,
                          history: list[dict], config: dict,
                          iteration: int, forced_answer: bool = False) -> str:
    """Build the user message for the agent's next decision."""
    parts = []

    # Role and instructions
    parts.append(
        "You are a visual data analyst. You explore 3D volumetric data "
        "by taking screenshots and video scans of a Neuroglancer viewer, "
        "then synthesize an answer.\n"
    )

    if forced_answer:
        parts.append(
            "YOU MUST ANSWER NOW. This is the final iteration. "
            "Provide your best answer based on all findings so far.\n"
            "Respond with: {\"action\": \"answer\", \"answer\": \"your answer here\"}\n"
        )
    else:
        parts.append(build_action_schema(volume_info, config["max_scan_frames"]))

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
                elif atype in ("scan", "count"):
                    summary_lines.append(f"  [{atype}, {a.get('scan_type','?')}, {a.get('frames',0)} frames]")
                elif atype == "reason":
                    reasoning = a.get("reasoning", a.get("question", ""))[:100]
                    summary_lines.append(f"  [{atype}: {reasoning}...]")
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
    elif atype in ("scan", "count"):
        header = (f"  [action {iteration}: {atype}, {a.get('scan_type','?')}, "
                  f"{a.get('frames',0)} frames"
                  f"{', target=' + a.get('target','') if atype == 'count' else ''}]")
    elif atype == "reason":
        header = f"  [action {iteration}: {atype}]"
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

    # ── Transcript log — full prompts and responses, appended live ─────
    transcript_path = RESULTS_DIR / "transcript.md"
    transcript_path.write_text("# molmo-glancer v3 — Transcript\n\n")

    def log_exchange(iteration, step, prompt, response, tokens=None, note=None):
        """Append a prompt/response pair to the transcript file."""
        with open(transcript_path, "a") as f:
            f.write(f"---\n\n## Iteration {iteration} — {step}\n\n")
            if note:
                f.write(f"_{note}_\n\n")
            if tokens:
                f.write(f"**Tokens:** {tokens['input_tokens']} in / {tokens['output_tokens']} out\n\n")
            f.write(f"### Prompt\n\n```\n{prompt}\n```\n\n")
            f.write(f"### Response\n\n```\n{response}\n```\n\n")

    # ── Parse NG state and discover volume ──────────────────────────────
    print("\n[Setup] Parsing NG link and discovering volume metadata ...")
    base_state = NeuroglancerState.from_url(ng_link)
    volume_info = discover_volume(base_state.data)

    # ── Save prompt templates for inspection ─────────────────────────
    save_prompt_templates(volume_info, config, question)

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
    frame_cache = {}      # geometry_fingerprint → list of PIL frames
    screenshot_count = 0
    scan_count = 0
    consecutive_duplicates = 0
    max_consecutive_duplicates = 3  # force answer after this many in a row
    final_answer = None
    max_iter = config["max_agent_iterations"]

    print(f"\n[Agent] Starting loop (max {max_iter} iterations)")
    print(f"  Question: {question}\n")

    with sync_playwright() as pw:
        browser, page = create_browser(pw, config)

        # ── Phase 1: First Look — "What am I looking at?" ────────────
        print("[Phase 1] First Look — What am I looking at?")
        screenshot_count += 1
        cx, cy, cz = volume_info.shape[0] / 2, volume_info.shape[1] / 2, volume_info.shape[2] / 2
        first_look_state = build_clean_state(base_state, {
            "x": cx, "y": cy, "z": cz,
        }, volume_info)
        first_look_img = capture_screenshot(page, first_look_state, config, screenshot_count)

        first_look_prompt = (
            f"This is a Neuroglancer view of a 3D volume "
            f"({volume_info.shape[0]}×{volume_info.shape[1]}×{volume_info.shape[2]} voxels).\n"
            f"{volume_info.format_for_prompt()}\n\n"
            f"Describe what you see: what kind of data, what structures are visible, "
            f"how dense or sparse is the content?"
        )
        first_look_finding, tokens = ask_vision(
            model, processor, first_look_img, first_look_prompt,
            max_new_tokens=512, config=config,
        )
        track(0, "first_look", tokens)
        log_exchange(0, "first_look", first_look_prompt, first_look_finding, tokens, "image: view_001.png")
        print(f"  First look: {first_look_finding[:300]}...")

        history.append({
            "iteration": 0,
            "action_data": {"action": "screenshot", "view": {"layout": base_state.data.get("layout", "4panel")},
                            "prompt": "Phase 1: What am I looking at?"},
            "finding": first_look_finding,
            "fov_feedback": "[user's original view — default zoom and position]",
        })

        # ── Phase 2: Plan — "What views should I examine?" ──────────
        print("\n[Phase 2] Planning — What views should I examine?")
        plan_prompt = (
            f"You examined a 3D volume and described it as:\n"
            f"\"{first_look_finding}\"\n\n"
            f"Question: \"{question}\"\n"
            f"Volume: {volume_info.format_for_prompt()}\n\n"
            f"You can take screenshots (2D cross-sections), video scans (sweeps along an axis), "
            f"and counting scans (pointing to specific objects across keyframes).\n\n"
            f"What strategy should you use to answer this question? "
            f"Think about what regions to examine, what to look for, "
            f"and whether counting or scanning would be most useful. "
            f"Respond in plain text, not JSON."
        )
        plan_response, tokens = ask_text(model, processor, plan_prompt, max_new_tokens=512)
        track(0, "plan", tokens)
        log_exchange(0, "plan", plan_prompt, plan_response, tokens)
        print(f"  Plan: {plan_response[:400]}...")

        history.append({
            "iteration": 0,
            "action_data": {"action": "reason", "question": "Plan strategy"},
            "finding": plan_response,
            "fov_feedback": "",
        })

        # ── Agent loop ──────────────────────────────────────────────────
        for iteration in range(1, max_iter + 1):
            print(f"\n{'='*60}")
            print(f"  Iteration {iteration}/{max_iter}")
            vram = get_vram_usage()
            print(f"  VRAM: {vram['allocated']:.1f} / {vram['total']:.1f} GB")
            print(f"{'='*60}")

            # ── Build decision prompt ───────────────────────────────────
            forced = (iteration == max_iter) or (consecutive_duplicates >= max_consecutive_duplicates)
            if forced and consecutive_duplicates >= max_consecutive_duplicates:
                print("  [Forced] Too many repeated actions — requiring answer now.")
            prompt = build_decision_prompt(
                question, volume_info, history, config, iteration, forced_answer=forced,
            )

            # ── Ask model for next action ───────────────────────────────
            print("\n  [Decision] Asking model for next action ...")
            decision_text, tokens = ask_text(model, processor, prompt, max_new_tokens=512)
            track(iteration, "decision", tokens)
            log_exchange(iteration, "decision", prompt, decision_text, tokens)
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
                log_exchange(iteration, "decision_retry", retry_prompt, decision_text, tokens)
                action = parse_action(decision_text)

            if action is None:
                print("  ERROR: Failed to parse action after retry. Forcing reason action.")
                action = {"action": "reason", "question": f"Failed to produce valid JSON: {decision_text[:200]}"}

            action = validate_action(action, volume_info)
            action_type = action.get("action", "unknown")
            print(f"\n  [Action] {action_type}")

            # ── Duplicate check — allow up to 2, then force reason ──────
            prior_count = count_prior_matches(action, history)
            if prior_count >= 2:
                consecutive_duplicates += 1
                print(f"  BLOCKED: action done {prior_count} times already — forcing reason step")

                reason_prompt = (
                    f"The user's question is: \"{question}\"\n\n"
                    f"You just tried to repeat an action you've already done {prior_count} times. "
                    f"Step back and think: what have you learned so far?\n\n"
                    f"Your findings so far:\n"
                )
                for entry in history:
                    f = entry.get("finding", "")
                    if f and not f.startswith("["):
                        reason_prompt += f"- {f[:300]}\n"
                reason_prompt += (
                    f"\nBased on these findings, what should you do DIFFERENTLY next? "
                    f"Consider: different position, zoom, layout, layer visibility, axis, "
                    f"or a different action type entirely. "
                    f"If you have enough information, your next action should be 'answer'."
                )

                finding, tokens = ask_text(
                    model, processor, reason_prompt, max_new_tokens=512,
                )
                track(iteration, "forced_reason", tokens)
                log_exchange(iteration, "forced_reason", reason_prompt, finding, tokens)
                print(f"  Forced reasoning: {finding[:200]}...")

                history.append({
                    "iteration": iteration,
                    "action_data": {"action": "reason", "question": "[forced — repeated action blocked]"},
                    "finding": finding,
                    "fov_feedback": "",
                })
                continue
            else:
                consecutive_duplicates = 0

            # ── Execute action ──────────────────────────────────────────
            finding = ""
            fov_feedback = ""

            if action_type == "screenshot":
                view_spec = action.get("view", {})
                user_prompt = action.get("prompt", "Describe what you see in this view.")
                screenshot_count += 1
                state = build_clean_state(base_state, view_spec, volume_info)
                img = capture_screenshot(page, state, config, screenshot_count)

                interpret_prompt = (
                    f"Question: \"{question}\"\n\n"
                    f"{user_prompt}\n"
                    f"Describe what you see. Give counts or measurements where possible. "
                    f"What does this tell you about the question?"
                )

                # Model interprets the screenshot
                print(f"  [Interpret] {user_prompt[:80]}...")
                finding, tokens = ask_vision(
                    model, processor, img, interpret_prompt,
                    max_new_tokens=1024, config=config,
                )
                track(iteration, "interpret", tokens)
                log_exchange(iteration, "interpret_screenshot", interpret_prompt, finding, tokens,
                             f"image: view_{screenshot_count:03d}.png")
                print(f"  Finding: {finding[:200]}...")

                # FOV feedback
                pos = [view_spec.get("x", 0), view_spec.get("y", 0), view_spec.get("z", 0)]
                scale = view_spec.get("crossSectionScale", 1.0)
                layout = view_spec.get("layout", "xy")
                fov_feedback = format_fov_feedback(pos, scale, layout, volume_info)
                print(f"  {fov_feedback}")

            elif action_type == "scan":
                scan_count += 1
                user_prompt = action.get("prompt", "Describe what you observe in this scan.")

                geo_fp = _geometry_fingerprint(action)
                if geo_fp in frame_cache:
                    frames = frame_cache[geo_fp]
                    print(f"  [Cache hit] Reusing {len(frames)} frames from prior scan")
                    save_scan_video(frames, scan_count)
                else:
                    frames = execute_scan(base_state, action, volume_info, config, scan_count)
                    if geo_fp:
                        frame_cache[geo_fp] = frames

                # Compute inter-frame spacing for spatial context
                scan_start = action.get("start", {})
                scan_end = action.get("end", {})
                cx, cy, cz = volume_info.shape[0]/2, volume_info.shape[1]/2, volume_info.shape[2]/2
                s = np.array([scan_start.get("x", cx), scan_start.get("y", cy), scan_start.get("z", cz)])
                e = np.array([scan_end.get("x", cx), scan_end.get("y", cy), scan_end.get("z", cz)])
                total_dist = float(np.linalg.norm(e - s))
                frame_spacing = total_dist / max(len(frames) - 1, 1)
                scan_axis = action.get("scan_type", "z_sweep").replace("_sweep", "").replace("_pan", "")

                interpret_prompt = (
                    f"Question: \"{question}\"\n\n"
                    f"{user_prompt}\n"
                    f"Scan: {len(frames)} frames along {scan_axis}, "
                    f"~{frame_spacing:.1f}µm between frames, "
                    f"{total_dist:.0f}µm total.\n"
                    f"Describe what you see across the frames. "
                    f"Give counts or estimates where possible. "
                    f"What does this tell you about the question?"
                )

                # Model interprets the scan frames
                print(f"  [Interpret scan] {len(frames)} frames, {user_prompt[:80]}...")
                finding, tokens = ask_scan(
                    model, processor, frames, interpret_prompt,
                    max_new_tokens=1024, config=config,
                )
                track(iteration, "interpret_scan", tokens)
                log_exchange(iteration, "interpret_scan", interpret_prompt, finding, tokens,
                             f"video: scan_{scan_count:03d}.mp4, {len(frames)} frames")
                print(f"  Finding: {finding[:200]}...")

            elif action_type == "count":
                scan_count += 1
                target = action.get("target", "objects")

                geo_fp = _geometry_fingerprint(action)
                if geo_fp in frame_cache:
                    frames = frame_cache[geo_fp]
                    print(f"  [Cache hit] Reusing {len(frames)} frames from prior scan")
                    save_scan_video(frames, scan_count)
                else:
                    frames = execute_scan(base_state, action, volume_info, config, scan_count)
                    if geo_fp:
                        frame_cache[geo_fp] = frames

                # Step 1: Per-keyframe image pointing
                keyframe_interval = max(1, int(action.get("keyframe_interval", 5)))
                keyframe_indices = list(range(0, len(frames), keyframe_interval))
                print(f"  [Count] Pointing to '{target}' on {len(keyframe_indices)} keyframes "
                      f"(every {keyframe_interval} of {len(frames)} frames) ...")

                points = []  # (frame_idx, x, y) tuples
                total_point_tokens = {"input_tokens": 0, "output_tokens": 0}

                # Compute pixel size for pointing context
                scale = action.get("crossSectionScale",
                                   max(volume_info.shape[0], volume_info.shape[1]) / 1024)
                fov_um = scale * 1024
                neuron_pixels = int(30.0 / (fov_um / 1024))
                point_prompt = (
                    f"Point to the {target}. "
                    f"Each {target.rstrip('s')} is approximately {neuron_pixels} pixels across."
                )

                for ki in keyframe_indices:
                    _, frame_points, tokens = ask_vision_pointing(
                        model, processor, frames[ki], point_prompt,
                        max_new_tokens=2048, config=config,
                    )
                    total_point_tokens["input_tokens"] += tokens["input_tokens"]
                    total_point_tokens["output_tokens"] += tokens["output_tokens"]
                    for x, y in frame_points:
                        points.append((float(ki), x, y))
                    print(f"    keyframe {ki}: {len(frame_points)} points")

                track(iteration, "count_point", total_point_tokens)
                # Log pointing summary (individual keyframe outputs are structured coords, not prose)
                pointing_summary = "\n".join(
                    f"  keyframe {ki}: {sum(1 for p in points if int(p[0]) == ki)} points"
                    for ki in keyframe_indices
                )
                log_exchange(iteration, "count_point", point_prompt, pointing_summary,
                             total_point_tokens,
                             f"video: scan_{scan_count:03d}.mp4, {len(keyframe_indices)} keyframes")
                print(f"  Pointing total: {len(points)} points across {len(keyframe_indices)} keyframes")

                # Save annotated video with point markers
                if points:
                    annotate_scan_frames(frames, points, scan_count)

                # Step 2: Ask text model to interpret the count result
                scan_start = action.get("start", {})
                scan_end = action.get("end", {})
                cx, cy, cz = volume_info.shape[0]/2, volume_info.shape[1]/2, volume_info.shape[2]/2
                s = np.array([scan_start.get("x", cx), scan_start.get("y", cy), scan_start.get("z", cz)])
                e = np.array([scan_end.get("x", cx), scan_end.get("y", cy), scan_end.get("z", cz)])
                total_dist = float(np.linalg.norm(e - s))
                frame_spacing = total_dist / max(len(frames) - 1, 1)
                scan_axis = action.get("scan_type", "z_sweep").replace("_sweep", "").replace("_pan", "")

                # Summarize point distribution across keyframes
                frame_ids = sorted(set(int(p[0]) for p in points)) if points else []
                points_per_frame = {}
                for p in points:
                    fid = int(p[0])
                    points_per_frame[fid] = points_per_frame.get(fid, 0) + 1

                keyframe_spacing = keyframe_interval * frame_spacing
                interpret_prompt = (
                    f"The user's question is: \"{question}\"\n\n"
                    f"You pointed to {target} in {len(keyframe_indices)} keyframes sampled "
                    f"every {keyframe_interval} frames from a {scan_axis} sweep of {len(frames)} frames "
                    f"(~{frame_spacing:.1f}µm between frames, ~{keyframe_spacing:.1f}µm between keyframes, "
                    f"{total_dist:.0f}µm total).\n"
                    f"Found {len(points)} points across {len(frame_ids)} of "
                    f"{len(keyframe_indices)} sampled keyframes.\n"
                )
                if points_per_frame:
                    counts = sorted(points_per_frame.values())
                    interpret_prompt += (
                        f"Points per keyframe: min={counts[0]}, max={counts[-1]}, "
                        f"median={counts[len(counts)//2]}.\n"
                    )
                interpret_prompt += (
                    f"\nThese are automated pixel-level detections. "
                    f"The same {target} may appear in adjacent keyframes "
                    f"(keyframe spacing ~{keyframe_spacing:.1f}µm). "
                    f"Report the estimated number of unique {target} detected in this scan. "
                    f"Do NOT extrapolate to the full volume — just report what was detected."
                )

                print(f"  [Interpret count] ...")
                count_finding, tokens = ask_text(
                    model, processor, interpret_prompt, max_new_tokens=512,
                )
                track(iteration, "count_interpret", tokens)
                log_exchange(iteration, "count_interpret", interpret_prompt, count_finding, tokens)

                finding = (
                    f"DETECTED (automated pointing): {len(points)} instances of '{target}' "
                    f"across {len(frame_ids)}/{len(keyframe_indices)} keyframes "
                    f"(from {len(frames)} total frames). "
                    f"This is a grounded count — trust it over visual estimates. "
                    f"{count_finding}"
                )
                print(f"  Finding: {finding[:200]}...")

            elif action_type == "reason":
                reason_question = action.get("question", "Synthesize findings so far.")
                reason_prompt = (
                    f"The user's question is: \"{question}\"\n\n"
                    f"Your findings so far:\n"
                )
                for entry in history:
                    f = entry.get("finding", "")
                    if f and not f.startswith("["):
                        reason_prompt += f"- {f[:300]}\n"
                reason_prompt += f"\n{reason_question}"

                print(f"  [Reason] {reason_question[:80]}...")
                finding, tokens = ask_text(
                    model, processor, reason_prompt, max_new_tokens=1024,
                )
                track(iteration, "reason", tokens)
                log_exchange(iteration, "reason", reason_prompt, finding, tokens)
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
                print(f"  WARNING: Unknown action type '{action_type}'. Treating as reason.")
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
        log_exchange(max_iter, "forced_answer", synth_prompt, answer_text, tokens)
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

def save_prompt_templates(volume_info: VolumeInfo, config: dict, question: str):
    """Save all prompt templates to results/prompts.md for inspection."""
    from volume_info import format_zoom_table

    cx, cy, cz = volume_info.shape[0] / 2, volume_info.shape[1] / 2, volume_info.shape[2] / 2

    md = []
    md.append("# molmo-glancer v3 — Prompt Templates\n")
    md.append(f"Generated for question: *{question}*\n")
    md.append(f"Volume: {volume_info.format_for_prompt()}\n")

    md.append("---\n")
    md.append("## 1. First Look (image + text)\n")
    md.append("Sent with a center-position screenshot.\n")
    md.append("```")
    md.append(
        f"This is a Neuroglancer view of a 3D volume "
        f"({volume_info.shape[0]}×{volume_info.shape[1]}×{volume_info.shape[2]} voxels).\n"
        f"{volume_info.format_for_prompt()}\n\n"
        f"Describe what you see: what kind of data, what structures are visible, "
        f"how dense or sparse is the content?"
    )
    md.append("```\n")

    md.append("---\n")
    md.append("## 2. Plan (text-only)\n")
    md.append("Sent after first look, asks for strategy.\n")
    md.append("```")
    md.append(
        f'You examined a 3D volume and described it as:\n'
        f'"{{first_look_finding}}"\n\n'
        f'Question: "{question}"\n'
        f'Volume: {volume_info.format_for_prompt()}\n\n'
        f'You can take screenshots (2D cross-sections), video scans (sweeps along an axis), '
        f'and counting scans (pointing to specific objects across keyframes).\n\n'
        f'What strategy should you use to answer this question? '
        f'Think about what regions to examine, what to look for, '
        f'and whether counting or scanning would be most useful. '
        f'Respond in plain text, not JSON.'
    )
    md.append("```\n")

    md.append("---\n")
    md.append("## 3. Decision (text-only)\n")
    md.append("Sent each iteration. Includes action schema, volume info, and history.\n")
    md.append("### Action Schema\n")
    md.append("```")
    md.append(build_action_schema(volume_info, config["max_scan_frames"]))
    md.append("```\n")
    md.append("### Decision Wrapper\n")
    md.append("The action schema above is wrapped with:\n")
    md.append("```")
    md.append(
        "You are a visual data analyst. You explore 3D volumetric data "
        "by taking screenshots and video scans of a Neuroglancer viewer, "
        "then synthesize an answer.\n\n"
        "{action_schema}\n\n"
        "VOLUME INFO:\n  {volume_info}\n\n"
        "FINDINGS SO FAR (iterations 1-N):\n  {history_entries}\n\n"
        "QUESTION: {question}\n\n"
        "Iteration X/Y. What is your next action? Respond with a JSON object."
    )
    md.append("```\n")

    md.append("---\n")
    md.append("## 4. Forced Answer (text-only)\n")
    md.append("Sent when max iterations reached or too many duplicates.\n")
    md.append("```")
    md.append(
        "YOU MUST ANSWER NOW. This is the final iteration. "
        "Provide your best answer based on all findings so far.\n"
        'Respond with: {"action": "answer", "answer": "your answer here"}'
    )
    md.append("```\n")

    md.append("---\n")
    md.append("## 5. Screenshot Interpret (image + text)\n")
    md.append("Sent with the captured screenshot image.\n")
    md.append("```")
    md.append(
        'Question: "{question}"\n\n'
        '{user_prompt}\n'
        'Describe what you see. Give counts or measurements where possible. '
        'What does this tell you about the question?'
    )
    md.append("```\n")

    md.append("---\n")
    md.append("## 6. Scan Interpret (video + text)\n")
    md.append("Sent with the captured scan video frames.\n")
    md.append("```")
    md.append(
        'Question: "{question}"\n\n'
        '{user_prompt}\n'
        'Scan: {num_frames} frames along {axis}, ~{spacing}µm between frames, {total}µm total.\n'
        'Describe what you see across the frames. '
        'Give counts or estimates where possible. '
        'What does this tell you about the question?'
    )
    md.append("```\n")

    md.append("---\n")
    md.append("## 7. Count — Keyframe Pointing (image + text)\n")
    md.append("Sent once per sampled keyframe with that frame's image.\n")
    md.append("```")
    md.append("Point to the {target}.")
    md.append("```\n")

    md.append("---\n")
    md.append("## 8. Count — Interpret (text-only)\n")
    md.append("Sent after all keyframe pointing is complete.\n")
    md.append("```")
    md.append(
        'The user\'s question is: "{question}"\n\n'
        'You pointed to {target} in {num_keyframes} keyframes sampled '
        'every {interval} frames from a {axis} sweep of {num_frames} frames '
        '(~{frame_spacing}µm between frames, ~{keyframe_spacing}µm between keyframes, '
        '{total_dist}µm total).\n'
        'Found {num_points} points across {frames_with_points} of {num_keyframes} sampled keyframes.\n'
        'Points per keyframe: min={min}, max={max}, median={median}.\n\n'
        'Based on these detections and the spatial extent of the scan, '
        'what is your estimate? Consider that the same {target} may appear '
        'in adjacent keyframes (keyframe spacing ~{keyframe_spacing}µm). '
        'Give a specific count or range.'
    )
    md.append("```\n")

    md.append("---\n")
    md.append("## 9. Decision Retry (text-only)\n")
    md.append("Appended to decision prompt when JSON parsing fails.\n")
    md.append("```")
    md.append(
        'Your previous response was not valid JSON. '
        'Please respond with ONLY a JSON object like: '
        '{"action": "screenshot", "view": {"x": 100, "y": 100, "z": 100, '
        '"layout": "xy", "crossSectionScale": 1.0}, "prompt": "describe what you see"}'
    )
    md.append("```\n")

    out_path = RESULTS_DIR / "prompts.md"
    out_path.write_text("\n".join(md))
    print(f"  Prompt templates saved: {out_path}")


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
    import argparse
    parser = argparse.ArgumentParser(description="molmo-glancer v3")
    parser.add_argument("--preset", choices=list(PRESETS.keys()),
                        help="Named run preset (overrides NG_LINK_FILE and QUESTION env vars)")
    args = parser.parse_args()

    # Resolve inputs: --preset > env vars > defaults
    if args.preset:
        p = PRESETS[args.preset]
        ng_link_file = p["ng_link"]
        question = p["question"]
    else:
        ng_link_file = NG_LINK_FILE
        question = QUESTION

    print("\n" + "=" * 60)
    print("  molmo-glancer v3 — Autonomous Neuroglancer Visual Analysis")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model ...")
    model, processor, config = load_model()

    # Read inputs
    print("\n[2/3] Reading inputs ...")
    ng_link = Path(ng_link_file).read_text().strip()
    print(f"  NG link file: {ng_link_file}")
    print(f"  Question: {question}")
    if args.preset:
        print(f"  Preset: {args.preset}")

    # Run agent
    print("\n[3/3] Running agent loop ...")
    t0 = time.time()
    answer = run_agent(model, processor, config, ng_link, question)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
