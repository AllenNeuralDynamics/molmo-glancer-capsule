"""
molmo-glancer — Autonomous Neuroglancer Visual Analysis
=======================================================
Agent loop: model decides actions (screenshot, scan, count, reason, answer),
system executes them (Playwright + NeuroglancerState), model interprets.
Iterates until confident or max iterations reached.

Usage:
    python3 -u /code/molmo_glancer.py
    bash /code/run.sh              # preferred (sets env vars, logs output)
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

from gpu_config import ModelManager, get_vram_usage, CONFIG
from volume_info import (
    VolumeInfo, discover_volume, format_fov_feedback,
    resolve_zoom, pixel_to_physical, summarize_spatial_distribution,
)
from visual_capture import (
    build_clean_state, capture_screenshot, execute_scan,
    create_browser, save_scan_video, annotate_scan_frames,
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
    "segmentation": {
        "ng_link": "/root/capsule/code/ng_links/segmentation.txt",
        "question": (
            "Evaluate the quality of the GFP segmentation overlay. "
            "Compare the segmentation layer against the raw image channel — "
            "do the segmented regions accurately capture the fluorescent structures? "
            "Look for over-segmentation, under-segmentation, and boundary accuracy."
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


# ── OLMo Text Generation ───────────────────────────────────────────────────

OLMO_SYSTEM_PROMPT = """\
You are the reasoning engine of an autonomous visual analysis system.
You analyze 3D volumetric microscopy data loaded in Neuroglancer.

Your partner is a vision model (Molmo2) that captures and interprets
screenshots and video scans of the data. You cannot see images directly.
You plan what views to capture, and Molmo2 reports back what it sees.

RESPONSE FORMAT: Start your response DIRECTLY with the requested output.
Do NOT begin with "Okay", "Let me think", or internal reasoning.
If you use <think> tags, keep thinking under 200 words.
When JSON is requested, output ONLY the JSON object — nothing else."""


def strip_think_tokens(text: str) -> tuple[str, str, bool]:
    """Extract and separate think blocks from OLMo output.

    Handles three cases:
      1. Matched <think>...</think> blocks
      2. Bare </think> without opening tag (model omits <think>)
      3. Unclosed <think> (model ran out of tokens mid-thought)

    Parameters
    ----------
    text : str
        Raw OLMo output potentially containing think blocks.

    Returns
    -------
    clean_text : str
        Output with all think blocks removed.
    think_content : str
        Concatenated content of all think blocks.
    was_truncated : bool
        True if a think block was never closed.
    """
    open_count = text.count("<think>")
    close_count = text.count("</think>")

    # Case 2: bare </think> without <think> — everything before it is thinking
    if close_count > 0 and open_count == 0:
        idx = text.index("</think>")
        think_content = text[:idx].strip()
        clean = text[idx + len("</think>"):].strip()
        return clean, think_content, False

    was_truncated = open_count > close_count

    # Case 1: matched <think>...</think> blocks
    think_blocks = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Case 3: unclosed trailing <think> block
    if was_truncated:
        clean = re.sub(r'<think>(?!.*</think>).*$', '', clean, flags=re.DOTALL)

    return clean.strip(), '\n'.join(think_blocks), was_truncated


def ask_text_olmo(manager, system_prompt: str, user_prompt: str,
                  max_new_tokens: int = 4096, sampling: dict | None = None):
    """OLMo text generation via ChatML system+user roles.

    Handles think token stripping and truncation detection.
    Returns (text, token_counts) matching ask_text() shape.

    Parameters
    ----------
    manager : ModelManager
        Must have OLMo loaded (manager.active == "olmo").
    system_prompt : str
        System role content (OLMo supports native system role).
    user_prompt : str
        User role content.
    max_new_tokens : int
        Max tokens to generate (includes think + response).
    sampling : dict or None
        Sampling parameters (temperature, top_p, etc.). If None, uses
        CONFIG["olmo_sampling_structured"].
    """
    model = manager.olmo_model
    tokenizer = manager.olmo_tokenizer

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
    ).to(model.device)
    input_len = input_ids.shape[1]

    gen_kwargs = dict(sampling or CONFIG["olmo_sampling_structured"])
    gen_kwargs["max_new_tokens"] = max_new_tokens

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **gen_kwargs)

    generated = output_ids[0, input_len:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    clean_text, think_content, was_truncated = strip_think_tokens(raw_text)

    if was_truncated:
        print(f"  WARNING: OLMo think block truncated at {max_new_tokens} tokens")

    token_counts = {
        "input_tokens": input_len,
        "output_tokens": len(generated),
        "think_tokens": sum(len(tokenizer.encode(b)) for b in think_content.split('\n')) if think_content else 0,
    }

    return clean_text, token_counts, raw_text


def ask_vision(model, processor, image: Image.Image, prompt: str,
               max_new_tokens: int = 512):
    """Image+text call to Molmo2. Returns (text, token_counts)."""
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
                        max_new_tokens: int = 2048):
    """Image pointing call to Molmo2. Returns (raw_text, points, token_counts).

    Uses same pipeline as ask_vision but returns parsed points too.
    Points are list of (x, y) tuples in pixel coordinates.
    """
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


# ── v4 Structured Findings ─────────────────────────────────────────────────

def format_structured_findings(history: list[dict]) -> str:
    """Format history entries as structured findings for OLMo prompts."""
    lines = []
    for entry in history:
        iteration = entry.get("iteration", "?")
        a = entry.get("action_data", {})
        finding = entry.get("finding", "")
        fov = entry.get("fov_feedback", "")
        purpose = a.get("purpose", "")
        atype = a.get("action", "?")

        if iteration == 0:
            lines.append(f"── Iteration 0 (first look) {'─' * 40}")
        else:
            lines.append(f"── Iteration {iteration} {'─' * 45}")

        if atype == "screenshot":
            v = a.get("view", {})
            zoom = v.get("zoom", "full")
            lines.append(f"  View: screenshot, {v.get('layout','xy')}, "
                         f"pos=({v.get('x',0):.0f},{v.get('y',0):.0f},{v.get('z',0):.0f}), "
                         f"zoom={zoom}")
        elif atype in ("scan", "count"):
            lines.append(f"  View: {atype}, {a.get('scan_type','?')}, "
                         f"{a.get('frames',0)} frames"
                         f"{', target=' + a.get('target','') if atype == 'count' else ''}")
        elif atype == "reason":
            lines.append(f"  View: reason (no visual input)")
        else:
            lines.append(f"  View: {atype}")

        if purpose:
            lines.append(f"  Purpose: \"{purpose}\"")
        if finding:
            lines.append(f"  Finding: \"{finding[:400]}\"")
        if fov:
            lines.append(f"  {fov}")
        lines.append("")

    return "\n".join(lines)


# ── v4 Prompt Builders (6-step iteration) ──────────────────────────────────

def build_plan_prompt(question, volume_info, first_look_finding, findings_text,
                      iteration, max_iter):
    """Step 1: OLMo investigation plan (natural language)."""
    if iteration == 1:
        return (
            f"You have examined a 3D volume and received this initial description:\n"
            f"\"{first_look_finding}\"\n\n"
            f"QUESTION: \"{question}\"\n\n"
            f"VOLUME:\n{volume_info.format_for_prompt()}\n\n"
            f"Plan your investigation strategy. What should you look at first, and why?\n\n"
            f"Consider:\n"
            f"- What spatial regions need examination to answer the question?\n"
            f"- Would a different layout (xy vs xz vs yz) reveal different information?\n"
            f"- Would a scan (video sweep) show spatial distribution better than a static view?\n"
            f"- Would toggling layer visibility reveal alignment, segmentation quality, etc.?\n"
            f"- Is the question quantitative (need count action) or qualitative (scan/screenshot)?"
        )
    return (
        f"QUESTION: \"{question}\"\n\n"
        f"INVESTIGATION SO FAR:\n\n{findings_text}\n\n"
        f"Iteration {iteration}/{max_iter}. What should you investigate next, and why?\n\n"
        f"Consider what spatial regions remain unexplored, whether findings are\n"
        f"consistent, and whether you have enough evidence to answer."
    )


def build_action_prompt(plan_text, volume_info, config):
    """Step 2: OLMo strict JSON action from schema."""
    schema = build_action_schema(volume_info, config["max_scan_frames"])
    return (
        f"YOUR INVESTIGATION PLAN:\n{plan_text}\n\n"
        f"{schema}\n\n"
        f"Output the JSON action that executes your plan. Include a \"purpose\" field\n"
        f"explaining what you expect to learn from this view.\n\n"
        f"If you already have enough evidence, use the answer action instead."
    )


def build_vision_instructions_prompt(action, question, findings_text):
    """Step 3: OLMo crafts instructions for Molmo2."""
    action_type = action.get("action", "")
    purpose = action.get("purpose", "")
    action_summary = json.dumps(action, indent=2)[:500]

    findings_lines = findings_text.strip().split("\n── ")
    last_findings = "\n── ".join(findings_lines[-2:]) if len(findings_lines) >= 2 else findings_text[-500:]

    if action_type == "count":
        target = action.get("target", "objects")
        return (
            f"You have planned a count action:\n{action_summary}\n\n"
            f"PURPOSE: {purpose}\nTARGET: \"{target}\"\n\n"
            f"The vision model will point to each instance of the target on sampled keyframes.\n"
            f"Refine the target description to help the model identify the right objects:\n"
            f"- What size and shape are the targets?\n"
            f"- What intensity or color distinguishes them from background?\n"
            f"- Should the model ignore any similar-looking artifacts?\n\n"
            f"Output a refined pointing instruction (1-2 sentences)."
        )
    return (
        f"You have planned this view:\n{action_summary}\n\n"
        f"PURPOSE: {purpose}\nQUESTION: \"{question}\"\n\n"
        f"RECENT FINDINGS:\n{last_findings}\n\n"
        f"Write specific instructions for the vision model that will interpret this view.\n"
        f"Tell it:\n- What specific features or structures to focus on\n"
        f"- What region of the image matters most for this investigation\n"
        f"- What to compare against prior findings (if any)\n"
        f"- Any artifacts or confounds to watch for\n\n"
        f"Keep it concise (2-4 sentences)."
    )


def build_molmo_screenshot_prompt(olmo_instructions, question, action, volume_info):
    """Build Molmo2 screenshot interpretation prompt with OLMo-crafted instructions."""
    view = action.get("view", {})
    layout = view.get("layout", "xy")
    x, y, z = view.get("x", 0), view.get("y", 0), view.get("z", 0)
    zoom = view.get("zoom", "full")
    return (
        f"{olmo_instructions}\n\n---\n\n"
        f"Question: \"{question}\"\n\n"
        f"This is a {layout} view at position ({x:.0f}, {y:.0f}, {z:.0f}), zoom={zoom}.\n"
        f"{volume_info.format_for_prompt()}\n\n"
        f"Describe what you see. Report:\n"
        f"- What structures are present (type, shape, intensity)\n"
        f"- Approximate counts if objects are discrete and countable\n"
        f"- Spatial distribution (clustered, uniform, sparse/dense regions)\n"
        f"- Anything unusual or noteworthy"
    )


def build_molmo_scan_prompt(olmo_instructions, question, action, volume_info,
                            num_frames, frame_spacing, total_dist):
    """Build Molmo2 scan interpretation prompt with OLMo-crafted instructions."""
    layout = action.get("layout", "xy")
    zoom = action.get("zoom", "full")
    scan_axis = action.get("scan_type", "z_sweep").replace("_sweep", "").replace("_pan", "")
    start, end = action.get("start", {}), action.get("end", {})
    return (
        f"{olmo_instructions}\n\n---\n\n"
        f"Question: \"{question}\"\n\n"
        f"Scan: {num_frames} frames along {scan_axis}, "
        f"({start.get('x',0):.0f},{start.get('y',0):.0f},{start.get('z',0):.0f}) → "
        f"({end.get('x',0):.0f},{end.get('y',0):.0f},{end.get('z',0):.0f})\n"
        f"Frame spacing: ~{frame_spacing:.1f}µm, total distance: {total_dist:.0f}µm\n"
        f"Layout: {layout}, zoom: {zoom}\n\n"
        f"Describe what you observe across the frames:\n"
        f"- How does the content change along the scan axis?\n"
        f"- Where are structures most dense vs sparse?\n"
        f"- Are there boundaries, transitions, or abrupt changes?\n"
        f"- Estimate the spatial extent of notable features"
    )


def build_reasoning_prompt(question, finding, findings_text, iteration,
                           fov_feedback=""):
    """Step 6: OLMo post-finding reasoning."""
    finding_block = finding
    if fov_feedback:
        finding_block += f"\n{fov_feedback}"
    return (
        f"QUESTION: \"{question}\"\n\n"
        f"NEW FINDING (iteration {iteration}):\n{finding_block}\n\n"
        f"INVESTIGATION SO FAR:\n{findings_text}\n\n"
        f"Analyze the new finding in context of your prior investigation:\n\n"
        f"1. Does this finding confirm, contradict, or extend previous findings?\n"
        f"2. What spatial regions remain unexplored?\n"
        f"3. Do you have sufficient evidence to answer the question confidently?\n\n"
        f"If you have enough evidence, respond with your answer:\n"
        f'{{\"action\": \"answer\", \"answer\": \"...\"}}\n\n'
        f"Otherwise, summarize your current understanding and what remains uncertain."
    )


def build_count_reasoning_prompt(question, target, pointing_stats, findings_text,
                                 iteration):
    """Step 6 count variant: OLMo interprets pointing statistics."""
    return (
        f"QUESTION: \"{question}\"\n\n"
        f"COUNT RESULTS (iteration {iteration}):\nTarget: \"{target}\"\n"
        f"{pointing_stats}\n\n"
        f"INVESTIGATION SO FAR:\n{findings_text}\n\n"
        f"Interpret these detection results:\n"
        f"- Account for double-counting (objects spanning multiple z-slices)\n"
        f"- Keyframe spacing vs object size: if spacing < diameter, expect overcounting\n"
        f"- Detection confidence: low-contrast or partial objects may be missed\n\n"
        f"Then decide: enough evidence to answer, or need more investigation?\n\n"
        f"If you have enough evidence, respond with your answer:\n"
        f'{{\"action\": \"answer\", \"answer\": \"...\"}}'
    )


def build_reason_shortcircuit_prompt(question, reason_question, findings_text):
    """Step 6 variant for explicit reason action (short-circuit from step 2)."""
    return (
        f"QUESTION: \"{question}\"\n\n"
        f"INVESTIGATION SO FAR:\n{findings_text}\n\n"
        f"Your reasoning request: \"{reason_question}\"\n\n"
        f"Analyze the evidence and decide your next step:\n"
        f"1. If evidence is sufficient, provide your answer.\n"
        f"2. If findings conflict, identify the contradiction.\n"
        f"3. If critical regions remain unexplored, describe what to investigate next."
    )

# ── Agent Loop (v4 — 6-step iteration) ─────────────────────────────────────────

def run_agent(manager, config: dict, ng_link: str, question: str):
    """Run the v4 agent loop with OLMo reasoning + Molmo vision.

    6-step iteration:
        1. OLMo: investigation plan (natural language)
        2. OLMo: action decision (strict JSON from schema)
        3. OLMo: vision instructions for Molmo2
        4. System: capture view (screenshot / scan / count frames)
        5. Molmo2: interpret the captured view
        6. OLMo: reasoning over finding → continue or answer

    Steps 1-3 and 6 run on OLMo (one swap_to_olmo call stays active).
    Steps 4-5 run on Molmo2 (swap_to_molmo).
    Short-circuit: if step 2 outputs reason or answer, steps 3-5 are skipped.
    """
    from neuroglancer_state import NeuroglancerState
    from playwright.sync_api import sync_playwright

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Transcript log — full prompts and responses, appended live ─────
    transcript_path = RESULTS_DIR / "transcript.md"
    transcript_path.write_text("# molmo-glancer — Transcript\n\n")

    def log_exchange(iteration, step, prompt, response, tokens=None,
                     note=None, elapsed=None):
        """Append a prompt/response pair to the transcript file."""
        def _blockquote(text):
            return "\n".join(f"> {line}" for line in str(text).split("\n"))

        with open(transcript_path, "a") as f:
            f.write(f"---\n\n## Iteration {iteration} — {step}\n\n")
            if note:
                f.write(f"_{note}_\n\n")
            meta_parts = []
            if tokens:
                tok_str = f"{tokens['input_tokens']} in / {tokens['output_tokens']} out"
                if "think_tokens" in tokens:
                    tok_str += f" ({tokens['think_tokens']} think)"
                meta_parts.append(f"Tokens: {tok_str}")
            if elapsed is not None:
                meta_parts.append(f"Time: {elapsed:.1f}s")
            if meta_parts:
                f.write(f"**{' | '.join(meta_parts)}**\n\n")
            f.write(f"### Prompt\n\n{_blockquote(prompt)}\n\n")
            f.write(f"### Response\n\n{_blockquote(response)}\n\n")

    # ── Parse NG state and discover volume ──────────────────────────
    print("\n[Setup] Parsing NG link and discovering volume metadata ...")
    base_state = NeuroglancerState.from_url(ng_link)
    volume_info = discover_volume(base_state.data)

    # ── Save prompt templates for inspection ─────────────────────────
    save_prompt_templates(volume_info, config, question)

    # ── Token tracking ──────────────────────────────────────────────────
    token_usage = {
        "iterations": [],
        "totals": {"input_tokens": 0, "output_tokens": 0},
        "by_model": {
            "molmo": {"input_tokens": 0, "output_tokens": 0},
            "olmo": {"input_tokens": 0, "output_tokens": 0},
        },
    }

    def track(iteration, step, tokens, model_name="molmo", elapsed=None):
        entry = {"iteration": iteration, "step": step, "model": model_name, **tokens}
        if elapsed is not None:
            entry["elapsed_s"] = round(elapsed, 1)
        token_usage["iterations"].append(entry)
        token_usage["totals"]["input_tokens"] += tokens["input_tokens"]
        token_usage["totals"]["output_tokens"] += tokens["output_tokens"]
        token_usage["by_model"][model_name]["input_tokens"] += tokens["input_tokens"]
        token_usage["by_model"][model_name]["output_tokens"] += tokens["output_tokens"]
        t_str = f" ({elapsed:.1f}s)" if elapsed is not None else ""
        print(f"    [{model_name}] {tokens['input_tokens']} in / {tokens['output_tokens']} out{t_str}")

    # ── Agent loop state ────────────────────────────────────────────────
    history = []          # list of {iteration, action_data, finding, fov_feedback}
    frame_cache = {}      # geometry_fingerprint → list of PIL frames
    screenshot_count = 0
    scan_count = 0
    consecutive_duplicates = 0
    max_consecutive_duplicates = 3
    final_answer = None
    max_iter = config["max_agent_iterations"]

    print(f"\n[Agent] Starting loop (max {max_iter} iterations)")
    print(f"  Question: {question}\n")

    with sync_playwright() as pw:
        browser, page = create_browser(pw)

        # ── Phase 1: First Look (Molmo2) — "What am I looking at?" ────
        print("[Phase 1] First Look — What am I looking at?")
        manager.swap_to_molmo()
        screenshot_count += 1
        cx = volume_info.shape[0] / 2
        cy = volume_info.shape[1] / 2
        cz = volume_info.shape[2] / 2
        first_look_state = build_clean_state(base_state, {
            "x": cx, "y": cy, "z": cz,
        }, volume_info)
        first_look_img = capture_screenshot(page, first_look_state, screenshot_count)

        first_look_prompt = (
            f"This is a Neuroglancer view of a 3D volume "
            f"({volume_info.shape[0]}\u00d7{volume_info.shape[1]}\u00d7{volume_info.shape[2]} voxels).\n"
            f"{volume_info.format_for_prompt()}\n\n"
            f"Describe what you see: what kind of data, what structures are visible, "
            f"how dense or sparse is the content?"
        )
        t0 = time.time()
        first_look_finding, tokens = ask_vision(
            manager.molmo_model, manager.molmo_processor,
            first_look_img, first_look_prompt, max_new_tokens=512,
        )
        elapsed = time.time() - t0
        track(0, "first_look", tokens, "molmo", elapsed)
        log_exchange(0, "first_look", first_look_prompt, first_look_finding,
                     tokens, "image: view_001.png", elapsed)
        print(f"  First look: {first_look_finding[:300]}...")

        history.append({
            "iteration": 0,
            "action_data": {
                "action": "screenshot",
                "view": {"layout": base_state.data.get("layout", "4panel")},
                "prompt": "Phase 1: What am I looking at?",
            },
            "finding": first_look_finding,
            "fov_feedback": "[user's original view — default zoom and position]",
        })

        # ── 6-step agent loop ────────────────────────────────────────
        for iteration in range(1, max_iter + 1):
            print(f"\n{'='*60}")
            print(f"  Iteration {iteration}/{max_iter}")
            vram = get_vram_usage()
            print(f"  VRAM: {vram['allocated']:.1f} / {vram['total']:.1f} GB")
            print(f"{'='*60}")

            findings_text = format_structured_findings(history)

            # ── Step 1: OLMo investigation plan ──────────────────────
            manager.swap_to_olmo()
            print("\n  [Step 1] OLMo: Investigation plan ...")
            plan_prompt = build_plan_prompt(
                question, volume_info, first_look_finding, findings_text,
                iteration, max_iter,
            )
            t0 = time.time()
            plan_text, tokens, _ = ask_text_olmo(
                manager, OLMO_SYSTEM_PROMPT, plan_prompt,
                max_new_tokens=config["olmo_max_new_tokens_plan"],
                sampling=config["olmo_sampling_structured"],
            )
            elapsed = time.time() - t0
            track(iteration, "plan", tokens, "olmo", elapsed)
            log_exchange(iteration, "step1_plan", plan_prompt, plan_text,
                         tokens, elapsed=elapsed)
            print(f"  Plan: {plan_text[:300]}...")

            # ── Step 2: OLMo action decision (strict JSON) ──────────
            print("\n  [Step 2] OLMo: Action decision ...")
            action_prompt = build_action_prompt(plan_text, volume_info, config)
            t0 = time.time()
            action_text, tokens, _ = ask_text_olmo(
                manager, OLMO_SYSTEM_PROMPT, action_prompt,
                max_new_tokens=config["olmo_max_new_tokens_decision"],
                sampling=config["olmo_sampling_structured"],
            )
            elapsed = time.time() - t0
            track(iteration, "action", tokens, "olmo", elapsed)
            log_exchange(iteration, "step2_action", action_prompt, action_text,
                         tokens, elapsed=elapsed)
            print(f"  Action text: {action_text[:200]}...")

            # Parse JSON action
            action = parse_action(action_text)
            if action is None:
                print("  WARNING: Could not parse action JSON. Retrying ...")
                retry_prompt = (
                    action_prompt
                    + "\n\nYour previous response was not valid JSON. "
                    "Please respond with ONLY a JSON object."
                )
                t0 = time.time()
                action_text, tokens, _ = ask_text_olmo(
                    manager, OLMO_SYSTEM_PROMPT, retry_prompt,
                    max_new_tokens=config["olmo_max_new_tokens_retry"],
                    sampling=config["olmo_sampling_retry"],
                )
                elapsed = time.time() - t0
                track(iteration, "action_retry", tokens, "olmo", elapsed)
                log_exchange(iteration, "step2_retry", retry_prompt, action_text,
                             tokens, elapsed=elapsed)
                action = parse_action(action_text)

            if action is None:
                print("  ERROR: Failed to parse after retry. Forcing reason.")
                action = {
                    "action": "reason",
                    "question": f"JSON parse failed: {action_text[:200]}",
                }

            action = validate_action(action, volume_info)
            action_type = action.get("action", "unknown")
            print(f"\n  [Action] {action_type}")

            # ── Short-circuit: answer ────────────────────────────────
            if action_type == "answer":
                final_answer = action.get("answer", "")
                print(f"\n  [ANSWER] {final_answer}")
                history.append({
                    "iteration": iteration,
                    "action_data": action,
                    "finding": final_answer,
                    "fov_feedback": "",
                })
                break

            # ── Short-circuit: reason (skip steps 3-5) ───────────────
            if action_type == "reason":
                reason_q = action.get("question", "Synthesize findings.")
                print("  [Short-circuit] reason — skipping steps 3-5")
                reason_prompt = build_reason_shortcircuit_prompt(
                    question, reason_q, findings_text,
                )
                t0 = time.time()
                finding, tokens, _ = ask_text_olmo(
                    manager, OLMO_SYSTEM_PROMPT, reason_prompt,
                    max_new_tokens=config["olmo_max_new_tokens_reasoning"],
                    sampling=config["olmo_sampling_structured"],
                )
                elapsed = time.time() - t0
                track(iteration, "reason_shortcircuit", tokens, "olmo", elapsed)
                log_exchange(iteration, "reason_shortcircuit", reason_prompt,
                             finding, tokens, elapsed=elapsed)
                print(f"  Reasoning: {finding[:200]}...")

                # Check if OLMo decided to answer within reasoning
                embedded = parse_action(finding)
                if embedded and embedded.get("action") == "answer":
                    final_answer = embedded.get("answer", finding)
                    print(f"\n  [ANSWER from reasoning] {final_answer}")
                    history.append({
                        "iteration": iteration,
                        "action_data": {"action": "answer", "answer": final_answer},
                        "finding": final_answer,
                        "fov_feedback": "",
                    })
                    break

                history.append({
                    "iteration": iteration,
                    "action_data": action,
                    "finding": finding,
                    "fov_feedback": "",
                })
                continue

            # ── Duplicate check ─────────────────────────────────────
            prior_count = count_prior_matches(action, history)
            if prior_count >= 2:
                consecutive_duplicates += 1
                print(f"  BLOCKED: action done {prior_count} times — forcing reason")
                if consecutive_duplicates >= max_consecutive_duplicates:
                    print("  [Forced] Too many duplicates — will force answer.")

                dup_prompt = (
                    f"QUESTION: \"{question}\"\n\n"
                    f"INVESTIGATION SO FAR:\n{findings_text}\n\n"
                    f"You just tried to repeat an action done {prior_count} times.\n"
                    f"What should you do DIFFERENTLY, or do you have enough "
                    f"evidence to answer?"
                )
                t0 = time.time()
                finding, tokens, _ = ask_text_olmo(
                    manager, OLMO_SYSTEM_PROMPT, dup_prompt,
                    max_new_tokens=config["olmo_max_new_tokens_reasoning"],
                    sampling=config["olmo_sampling_structured"],
                )
                elapsed = time.time() - t0
                track(iteration, "forced_reason", tokens, "olmo", elapsed)
                log_exchange(iteration, "forced_reason", dup_prompt, finding,
                             tokens, elapsed=elapsed)
                print(f"  Forced reasoning: {finding[:200]}...")

                history.append({
                    "iteration": iteration,
                    "action_data": {"action": "reason",
                                    "question": "[forced — repeated action blocked]"},
                    "finding": finding,
                    "fov_feedback": "",
                })
                continue
            else:
                consecutive_duplicates = 0

            # ── Step 3: OLMo vision instructions for Molmo2 ─────────
            print("\n  [Step 3] OLMo: Vision instructions ...")
            instr_prompt = build_vision_instructions_prompt(
                action, question, findings_text,
            )
            t0 = time.time()
            olmo_instructions, tokens, _ = ask_text_olmo(
                manager, OLMO_SYSTEM_PROMPT, instr_prompt,
                max_new_tokens=config["olmo_max_new_tokens_vision_instr"],
                sampling=config["olmo_sampling_structured"],
            )
            elapsed = time.time() - t0
            track(iteration, "vision_instructions", tokens, "olmo", elapsed)
            log_exchange(iteration, "step3_vision_instr", instr_prompt,
                         olmo_instructions, tokens, elapsed=elapsed)
            print(f"  Instructions: {olmo_instructions[:200]}...")

            # ── Steps 4-5: Molmo2 capture + interpret ────────────────
            manager.swap_to_molmo()
            finding = ""
            fov_feedback = ""

            if action_type == "screenshot":
                # Step 4: Capture screenshot
                view_spec = action.get("view", {})
                screenshot_count += 1
                state = build_clean_state(base_state, view_spec, volume_info)
                img = capture_screenshot(page, state, screenshot_count)

                # Step 5: Molmo2 interprets
                interpret_prompt = build_molmo_screenshot_prompt(
                    olmo_instructions, question, action, volume_info,
                )
                print(f"  [Step 5] Molmo2: Interpreting screenshot ...")
                t0 = time.time()
                finding, tokens = ask_vision(
                    manager.molmo_model, manager.molmo_processor,
                    img, interpret_prompt, max_new_tokens=1024,
                )
                elapsed = time.time() - t0
                track(iteration, "interpret_screenshot", tokens, "molmo", elapsed)
                log_exchange(iteration, "step5_interpret", interpret_prompt,
                             finding, tokens,
                             f"image: view_{screenshot_count:03d}.png", elapsed)
                print(f"  Finding: {finding[:200]}...")

                # FOV feedback
                pos = [view_spec.get("x", 0), view_spec.get("y", 0),
                       view_spec.get("z", 0)]
                scale = view_spec.get("crossSectionScale", 1.0)
                layout = view_spec.get("layout", "xy")
                fov_feedback = format_fov_feedback(pos, scale, layout, volume_info)
                print(f"  {fov_feedback}")

            elif action_type == "scan":
                # Step 4: Capture scan
                scan_count += 1
                geo_fp = _geometry_fingerprint(action)
                if geo_fp in frame_cache:
                    frames = frame_cache[geo_fp]
                    print(f"  [Cache hit] Reusing {len(frames)} frames")
                    save_scan_video(frames, scan_count)
                else:
                    frames = execute_scan(base_state, action, volume_info,
                                          config, scan_count)
                    if geo_fp:
                        frame_cache[geo_fp] = frames

                # Compute spatial context
                scan_start = action.get("start", {})
                scan_end = action.get("end", {})
                s = np.array([scan_start.get("x", cx), scan_start.get("y", cy),
                              scan_start.get("z", cz)])
                e = np.array([scan_end.get("x", cx), scan_end.get("y", cy),
                              scan_end.get("z", cz)])
                total_dist = float(np.linalg.norm(e - s))
                frame_spacing = total_dist / max(len(frames) - 1, 1)

                # Step 5: Molmo2 interprets scan
                interpret_prompt = build_molmo_scan_prompt(
                    olmo_instructions, question, action, volume_info,
                    len(frames), frame_spacing, total_dist,
                )
                print(f"  [Step 5] Molmo2: Interpreting scan "
                      f"({len(frames)} frames) ...")
                t0 = time.time()
                finding, tokens = ask_scan(
                    manager.molmo_model, manager.molmo_processor,
                    frames, interpret_prompt,
                    max_new_tokens=1024, config=config,
                )
                elapsed = time.time() - t0
                track(iteration, "interpret_scan", tokens, "molmo", elapsed)
                log_exchange(iteration, "step5_interpret_scan", interpret_prompt,
                             finding, tokens,
                             f"video: scan_{scan_count:03d}.mp4, "
                             f"{len(frames)} frames", elapsed)
                print(f"  Finding: {finding[:200]}...")

            elif action_type == "count":
                # Step 4: Capture + per-keyframe pointing
                scan_count += 1
                target = action.get("target", "objects")

                geo_fp = _geometry_fingerprint(action)
                if geo_fp in frame_cache:
                    frames = frame_cache[geo_fp]
                    print(f"  [Cache hit] Reusing {len(frames)} frames")
                    save_scan_video(frames, scan_count)
                else:
                    frames = execute_scan(base_state, action, volume_info,
                                          config, scan_count)
                    if geo_fp:
                        frame_cache[geo_fp] = frames

                # Step 5: Per-keyframe pointing with OLMo-refined target
                keyframe_interval = max(1, int(
                    action.get("keyframe_interval", 5)))
                keyframe_indices = list(range(0, len(frames),
                                              keyframe_interval))
                print(f"  [Step 5] Pointing to '{target}' on "
                      f"{len(keyframe_indices)} keyframes ...")

                # Use OLMo instructions as refined pointing prompt
                point_prompt = (olmo_instructions.strip()
                                if olmo_instructions.strip()
                                else f"Point to the {target}.")

                points = []
                total_point_tokens = {"input_tokens": 0, "output_tokens": 0}

                t0 = time.time()
                for ki in keyframe_indices:
                    _, frame_points, tokens = ask_vision_pointing(
                        manager.molmo_model, manager.molmo_processor,
                        frames[ki], point_prompt, max_new_tokens=2048,
                    )
                    total_point_tokens["input_tokens"] += tokens["input_tokens"]
                    total_point_tokens["output_tokens"] += tokens["output_tokens"]
                    for x, y in frame_points:
                        points.append((float(ki), x, y))
                    print(f"    keyframe {ki}: {len(frame_points)} points")
                elapsed = time.time() - t0

                track(iteration, "count_point", total_point_tokens, "molmo",
                      elapsed)
                pointing_summary = "\n".join(
                    f"  keyframe {ki}: "
                    f"{sum(1 for p in points if int(p[0]) == ki)} points"
                    for ki in keyframe_indices
                )
                log_exchange(iteration, "count_point", point_prompt,
                             pointing_summary, total_point_tokens,
                             f"video: scan_{scan_count:03d}.mp4, "
                             f"{len(keyframe_indices)} keyframes", elapsed)
                print(f"  Pointing total: {len(points)} points")

                if points:
                    annotate_scan_frames(frames, points, scan_count)

                # Build pointing stats for OLMo reasoning (step 6)
                scan_start = action.get("start", {})
                scan_end = action.get("end", {})
                s = np.array([scan_start.get("x", cx), scan_start.get("y", cy),
                              scan_start.get("z", cz)])
                e = np.array([scan_end.get("x", cx), scan_end.get("y", cy),
                              scan_end.get("z", cz)])
                total_dist = float(np.linalg.norm(e - s))
                frame_spacing = total_dist / max(len(frames) - 1, 1)
                scan_axis = action.get("scan_type", "z_sweep").replace(
                    "_sweep", "").replace("_pan", "")

                frame_ids = sorted(set(int(p[0]) for p in points)) \
                    if points else []
                points_per_frame = {}
                for p in points:
                    fid = int(p[0])
                    points_per_frame[fid] = points_per_frame.get(fid, 0) + 1

                keyframe_spacing = keyframe_interval * frame_spacing
                pointing_stats = (
                    f"Scan: {len(frames)} frames along {scan_axis}, "
                    f"~{frame_spacing:.1f}\u00b5m between frames, "
                    f"~{keyframe_spacing:.1f}\u00b5m between keyframes, "
                    f"{total_dist:.0f}\u00b5m total.\n"
                    f"Detected {len(points)} points across "
                    f"{len(frame_ids)}/{len(keyframe_indices)} keyframes.\n"
                )
                if points_per_frame:
                    counts = sorted(points_per_frame.values())
                    pointing_stats += (
                        f"Points per keyframe: min={counts[0]}, "
                        f"max={counts[-1]}, "
                        f"median={counts[len(counts)//2]}.\n"
                    )

                # Convert pixel detections to physical coordinates
                scale = action.get("crossSectionScale",
                                   max(volume_info.shape[0],
                                       volume_info.shape[1]) / 1024)
                layout = action.get("layout", "xy")
                phys_points = []
                if points:
                    n_frames = max(len(frames) - 1, 1)
                    for frame_idx, px, py in points:
                        t = frame_idx / n_frames
                        center = (
                            s[0] + t * (e[0] - s[0]),
                            s[1] + t * (e[1] - s[1]),
                            s[2] + t * (e[2] - s[2]),
                        )
                        phys = pixel_to_physical(
                            px, py, center, scale, layout)
                        phys_points.append(phys)

                spatial_summary = summarize_spatial_distribution(
                    phys_points, volume_info) if phys_points else ""
                if spatial_summary:
                    pointing_stats += f"\n{spatial_summary}\n"

                finding = (
                    f"DETECTED (automated pointing): {len(points)} instances "
                    f"of '{target}' across {len(frame_ids)}/"
                    f"{len(keyframe_indices)} keyframes "
                    f"(from {len(frames)} total frames)."
                )

            # ── Step 6: OLMo reasoning over finding ─────────────────
            manager.swap_to_olmo()
            print(f"\n  [Step 6] OLMo: Reasoning over finding ...")

            if action_type == "count":
                reasoning_prompt = build_count_reasoning_prompt(
                    question, target, pointing_stats, findings_text,
                    iteration,
                )
            else:
                reasoning_prompt = build_reasoning_prompt(
                    question, finding, findings_text, iteration,
                    fov_feedback=fov_feedback,
                )

            t0 = time.time()
            reasoning_text, tokens, _ = ask_text_olmo(
                manager, OLMO_SYSTEM_PROMPT, reasoning_prompt,
                max_new_tokens=config["olmo_max_new_tokens_reasoning"],
                sampling=config["olmo_sampling_structured"],
            )
            elapsed = time.time() - t0
            track(iteration, "reasoning", tokens, "olmo", elapsed)
            log_exchange(iteration, "step6_reasoning", reasoning_prompt,
                         reasoning_text, tokens, elapsed=elapsed)
            print(f"  Reasoning: {reasoning_text[:200]}...")

            # Append OLMo interpretation to count findings
            if action_type == "count":
                finding += f" {reasoning_text}"

            # Check if OLMo decided to answer in step 6
            embedded = parse_action(reasoning_text)
            if embedded and embedded.get("action") == "answer":
                final_answer = embedded.get("answer", reasoning_text)
                print(f"\n  [ANSWER from step 6] {final_answer}")
                history.append({
                    "iteration": iteration,
                    "action_data": {"action": "answer",
                                    "answer": final_answer},
                    "finding": final_answer,
                    "fov_feedback": fov_feedback,
                })
                break

            # ── Append finding to history ───────────────────────────
            history.append({
                "iteration": iteration,
                "action_data": action,
                "finding": finding,
                "fov_feedback": fov_feedback,
            })

        browser.close()

    # ── If loop ended without answer, force one via OLMo ─────────────
    if final_answer is None:
        print("\n  [Forced Answer] Max iterations — OLMo synthesis ...")
        manager.swap_to_olmo()
        findings_text = format_structured_findings(history)
        synth_prompt = (
            f"QUESTION: \"{question}\"\n\n"
            f"INVESTIGATION COMPLETE \u2014 ALL FINDINGS:\n{findings_text}\n\n"
            f"You have reached the maximum number of iterations.\n"
            f"Synthesize ALL findings into a comprehensive answer.\n"
            f"Be specific: include counts, spatial descriptions, "
            f"and confidence level."
        )
        t0 = time.time()
        answer_text, tokens, _ = ask_text_olmo(
            manager, OLMO_SYSTEM_PROMPT, synth_prompt,
            max_new_tokens=config["olmo_max_new_tokens_synthesis"],
            sampling=config["olmo_sampling_synthesis"],
        )
        elapsed = time.time() - t0
        track(max_iter, "forced_answer", tokens, "olmo", elapsed)
        log_exchange(max_iter, "forced_answer", synth_prompt,
                     answer_text, tokens, elapsed=elapsed)

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

    md = []
    md.append("# molmo-glancer v4 — Prompt Templates\n")
    md.append(f"Generated for question: *{question}*\n")
    md.append(f"Volume: {volume_info.format_for_prompt()}\n")

    # OLMo system prompt
    md.append("---\n")
    md.append("## OLMo System Prompt\n")
    md.append("Sent as system role in every OLMo call (steps 1, 2, 3, 6).\n")
    md.append("```")
    md.append(OLMO_SYSTEM_PROMPT)
    md.append("```\n")

    # Phase 1: First Look (Molmo2)
    md.append("---\n")
    md.append("## Phase 1: First Look (Molmo2, image + text)\n")
    md.append("Sent with a center-position screenshot.\n")
    md.append("```")
    md.append(
        f"This is a Neuroglancer view of a 3D volume "
        f"({volume_info.shape[0]:.0f}\u00d7{volume_info.shape[1]:.0f}"
        f"\u00d7{volume_info.shape[2]:.0f} \u00b5m).\n"
        f"{volume_info.format_for_prompt()}\n\n"
        f"Describe what you see: what kind of data, what structures are visible, "
        f"how dense or sparse is the content?"
    )
    md.append("```\n")

    # Step 1: Investigation Plan (OLMo)
    md.append("---\n")
    md.append("## Step 1: Investigation Plan (OLMo, text-only)\n")
    md.append(f"Token budget: {config['olmo_max_new_tokens_plan']}\n")
    md.append("### Iteration 1 variant:\n```")
    md.append(
        'You have examined a 3D volume and received this initial description:\n'
        '"{first_look_finding}"\n\n'
        f'QUESTION: "{question}"\n\n'
        f'VOLUME:\n{volume_info.format_for_prompt()}\n\n'
        'Plan your investigation strategy. What should you look at first, and why?'
    )
    md.append("```\n### Iteration N variant:\n```")
    md.append(
        f'QUESTION: "{question}"\n\n'
        'INVESTIGATION SO FAR:\n{findings_text}\n\n'
        'Iteration N/M. What should you investigate next, and why?'
    )
    md.append("```\n")

    # Step 2: Action Decision (OLMo)
    md.append("---\n")
    md.append("## Step 2: Action Decision (OLMo, text-only)\n")
    md.append(f"Token budget: {config['olmo_max_new_tokens_decision']}\n")
    md.append("### Action Schema:\n```")
    md.append(build_action_schema(volume_info, config["max_scan_frames"]))
    md.append("```\n### Wrapper:\n```")
    md.append(
        'YOUR INVESTIGATION PLAN:\n{plan_text}\n\n'
        '{action_schema}\n\n'
        'Output the JSON action that executes your plan. '
        'Include a "purpose" field explaining what you expect to learn.\n\n'
        'If you already have enough evidence, use the answer action instead.'
    )
    md.append("```\n")

    # Step 3: Vision Instructions (OLMo)
    md.append("---\n")
    md.append("## Step 3: Vision Instructions (OLMo, text-only)\n")
    md.append(f"Token budget: {config['olmo_max_new_tokens_vision_instr']}\n")
    md.append("### Screenshot/scan variant:\n```")
    md.append(
        'You have planned this view:\n{action_json}\n\n'
        'PURPOSE: {purpose}\nQUESTION: "{question}"\n\n'
        'RECENT FINDINGS:\n{last_findings}\n\n'
        'Write specific instructions for the vision model (2-4 sentences).'
    )
    md.append("```\n### Count variant:\n```")
    md.append(
        'You have planned a count action:\n{action_json}\n\n'
        'PURPOSE: {purpose}\nTARGET: "{target}"\n\n'
        'Refine the target description for the pointing model (1-2 sentences).'
    )
    md.append("```\n")

    # Step 5: Molmo2 Interpretation
    md.append("---\n")
    md.append("## Step 5: Screenshot Interpret (Molmo2, image + text)\n")
    md.append("```")
    md.append(
        '{olmo_instructions}\n\n---\n\n'
        f'Question: "{question}"\n\n'
        'This is a {{layout}} view at position (x, y, z), zoom={{zoom}}.\n'
        f'{volume_info.format_for_prompt()}\n\n'
        'Describe what you see. Report structures, counts, distribution.'
    )
    md.append("```\n")

    md.append("---\n")
    md.append("## Step 5: Scan Interpret (Molmo2, video + text)\n")
    md.append("```")
    md.append(
        '{olmo_instructions}\n\n---\n\n'
        f'Question: "{question}"\n\n'
        'Scan: {{num_frames}} frames along {{axis}}, ~{{spacing}}\u00b5m between frames.\n'
        'Describe what you observe across the frames.'
    )
    md.append("```\n")

    md.append("---\n")
    md.append("## Step 5: Count Pointing (Molmo2, image + text)\n")
    md.append("OLMo-refined target description sent per keyframe.\n")
    md.append("```")
    md.append("{olmo_refined_target_description}")
    md.append("```\n")

    # Step 6: OLMo Reasoning
    md.append("---\n")
    md.append("## Step 6: Reasoning (OLMo, text-only)\n")
    md.append(f"Token budget: {config['olmo_max_new_tokens_reasoning']}\n")
    md.append("### Screenshot/scan variant:\n```")
    md.append(
        f'QUESTION: "{question}"\n\n'
        'NEW FINDING (iteration N):\n{finding}\n\n'
        'INVESTIGATION SO FAR:\n{findings_text}\n\n'
        'Analyze the new finding. Confirm, contradict, or extend prior findings.\n'
        'If enough evidence, respond with: {"action": "answer", "answer": "..."}'
    )
    md.append("```\n### Count variant:\n```")
    md.append(
        f'QUESTION: "{question}"\n\n'
        'COUNT RESULTS (iteration N):\nTarget: "{{target}}"\n{pointing_stats}\n\n'
        'INVESTIGATION SO FAR:\n{findings_text}\n\n'
        'Interpret detections: account for double-counting, keyframe spacing vs '
        'object size, detection confidence.'
    )
    md.append("```\n")

    # Forced Answer
    md.append("---\n")
    md.append("## Forced Answer / Synthesis (OLMo, text-only)\n")
    md.append(f"Token budget: {config['olmo_max_new_tokens_synthesis']}\n")
    md.append("```")
    md.append(
        f'QUESTION: "{question}"\n\n'
        'INVESTIGATION COMPLETE \u2014 ALL FINDINGS:\n{findings_text}\n\n'
        'You have reached the maximum number of iterations.\n'
        'Synthesize ALL findings into a comprehensive answer.\n'
        'Be specific: include counts, spatial descriptions, and confidence level.'
    )
    md.append("```\n")

    # Zoom table
    md.append("---\n")
    md.append("## Appendix: Zoom Options\n")
    md.append("```")
    md.append(format_zoom_table())
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
    parser = argparse.ArgumentParser(description="molmo-glancer")
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
    print("  molmo-glancer — Autonomous Neuroglancer Visual Analysis")
    print("=" * 60)

    # Initialize model manager
    print("\n[1/3] Initializing ModelManager ...")
    manager = ModelManager()
    config = CONFIG

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
    answer = run_agent(manager, config, ng_link, question)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
