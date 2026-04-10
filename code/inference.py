"""
inference — Model inference wrappers for Molmo2 and OLMo.

Molmo2-O-7B: vision+text (image, video, pointing).
OLMo 3.1 32B Think: text reasoning with think token handling.
"""

import re

import torch
from PIL import Image

from gpu_config import CONFIG

# ── Molmo2 Inference ──────────────────────────────────────────────────────


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


# ── Pointing / Counting ──────────────────────────────────────────────────

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
    from molmo_utils import process_vision_info

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


# ── OLMo Text Generation ─────────────────────────────────────────────────

OLMO_SYSTEM_PROMPT = """\
You are the reasoning engine of an autonomous visual analysis system.
You analyze 3D volumetric microscopy data loaded in Neuroglancer.

Your partner is a vision model (Molmo2) that captures and interprets
screenshots and video scans of the data. You cannot see images directly.
You plan what views to capture, and Molmo2 reports back what it sees.

Think carefully about what each finding means before deciding your next step."""


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
                  max_new_tokens: int = 4096, sampling: dict | None = None,
                  think: bool = True):
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
    think : bool
        If False, append </think> to input so model skips thinking and
        responds directly. Saves tokens and time for structured outputs.
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

    # Disable thinking by closing the <think> block immediately
    if not think:
        close_ids = tokenizer.encode("</think>\n", add_special_tokens=False)
        close_tensor = torch.tensor([close_ids], device=input_ids.device)
        input_ids = torch.cat([input_ids, close_tensor], dim=1)

    input_len = input_ids.shape[1]

    gen_kwargs = dict(sampling or CONFIG["olmo_sampling_structured"])
    gen_kwargs["max_new_tokens"] = max_new_tokens

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **gen_kwargs)

    generated = output_ids[0, input_len:]
    raw_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    clean_text, think_content, was_truncated = strip_think_tokens(raw_text)

    # Detect invisible truncation: skip_special_tokens strips <think>, so when
    # the model hits max_new_tokens mid-thought (no </think> generated),
    # strip_think_tokens sees no markers and returns think content as "clean".
    if (think and not was_truncated and not think_content
            and len(generated) >= max_new_tokens
            and "</think>" not in raw_text):
        was_truncated = True
        think_content = clean_text
        clean_text = ""

    if was_truncated:
        print(f"  WARNING: OLMo think block truncated at {max_new_tokens} tokens"
              f" — no usable response generated")

    think_tok_count = (
        sum(len(tokenizer.encode(b)) for b in think_content.split('\n'))
        if think_content else 0
    )
    token_counts = {
        "input_tokens": input_len,
        "output_tokens": len(generated),
        "think_tokens": think_tok_count,
        "think_content": think_content if think_content else "",
        "was_truncated": was_truncated,
    }

    return clean_text, token_counts, raw_text
