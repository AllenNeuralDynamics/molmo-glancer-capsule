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
import time
from pathlib import Path

import numpy as np

from gpu_config import ModelManager, get_vram_usage, CONFIG
from volume_info import (
    VolumeInfo, discover_volume, format_fov_feedback,
    pixel_to_physical, summarize_spatial_distribution,
)
from visual_capture import (
    build_clean_state, capture_screenshot, execute_scan,
    create_browser, save_scan_video, annotate_scan_frames,
)
from inference import (
    ask_vision, ask_scan, ask_vision_pointing,
    ask_text_olmo, OLMO_SYSTEM_PROMPT,
)
from prompts import (
    build_action_schema, format_structured_findings,
    build_plan_prompt, build_action_prompt,
    build_molmo_screenshot_prompt, build_molmo_scan_prompt,
    build_reason_shortcircuit_prompt,
)
from actions import (
    parse_action, validate_action,
    geometry_fingerprint, count_prior_matches,
)

# ── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("/results")
PRESETS_DIR = Path("/root/capsule/code/presets")

# Defaults — overridden by --preset or env vars
NG_LINK_FILE = os.environ.get("NG_LINK_FILE",
    "/root/capsule/code/ng_links/example_ng_link.txt")
QUESTION = os.environ.get("QUESTION",
    "How many neurons can you count in this volume?")


def discover_presets(presets_dir: Path = PRESETS_DIR) -> dict[str, dict]:
    """Load all preset JSON files from the presets directory.

    Each JSON file must contain {"ng_link": "...", "question": "..."}.
    The preset name is the filename stem (e.g. neurons.json -> "neurons").
    """
    presets = {}
    if not presets_dir.is_dir():
        return presets
    for f in sorted(presets_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        if "ng_link" not in data or "question" not in data:
            print(f"  WARNING: skipping {f.name} — missing ng_link or question")
            continue
        presets[f.stem] = data
    return presets

# ── Agent Loop (v4 — 4-step iteration) ─────────────────────────────────────────


def run_agent(manager, config: dict, ng_link: str, question: str):
    """Run the v4 agent loop with OLMo reasoning + Molmo vision.

    4-step iteration:
        1. OLMo (think=True):  reason over last finding + plan next action
        2. OLMo (think=False): action JSON (with prompt) from schema
        3. System: capture view (screenshot / scan / count frames)
        4. Molmo2: interpret the captured view

    Steps 1-2 run on OLMo (one swap_to_olmo call).
    Steps 3-4 run on Molmo2 (swap_to_molmo).
    One OLMo swap saved per iteration vs the 6-step design.
    Short-circuit: if step 2 outputs reason or answer, steps 3-4 are skipped.
    """
    from neuroglancer_state import NeuroglancerState
    from playwright.sync_api import sync_playwright

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Transcript log — full prompts and responses, appended live ─────
    transcript_path = RESULTS_DIR / "transcript.md"
    transcript_path.write_text("# molmo-glancer — Transcript\n\n")

    def log_exchange(iteration, step, prompt, response, tokens=None,
                     note=None, elapsed=None, think_content=None):
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
                if tokens.get("think_tokens", 0) > 0:
                    tok_str += f" ({tokens['think_tokens']} think)"
                meta_parts.append(f"Tokens: {tok_str}")
            if elapsed is not None:
                meta_parts.append(f"Time: {elapsed:.1f}s")
            if meta_parts:
                f.write(f"**{' | '.join(meta_parts)}**\n\n")
            f.write(f"### Prompt\n\n{_blockquote(prompt)}\n\n")
            f.write(f"### Response\n\n{_blockquote(response)}\n\n")
            if think_content:
                f.write(f"<details>\n<summary>Thinking ({tokens.get('think_tokens', '?')} tokens)</summary>\n\n")
                f.write(f"{_blockquote(think_content)}\n\n")
                f.write(f"</details>\n\n")

    # ── Parse NG state and discover volume ──────────────────────────
    print("\n[Setup] Parsing NG link and discovering volume metadata ...")
    base_state = NeuroglancerState.from_url(ng_link)
    volume_info = discover_volume(base_state.data)

    # ── Save prompt templates for inspection ─────────────────────────
    save_prompt_templates(volume_info, config, question)

    # ── Token tracking ──────────────────────────────────────────────────
    token_usage = {
        "iterations": [],
        "totals": {"input_tokens": 0, "output_tokens": 0, "think_tokens": 0},
        "by_model": {
            "molmo": {"input_tokens": 0, "output_tokens": 0},
            "olmo": {"input_tokens": 0, "output_tokens": 0, "think_tokens": 0},
        },
    }

    def track(iteration, step, tokens, model_name="molmo", elapsed=None):
        # Exclude think_content (big string) from serializable entry
        nums = {k: v for k, v in tokens.items() if k != "think_content"}
        entry = {"iteration": iteration, "step": step, "model": model_name, **nums}
        if elapsed is not None:
            entry["elapsed_s"] = round(elapsed, 1)
        token_usage["iterations"].append(entry)
        token_usage["totals"]["input_tokens"] += tokens["input_tokens"]
        token_usage["totals"]["output_tokens"] += tokens["output_tokens"]
        think = tokens.get("think_tokens", 0)
        token_usage["totals"]["think_tokens"] += think
        token_usage["by_model"][model_name]["input_tokens"] += tokens["input_tokens"]
        token_usage["by_model"][model_name]["output_tokens"] += tokens["output_tokens"]
        if "think_tokens" in token_usage["by_model"][model_name]:
            token_usage["by_model"][model_name]["think_tokens"] += think
        t_str = f" ({elapsed:.1f}s)" if elapsed is not None else ""
        tk_str = f" [{think} think]" if think else ""
        print(f"    [{model_name}] {tokens['input_tokens']} in / {tokens['output_tokens']} out{tk_str}{t_str}")

    # ── Agent loop state ────────────────────────────────────────────────
    history = []          # list of {iteration, action_data, finding, fov_feedback}
    frame_cache = {}      # geometry_fingerprint → list of PIL frames
    screenshot_count = 0
    scan_count = 0
    consecutive_duplicates = 0
    max_consecutive_duplicates = 3
    final_answer = None
    max_iter = config["max_agent_iterations"]
    min_iter = config.get("min_iterations_before_answer", 3)
    last_finding = None   # carried from iteration N to N+1 for reasoning
    last_fov = None

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
            f"Describe what you see, focusing on what's relevant to the question:\n"
            f"\"{question}\""
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

        # ── 4-step agent loop ────────────────────────────────────────
        for iteration in range(1, max_iter + 1):
            print(f"\n{'='*60}")
            print(f"  Iteration {iteration}/{max_iter}")
            vram = get_vram_usage()
            print(f"  VRAM: {vram['allocated']:.1f} / {vram['total']:.1f} GB")
            print(f"{'='*60}")

            findings_text = format_structured_findings(history)

            # ── Step 1: OLMo reason + plan (think=True) ──────────────
            manager.swap_to_olmo()
            print("\n  [Step 1] OLMo: Reason + Plan ...")
            plan_prompt = build_plan_prompt(
                question, volume_info, first_look_finding, findings_text,
                iteration,
                last_finding=last_finding, last_fov=last_fov,
            )
            t0 = time.time()
            plan_text, tokens, _ = ask_text_olmo(
                manager, OLMO_SYSTEM_PROMPT, plan_prompt,
                max_new_tokens=config["olmo_max_new_tokens_plan"],
                sampling=config["olmo_sampling_structured"],
            )
            elapsed = time.time() - t0
            track(iteration, "reason_plan", tokens, "olmo", elapsed)
            log_exchange(iteration, "step1_reason_plan", plan_prompt, plan_text,
                         tokens, elapsed=elapsed,
                         think_content=tokens.get("think_content"))

            # Handle truncated think: retry with think=False
            if tokens.get("was_truncated") and not plan_text.strip():
                print("  [Retry] Think truncated — retrying with think=False")
                t0 = time.time()
                plan_text, tokens, _ = ask_text_olmo(
                    manager, OLMO_SYSTEM_PROMPT, plan_prompt,
                    max_new_tokens=config["olmo_max_new_tokens_plan"],
                    sampling=config["olmo_sampling_structured"],
                    think=False,
                )
                elapsed = time.time() - t0
                track(iteration, "reason_plan_retry", tokens, "olmo", elapsed)
                log_exchange(iteration, "step1_reason_plan_retry", plan_prompt,
                             plan_text, tokens, elapsed=elapsed)

            print(f"  Plan: {plan_text[:300]}...")

            # Check if OLMo decided to answer in step 1
            min_answer_iter = min(min_iter, max_iter)
            embedded = parse_action(plan_text)
            if embedded and embedded.get("action") == "answer":
                if iteration < min_answer_iter:
                    print(f"  [BLOCKED] OLMo tried to answer on iteration "
                          f"{iteration} (min={min_answer_iter}). Continuing.")
                else:
                    final_answer = embedded.get("answer", plan_text)
                    print(f"\n  [ANSWER from step 1] {final_answer}")
                    history.append({
                        "iteration": iteration,
                        "action_data": {"action": "answer",
                                        "answer": final_answer},
                        "finding": final_answer,
                        "fov_feedback": "",
                    })
                    break

            # ── Step 2: OLMo action + prompt (think=False) ────────
            print("\n  [Step 2] OLMo: Action decision ...")
            action_prompt = build_action_prompt(plan_text, volume_info, config)
            t0 = time.time()
            action_text, tokens, _ = ask_text_olmo(
                manager, OLMO_SYSTEM_PROMPT, action_prompt,
                max_new_tokens=config["olmo_max_new_tokens_decision"],
                sampling=config["olmo_sampling_structured"],
                think=False,
            )
            elapsed = time.time() - t0
            track(iteration, "action", tokens, "olmo", elapsed)
            log_exchange(iteration, "step2_action", action_prompt, action_text,
                         tokens, elapsed=elapsed,
                         think_content=tokens.get("think_content"))
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
                    think=False,
                )
                elapsed = time.time() - t0
                track(iteration, "action_retry", tokens, "olmo", elapsed)
                log_exchange(iteration, "step2_retry", retry_prompt, action_text,
                             tokens, elapsed=elapsed,
                             think_content=tokens.get("think_content"))
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

            # Extract prompt for vision model from action JSON
            olmo_instructions = action.pop("prompt", "") or action.pop("vision_prompt", "")
            target_refinement = action.pop("target_refinement", "")

            # ── Short-circuit: answer ────────────────────────────────
            if action_type == "answer":
                if iteration < min_answer_iter:
                    print(f"  [BLOCKED] OLMo tried to answer on iteration "
                          f"{iteration} (min={min_answer_iter}). "
                          f"Forcing reason instead.")
                    action = {
                        "action": "reason",
                        "question": (
                            "You tried to answer too early. What views, "
                            "orientations, or regions still need examination?"
                        ),
                    }
                    action_type = "reason"
                else:
                    final_answer = action.get("answer", "")
                    print(f"\n  [ANSWER] {final_answer}")
                    history.append({
                        "iteration": iteration,
                        "action_data": action,
                        "finding": final_answer,
                        "fov_feedback": "",
                    })
                    break

            # ── Short-circuit: reason (skip steps 3-4) ───────────────
            if action_type == "reason":
                reason_q = action.get("question", "Synthesize findings.")
                print("  [Short-circuit] reason — skipping steps 3-4")
                reason_prompt = build_reason_shortcircuit_prompt(
                    question, reason_q, findings_text,
                )
                t0 = time.time()
                finding, tokens, _ = ask_text_olmo(
                    manager, OLMO_SYSTEM_PROMPT, reason_prompt,
                    max_new_tokens=config["olmo_max_new_tokens_plan"],
                    sampling=config["olmo_sampling_structured"],
                )
                elapsed = time.time() - t0
                track(iteration, "reason_shortcircuit", tokens, "olmo", elapsed)
                log_exchange(iteration, "reason_shortcircuit", reason_prompt,
                             finding, tokens, elapsed=elapsed,
                             think_content=tokens.get("think_content"))
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

                last_finding = finding
                last_fov = ""
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
                    max_new_tokens=config["olmo_max_new_tokens_plan"],
                    sampling=config["olmo_sampling_structured"],
                )
                elapsed = time.time() - t0
                track(iteration, "forced_reason", tokens, "olmo", elapsed)
                log_exchange(iteration, "forced_reason", dup_prompt, finding,
                             tokens, elapsed=elapsed,
                             think_content=tokens.get("think_content"))
                print(f"  Forced reasoning: {finding[:200]}...")

                last_finding = finding
                last_fov = ""
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

            # ── Steps 3-4: Molmo2 capture + interpret ────────────────
            manager.swap_to_molmo()
            finding = ""
            fov_feedback = ""

            if action_type == "screenshot":
                # Step 3: Capture screenshot
                view_spec = action.get("view", {})
                screenshot_count += 1
                state = build_clean_state(base_state, view_spec, volume_info)
                img = capture_screenshot(page, state, screenshot_count)

                # Step 4: Molmo2 interprets
                interpret_prompt = build_molmo_screenshot_prompt(
                    olmo_instructions, question, action, volume_info,
                )
                print(f"  [Step 4] Molmo2: Interpreting screenshot ...")
                t0 = time.time()
                finding, tokens = ask_vision(
                    manager.molmo_model, manager.molmo_processor,
                    img, interpret_prompt, max_new_tokens=1024,
                )
                elapsed = time.time() - t0
                track(iteration, "interpret_screenshot", tokens, "molmo", elapsed)
                log_exchange(iteration, "step4_interpret", interpret_prompt,
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
                # Step 3: Capture scan
                scan_count += 1
                geo_fp = geometry_fingerprint(action)
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

                # Step 4: Molmo2 interprets scan
                interpret_prompt = build_molmo_scan_prompt(
                    olmo_instructions, question, action, volume_info,
                    len(frames), frame_spacing, total_dist,
                )
                print(f"  [Step 4] Molmo2: Interpreting scan "
                      f"({len(frames)} frames) ...")
                t0 = time.time()
                finding, tokens = ask_scan(
                    manager.molmo_model, manager.molmo_processor,
                    frames, interpret_prompt,
                    max_new_tokens=1024, config=config,
                )
                elapsed = time.time() - t0
                track(iteration, "interpret_scan", tokens, "molmo", elapsed)
                log_exchange(iteration, "step4_interpret_scan", interpret_prompt,
                             finding, tokens,
                             f"video: scan_{scan_count:03d}.mp4, "
                             f"{len(frames)} frames", elapsed)
                print(f"  Finding: {finding[:200]}...")

            elif action_type == "count":
                # Step 3: Capture + per-keyframe pointing
                scan_count += 1
                target = action.get("target", "objects")

                geo_fp = geometry_fingerprint(action)
                if geo_fp in frame_cache:
                    frames = frame_cache[geo_fp]
                    print(f"  [Cache hit] Reusing {len(frames)} frames")
                    save_scan_video(frames, scan_count)
                else:
                    frames = execute_scan(base_state, action, volume_info,
                                          config, scan_count)
                    if geo_fp:
                        frame_cache[geo_fp] = frames

                # Step 4: Per-keyframe pointing
                keyframe_interval = max(1, int(
                    action.get("keyframe_interval", 5)))
                keyframe_indices = list(range(0, len(frames),
                                              keyframe_interval))
                print(f"  [Step 4] Pointing to '{target}' on "
                      f"{len(keyframe_indices)} keyframes ...")

                # Build pointing prompt: always start with "Point to each"
                refinement = target_refinement.strip()
                if refinement:
                    point_prompt = f"Point to each {target}. {refinement}"
                else:
                    point_prompt = f"Point to each {target}."

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

                # Build pointing stats for next iteration's reasoning
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
                    f"(from {len(frames)} total frames).\n"
                    f"{pointing_stats}"
                )

            # ── Carry finding to next iteration's step 1 ────────────
            last_finding = finding
            last_fov = fov_feedback

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
            f"Synthesize ALL findings into a comprehensive answer.\n"
            f"Be specific and cite evidence from your observations.\n\n"
            f"Weighting: Direct visual observations (what was seen in screenshots and scans)\n"
            f"are primary evidence. Quantitative metrics (counts, distributions) are supporting\n"
            f"context only \u2014 they describe what is present but do not by themselves indicate\n"
            f"whether something is correct or incorrect. If counts vary across the volume,\n"
            f"consider whether this reflects natural biological variation before attributing\n"
            f"it to errors."
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
                     answer_text, tokens, elapsed=elapsed,
                     think_content=tokens.get("think_content"))

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
    md = []
    md.append("# molmo-glancer v4 — Prompt Templates\n")
    md.append(f"Generated for question: *{question}*\n")
    md.append(f"Volume: {volume_info.format_for_prompt()}\n")

    # OLMo system prompt
    md.append("---\n")
    md.append("## OLMo System Prompt\n")
    md.append("Sent as system role in every OLMo call (steps 1, 2).\n")
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
        f"Describe what you see, focusing on what's relevant to the question:\n"
        f"\"{question}\""
    )
    md.append("```\n")

    # Step 1: Reason + Plan (OLMo, think=True)
    md.append("---\n")
    md.append("## Step 1: Reason + Plan (OLMo, think=True)\n")
    md.append(f"Token budget: {config['olmo_max_new_tokens_plan']}\n")
    md.append("### Iteration 1 variant (plan only):\n```")
    md.append(
        'You have examined a 3D volume and received this initial description:\n'
        '"{first_look_finding}"\n\n'
        f'QUESTION: "{question}"\n\n'
        f'VOLUME:\n{volume_info.format_for_prompt()}\n\n'
        'Plan your investigation strategy. What should you look at first, and why?'
    )
    md.append("```\n### Iteration N variant (reason + plan):\n```")
    md.append(
        f'QUESTION: "{question}"\n\n'
        'LATEST FINDING (iteration N-1):\n{last_finding}\n\n'
        'INVESTIGATION SO FAR:\n{findings_text}\n\n'
        'Analyze the latest finding in context, then decide what to investigate next.\n\n'
        'EVIDENCE CHECK — before concluding, ask yourself:\n'
        '- Does my evidence DIRECTLY address the question, or is it only indirect/proxy?\n'
        '  Counts and distributions describe what is present — they do NOT assess quality,\n'
        '  accuracy, alignment, or correctness. Those require visual comparison.\n'
        '- Could the pattern I see have a simpler explanation?\n'
        '  Variation in measurements across spatial positions usually reflects natural\n'
        '  biological variation (e.g. cell density differs by depth), not errors.\n'
        '  Do not interpret measurement variation as evidence of a problem without\n'
        '  visual confirmation at the specific locations in question.\n'
        '- Am I answering based on what I SAW, or on assumptions about what numbers MEAN?\n\n'
        'If you have gathered enough DIRECT visual evidence to answer confidently:\n'
        '  {"action": "answer", "answer": "your specific answer here"}\n'
        'Otherwise, describe your next investigation step IN PROSE (not JSON).\n'
        'Explain what to look at, where, and why.'
    )
    md.append("```\n")

    # Step 2: Action + Vision Prompt (OLMo, think=False)
    md.append("---\n")
    md.append("## Step 2: Action + Vision Prompt (OLMo, think=False)\n")
    md.append(f"Token budget: {config['olmo_max_new_tokens_decision']}\n")
    md.append("### Action Schema:\n```")
    md.append(build_action_schema(volume_info, config["max_scan_frames"]))
    md.append("```\n### Wrapper:\n```")
    md.append(
        'YOUR INVESTIGATION PLAN:\n{plan_text}\n\n'
        '{action_schema}\n\n'
        'Translate the investigation plan above into exactly one JSON action object.\n'
        'Your action MUST match the intent of the plan — do not substitute a different\n'
        'action type than what the plan describes.\n\n'
        'Include "purpose" and "prompt" for visual actions,\n'
        'and "target_refinement" for count actions.\n\n'
        'Your "prompt" will be sent directly to the vision model along with the\n'
        'captured image or video. Write it to elicit the observation you need.'
    )
    md.append("```\n")

    # Steps 3-4: Molmo2 Interpretation
    md.append("---\n")
    md.append("## Step 4: Screenshot Interpret (Molmo2, image + text)\n")
    md.append("```")
    md.append(
        '{prompt from action JSON}\n\n---\n\n'
        f'Question: "{question}"\n\n'
        'This is a {{layout}} view at position (x, y, z), zoom={{zoom}}.\n'
        f'{volume_info.format_for_prompt()}'
    )
    md.append("```\n")

    md.append("---\n")
    md.append("## Step 4: Scan Interpret (Molmo2, video + text)\n")
    md.append("```")
    md.append(
        '{prompt from action JSON}\n\n---\n\n'
        f'Question: "{question}"\n\n'
        'Scan: {{num_frames}} frames along {{axis}}, ~{{spacing}}\u00b5m between frames.'
    )
    md.append("```\n")

    md.append("---\n")
    md.append("## Step 4: Count Pointing (Molmo2, image + text)\n")
    md.append("target_refinement from action JSON appended after 'Point to each {target}.'\n")
    md.append("```")
    md.append("Point to each {target}. {target_refinement}")
    md.append("```\n")

    # Forced Answer
    md.append("---\n")
    md.append("## Forced Answer / Synthesis (OLMo, think=True)\n")
    md.append(f"Token budget: {config['olmo_max_new_tokens_synthesis']}\n")
    md.append("```")
    md.append(
        f'QUESTION: "{question}"\n\n'
        'INVESTIGATION COMPLETE \u2014 ALL FINDINGS:\n{findings_text}\n\n'
        'Synthesize ALL findings into a comprehensive answer.\n'
        'Be specific and cite evidence from your observations.\n\n'
        'Weighting: Direct visual observations (what was seen in screenshots and scans)\n'
        'are primary evidence. Quantitative metrics (counts, distributions) are supporting\n'
        'context only — they describe what is present but do not by themselves indicate\n'
        'whether something is correct or incorrect. If counts vary across the volume,\n'
        'consider whether this reflects natural biological variation before attributing\n'
        'it to errors.'
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

    presets = discover_presets()
    preset_names = sorted(presets.keys())

    parser = argparse.ArgumentParser(description="molmo-glancer")
    parser.add_argument("--preset", choices=preset_names,
                        help="Named run preset (overrides NG_LINK_FILE and QUESTION env vars)")
    args = parser.parse_args()

    # Resolve inputs: --preset > env vars > defaults
    if args.preset:
        p = presets[args.preset]
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
