"""
prompts — Prompt construction for OLMo and Molmo2.

Builds all prompts for the 4-step agent loop:
  Step 1: reason + plan (OLMo)
  Step 2: action JSON + vision prompt (OLMo)
  Step 4: screenshot/scan interpretation (Molmo2)
  + reason short-circuit, action schema, findings formatting.
"""

from volume_info import VolumeInfo, format_zoom_table


# ── Action Schema ────────────────────────────────────────────────────────


def build_action_schema(volume_info: VolumeInfo, max_scan_frames: int = 50) -> str:
    """Build the action schema with volume-appropriate example coordinates."""
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

2. scan — sweep through the volume as a video along one axis.
   Produces a continuous sequence of frames that reveals the 3D
   structure of the data — how features appear, persist, and change
   through depth. Screenshots show single isolated slices; only a
   scan shows the volume as a coherent whole.
   {{"action": "scan", "scan_type": "z_sweep",
    "start": {{"x": {cx:.0f}, "y": {cy:.0f}, "z": 0}},
    "end":   {{"x": {cx:.0f}, "y": {cy:.0f}, "z": {zmax:.0f}}},
    "frames": {max_scan_frames}, "layout": "xy", "zoom": "full",
    "prompt": "<what specifically to look for across this sweep>"}}
   scan_type options: z_sweep, x_pan, y_pan
   scan is for qualitative description. Use count for numerical results.

3. count — DETECT + COUNT specific objects via automated pointing on sampled keyframes
   {{"action": "count", "scan_type": "z_sweep",
    "start": {{"x": {cx:.0f}, "y": {cy:.0f}, "z": 0}},
    "end":   {{"x": {cx:.0f}, "y": {cy:.0f}, "z": {zmax:.0f}}},
    "frames": {max_scan_frames}, "layout": "xy", "zoom": "full",
    "target": "neurons", "keyframe_interval": 5}}
   The system automatically detects and marks each instance in sampled keyframes —
   you get back exact per-frame counts.
   Use count ONLY when the question asks "how many?" or you need an actual inventory.
   Count answers quantity — it does NOT assess quality, accuracy, or correctness
   of what it detects. Do not use count as a proxy for evaluating whether something
   is good or bad — use screenshot or scan for visual comparison instead.
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

OPTIONAL KEYS (for screenshot, scan, and count):
  "show": [1, 2]  — which layers to show (by number from the Layers list above). Omit to keep current visibility.
  "shaderRange": [vmin, vmax]  — adjust brightness/contrast for image layers

IMPORTANT: Zooms below "full" CROP the view — you will miss data outside the visible area.

{format_zoom_table()}
"""


# ── Structured Findings ──────────────────────────────────────────────────


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


# ── OLMo Prompt Builders ─────────────────────────────────────────────────


def build_plan_prompt(question, volume_info, first_look_finding, findings_text,
                      iteration, last_finding=None, last_fov=None):
    """Step 1: OLMo reason over last finding + plan next action.

    Iteration 1: plan only (no prior finding to reason over).
    Iteration 2+: reason about last finding, then plan next step.
    """
    if iteration == 1:
        return (
            f"You have examined a 3D volume and received this initial description:\n"
            f"\"{first_look_finding}\"\n\n"
            f"QUESTION: \"{question}\"\n\n"
            f"VOLUME:\n{volume_info.format_for_prompt()}\n\n"
            f"Plan your investigation strategy. What should you look at first, and why?"
        )

    # Iteration 2+: reason about last finding, then plan
    finding_block = ""
    if last_finding:
        finding_block = f"LATEST FINDING (iteration {iteration - 1}):\n{last_finding}\n"
        if last_fov:
            finding_block += f"{last_fov}\n"
        finding_block += "\n"

    return (
        f"QUESTION: \"{question}\"\n\n"
        f"{finding_block}"
        f"INVESTIGATION SO FAR:\n\n{findings_text}\n\n"
        f"Analyze the latest finding in context, then decide what to investigate next.\n\n"
        f"EVIDENCE CHECK — before concluding, ask yourself:\n"
        f"- Does my evidence DIRECTLY address the question, or is it only indirect/proxy?\n"
        f"  Counts and distributions describe what is present — they do NOT assess quality,\n"
        f"  accuracy, alignment, or correctness. Those require visual comparison.\n"
        f"- Could the pattern I see have a simpler explanation?\n"
        f"  Variation in measurements across spatial positions usually reflects natural\n"
        f"  biological variation (e.g. cell density differs by depth), not errors.\n"
        f"  Do not interpret measurement variation as evidence of a problem without\n"
        f"  visual confirmation at the specific locations in question.\n"
        f"- Am I answering based on what I SAW, or on assumptions about what numbers MEAN?\n\n"
        f"If you have gathered enough DIRECT visual evidence to answer confidently:\n"
        f'  {{\"action\": \"answer\", \"answer\": \"your specific answer here\"}}\n'
        f"Otherwise, describe your next investigation step IN PROSE (not JSON).\n"
        f"Explain what to look at, where, and why."
    )


def build_action_prompt(plan_text, volume_info, config):
    """Step 2: OLMo strict JSON action + vision prompt from schema."""
    schema = build_action_schema(volume_info, config["max_scan_frames"])
    return (
        f"YOUR INVESTIGATION PLAN:\n{plan_text}\n\n"
        f"{schema}\n\n"
        f"Translate the investigation plan above into exactly one JSON action object.\n"
        f"Your action MUST match the intent of the plan — do not substitute a different\n"
        f"action type than what the plan describes.\n\n"
        f"Include \"purpose\" and \"prompt\" for visual actions,\n"
        f"and \"target_refinement\" for count actions.\n\n"
        f"Your \"prompt\" will be sent directly to the vision model along with the\n"
        f"captured image or video. Write it to elicit the observation you need."
    )


# ── Molmo2 Prompt Builders ───────────────────────────────────────────────


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
        f"{volume_info.format_for_prompt()}"
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
        f"({start.get('x',0):.0f},{start.get('y',0):.0f},{start.get('z',0):.0f}) \u2192 "
        f"({end.get('x',0):.0f},{end.get('y',0):.0f},{end.get('z',0):.0f})\n"
        f"Frame spacing: ~{frame_spacing:.1f}\u00b5m, total distance: {total_dist:.0f}\u00b5m\n"
        f"Layout: {layout}, zoom: {zoom}"
    )


def build_reason_shortcircuit_prompt(question, reason_question, findings_text):
    """Reason action short-circuit (from step 2, skips steps 3-4)."""
    return (
        f"QUESTION: \"{question}\"\n\n"
        f"INVESTIGATION SO FAR:\n{findings_text}\n\n"
        f"Your reasoning request: \"{reason_question}\"\n\n"
        f"Analyze the evidence and decide your next step:\n"
        f"1. If evidence is sufficient, provide your answer.\n"
        f"2. If findings conflict, identify the contradiction.\n"
        f"3. If critical regions remain unexplored, describe what to investigate next."
    )
