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
                      iteration, max_iter, last_finding=None, last_fov=None,
                      min_iter=3):
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
            f"Plan your investigation strategy. What should you look at first, and why?\n\n"
            f"Consider:\n"
            f"- What spatial regions need examination to answer the question?\n"
            f"- Would a different layout (xy vs xz vs yz) reveal different information?\n"
            f"- Would a scan (video sweep) show spatial distribution better than a static view?\n"
            f"- Would toggling layer visibility reveal alignment, segmentation quality, etc.?\n"
            f"- Is the question quantitative (need count action) or qualitative (scan/screenshot)?"
        )

    # Iteration 2+: reason about last finding, then plan
    finding_block = ""
    if last_finding:
        finding_block = f"LATEST FINDING (iteration {iteration - 1}):\n{last_finding}\n"
        if last_fov:
            finding_block += f"{last_fov}\n"
        finding_block += "\n"

    min_answer_iter = min(min_iter, max_iter)
    if iteration < min_answer_iter:
        answer_block = (
            f"You are on iteration {iteration}/{max_iter} — it is TOO EARLY to answer.\n"
            f"You need to examine more views and orientations first.\n"
        )
    else:
        answer_block = (
            f"ONLY if you have examined multiple views/orientations and have enough\n"
            f"evidence to give a final answer, respond instead with:\n"
            f'{{\"action\": \"answer\", \"answer\": \"your specific answer here\"}}\n'
        )

    return (
        f"QUESTION: \"{question}\"\n\n"
        f"{finding_block}"
        f"INVESTIGATION SO FAR:\n\n{findings_text}\n\n"
        f"Iteration {iteration}/{max_iter}.\n\n"
        f"PART 1 — REASONING: Analyze the latest finding in context:\n"
        f"- Does it confirm, contradict, or extend previous findings?\n"
        f"- What spatial regions remain unexplored?\n"
        f"- Do you have sufficient evidence to answer the question?\n\n"
        f"PART 2 — PLAN: Based on your reasoning, what should you investigate next?\n"
        f"Consider what spatial regions remain unexplored, whether findings are\n"
        f"consistent, and whether you have enough evidence to answer.\n\n"
        f"{answer_block}"
    )


def build_action_prompt(plan_text, volume_info, config):
    """Step 2: OLMo strict JSON action + vision prompt from schema."""
    schema = build_action_schema(volume_info, config["max_scan_frames"])
    return (
        f"YOUR INVESTIGATION PLAN:\n{plan_text}\n\n"
        f"{schema}\n\n"
        f"Output ONLY the JSON action object that executes your plan — no other text.\n"
        f"Include a \"purpose\" field explaining what you expect to learn.\n\n"
        f"VISION PROMPT: For screenshot, scan, and count actions, include a\n"
        f"\"vision_prompt\" field (2-4 sentences) telling the vision model:\n"
        f"- What specific features or structures to focus on\n"
        f"- What to compare against prior findings (if any)\n"
        f"- Any artifacts or confounds to watch for\n"
        f"For count actions, include a \"target_refinement\" field (1-2 sentences)\n"
        f"to help the vision model identify the right objects.\n\n"
        f"If you already have enough evidence, use the answer action instead."
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
        f"({start.get('x',0):.0f},{start.get('y',0):.0f},{start.get('z',0):.0f}) \u2192 "
        f"({end.get('x',0):.0f},{end.get('y',0):.0f},{end.get('z',0):.0f})\n"
        f"Frame spacing: ~{frame_spacing:.1f}\u00b5m, total distance: {total_dist:.0f}\u00b5m\n"
        f"Layout: {layout}, zoom: {zoom}\n\n"
        f"Describe what you observe across the frames:\n"
        f"- How does the content change along the scan axis?\n"
        f"- Where are structures most dense vs sparse?\n"
        f"- Are there boundaries, transitions, or abrupt changes?\n"
        f"- Estimate the spatial extent of notable features"
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
