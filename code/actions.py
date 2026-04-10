"""
actions — Action parsing, validation, and duplicate detection.

Handles:
- JSON extraction from model text output
- Layer visibility resolution (show numbers → layerVisibility dict)
- Zoom name resolution and coordinate clamping
- Geometric fingerprinting for frame cache and dedup
"""

import json
import re

from volume_info import VolumeInfo, resolve_zoom


# ── Action Parsing ───────────────────────────────────────────────────────


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


# ── Duplicate Detection & Frame Cache ────────────────────────────────────


def geometry_fingerprint(action: dict) -> str:
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
