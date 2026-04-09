# v4 Prompting Deep-Dive: OLMo 3.1 32B Think Conversation Design

## Executive Summary

v3 sends every prompt — planning, decisions, reasoning, interpretation — through the
same 7B model (Molmo2's OLMo3-7B backbone). The prompts are designed to compensate for
the small model's weaknesses: concrete example JSON (which the model parrots), limited
history truncation, and forced single-action-per-iteration flow.

v4 splits responsibilities: OLMo 3.1 32B Think handles all text reasoning (plan,
decide, synthesize), while Molmo2 handles all vision (interpret screenshots, scans,
pointing). This report designs the conversation structure, prompt templates, and action
schema for the 32B model.

---

## 1. Current v3 Problems

### 1.1 The 7B model parrots example JSON

`build_action_schema()` (line 462-524 of `molmo_glancer.py`) gives concrete coordinate
examples using the volume center:

```json
{"action": "screenshot",
 "view": {"x": 250, "y": 250, "z": 110, "layout": "xy", "zoom": "full"},
 "prompt": "<what specifically to look for in this view>"}
```

The 7B model frequently copies these exact coordinates and layouts instead of reasoning
about what view would actually answer the question. It also repeats the placeholder
prompt text verbatim.

### 1.2 Single-action-per-iteration is wasteful

Each iteration plans one action, captures one view, interprets it, and then decides the
next. With model swapping this becomes 2 swap cycles per view — ~40-70s of overhead per
view. The model also has no way to request "compare these two things side by side" because
it can only see one view at a time.

### 1.3 Limited action vocabulary

The current actions are: `screenshot`, `scan`, `count`, `reason`, `answer`. Missing:
- No explicit way to change layer visibility without taking a screenshot
- No way to adjust shader/contrast for investigation
- No way to request a specific 4panel view vs a single-panel view with purpose
- Scan types are limited to `z_sweep`, `x_pan`, `y_pan` — no diagonal or custom path

### 1.4 History format is flat and lossy

The `format_history_entry()` function (line 591-618) encodes findings as flat strings.
Older entries get compressed to one-line summaries. The model has no structured way to
reference prior findings ("finding #3 showed neurons in the z=200 region...").

### 1.5 No think-token exploitation

The 7B model doesn't support structured reasoning. OLMo 3.1 32B Think produces
`<think>...</think>` blocks before answering — we should design prompts that
encourage deep chains for planning and synthesis.

---

## 2. v4 Conversation Architecture

### 2.1 Two-model conversation flow

```
┌─────────────── Iteration N ───────────────────────────────────┐
│                                                                │
│  ┌─ OLMo 3.1 32B Think (text) ─────────────────────────────┐  │
│  │                                                           │  │
│  │  SYSTEM: Role + capabilities + volume info                │  │
│  │  USER: Question + accumulated findings + action schema    │  │
│  │  ASSISTANT: <think>deep reasoning</think>                 │  │
│  │             {actions: [...], reasoning: "..."}             │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│              ↓ (planned actions)                                │
│  ┌─ Molmo2-O-7B (vision) ──────────────────────────────────┐  │
│  │                                                           │  │
│  │  For each planned action:                                 │  │
│  │    capture screenshot/scan → interpret with image prompt   │  │
│  │    → finding_N                                            │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│              ↓ (findings)                                       │
│  ┌─ OLMo 3.1 32B Think (text) ─────────────────────────────┐  │
│  │                                                           │  │
│  │  Reason over all findings from this iteration             │  │
│  │  Decide: plan more views? or answer?                      │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 Prompt routing

| Prompt type | Model | Why |
|---|---|---|
| First look interpret | Molmo2 | Needs vision |
| Strategy planning | OLMo 32B | Pure reasoning, benefits from <think> |
| Action decision | OLMo 32B | Complex spatial reasoning about what to look at next |
| Screenshot interpret | Molmo2 | Needs vision |
| Scan interpret | Molmo2 | Needs vision (video frames) |
| Count pointing | Molmo2 | Needs vision (per-keyframe) |
| Count interpret | OLMo 32B | Pure text, benefits from reasoning about statistics |
| Reason step | OLMo 32B | Pure reasoning, synthesis |
| Final answer | OLMo 32B | Quality-critical synthesis |
| Forced answer | OLMo 32B | Last-resort synthesis |

---

## 3. OLMo System Prompt Design

### 3.1 Exploiting `<think>` tokens

OLMo 3.1 32B Think emits `<think>...</think>` blocks before producing the actual
answer. These are "free" reasoning — they improve output quality at the cost of
generation time, not quality. The system prompt should explicitly encourage this.

**Design principle:** Do NOT tell the model to think step-by-step (it already does).
Instead, tell it what to reason ABOUT — that directs the thinking toward useful analysis.

### 3.2 System prompt template

```
SYSTEM PROMPT (sent once per OLMo call — OLMo is stateless, no multi-turn):

You are the reasoning engine of an autonomous visual analysis system.
You analyze 3D volumetric microscopy data loaded in Neuroglancer.

Your partner is a vision model (Molmo2) that captures and interprets
screenshots and video scans of the data. You cannot see images directly.
You plan what views to capture, and Molmo2 reports back what it sees.

Your job is to:
1. Plan efficient sequences of views that answer the question
2. Reason about findings — resolve contradictions, identify gaps, estimate quantities
3. Decide when you have enough evidence to answer confidently

Think carefully before acting. Consider:
- What spatial regions remain unexplored?
- Are findings consistent across views? If not, why?
- What view would most efficiently resolve remaining uncertainty?
- Are quantitative estimates grounded in actual detections, or guesses?
```

**Key differences from v3:**
- Explicitly describes the two-model partnership (the model knows it can't see)
- Frames the task as *planning views for another model to interpret*
- Lists reasoning priorities rather than prescribing step-by-step process
- No examples of JSON yet — that comes in the action schema section

---

## 4. Action Schema Redesign

### 4.1 Design principles

1. **Schematic placeholders (`<>`) not concrete examples** — The 32B model understands
   schema; it doesn't need to see `"x": 250` to know what a coordinate is. Concrete
   numbers cause parroting; descriptive placeholders force the model to reason about
   what values to use.

2. **Action batching** — The model returns an `actions` array, not a single action.
   This is the key architectural change that amortizes swaps.

3. **Expanded action vocabulary** — More fine-grained control over the Neuroglancer
   state, reflecting the full `build_clean_state()` API surface.

4. **Purpose-driven prompts** — Each action carries a `purpose` field explaining
   WHY this view is needed, not just what to look for.

### 4.2 New action schema

```
ACTION SCHEMA:

Respond with a JSON object. For visual actions, return an "actions" array
(1 to {max_actions_per_plan} actions). For terminal actions (answer, reason),
return a single action.

──── Visual Actions (batched) ────────────────────────────────────

screenshot — capture a 2D cross-section
  {"action": "screenshot",
   "view": {
     "x": <x_coordinate>, "y": <y_coordinate>, "z": <z_coordinate>,
     "layout": "<xy|xz|yz|4panel>",
     "zoom": "<wide|full|region|close-up|single-cell>"
   },
   "purpose": "<why this specific view answers the question>"}

  Optional view keys:
    "show": [<layer_numbers>]       — which layers to show (see layer list)
    "shaderRange": [<vmin>, <vmax>] — adjust brightness/contrast for image layers

scan — sweep through data as video (qualitative description)
  {"action": "scan",
   "scan_type": "<z_sweep|x_pan|y_pan>",
   "start": {"x": <x>, "y": <y>, "z": <z>},
   "end":   {"x": <x>, "y": <y>, "z": <z>},
   "frames": <num_frames>,
   "layout": "<xy|xz|yz|4panel>",
   "zoom": "<zoom_level>",
   "purpose": "<what spatial pattern or structure to look for>"}

  Use scan for QUALITATIVE observation — understanding structure, distribution,
  spatial extent. Do NOT use scan for numerical counts.

  Optional keys:
    "show": [<layer_numbers>]
    "shaderRange": [<vmin>, <vmax>]

count — automated object detection via pointing on sampled keyframes
  {"action": "count",
   "scan_type": "<z_sweep|x_pan|y_pan>",
   "start": {"x": <x>, "y": <y>, "z": <z>},
   "end":   {"x": <x>, "y": <y>, "z": <z>},
   "frames": <num_frames>,
   "layout": "<xy|xz|yz|4panel>",
   "zoom": "<zoom_level>",
   "target": "<short_noun_for_objects_to_count>",
   "keyframe_interval": <N>,
   "purpose": "<why counting here, what region, what density expected>"}

  The vision model automatically detects and marks each instance on sampled
  keyframes. You receive exact per-frame counts — these are grounded detections,
  more reliable than visual estimates.

  keyframe_interval: spacing between sampled frames.
    Dense objects or small region → 2-3
    Sparse objects or large region → 5-10

──── Terminal Actions (single) ───────────────────────────────────

reason — synthesize findings before committing to a visual action or answer
  {"action": "reason",
   "question": "<specific question to reason about>"}

  Use reason to:
  - Reconcile conflicting findings
  - Estimate quantities from detection data
  - Plan a refined investigation strategy
  - Decide if you have sufficient evidence to answer

answer — final answer (ends the session)
  {"action": "answer",
   "answer": "<your specific answer to the question>",
   "confidence": "<high|medium|low>",
   "evidence_summary": "<brief list of key evidence supporting the answer>"}

──── Batched Response Format ─────────────────────────────────────

For visual actions, wrap in an actions array:
  {"actions": [
     <action_1>,
     <action_2>,
     ...
   ],
   "reasoning": "<why these specific views, what you expect to learn>"}

For reason/answer, return the action directly (no array).

──── Constraints ─────────────────────────────────────────────────

LAYOUT: "xy", "xz", "yz", "4panel"
  Do NOT use "3d" — it renders only a wireframe bounding box for raw
  image data, not actual voxel data.

ZOOM: Zooms below "full" CROP the view — data outside the visible area
  is excluded. Only zoom in when you need fine detail in a specific region.

  {zoom_table}

MAX FRAMES: {max_scan_frames} per scan/count action.
```

### 4.3 What changed from v3 → v4

| Aspect | v3 | v4 |
|---|---|---|
| Example coordinates | Concrete volume center (`"x": 250, "y": 250`) | Schematic `<x_coordinate>` |
| Example prompt text | `"<what specifically to look for>"` (parroted verbatim) | `"purpose"` key with descriptive placeholder |
| Response format | Single `{"action": ...}` | `{"actions": [...], "reasoning": "..."}` for visual; single for terminal |
| Prompt field | `"prompt"` (free-text, often copy-pasted) | `"purpose"` (intent-driven, explained) |
| `answer` action | `{"action": "answer", "answer": "..."}` | Adds `confidence` and `evidence_summary` |
| Layer control | `"show": [1, 2]` only | `"show"` + `"shaderRange"` documented at action level |
| Model self-description | "You are a visual data analyst" | "You are the reasoning engine; Molmo2 is your vision partner" |

### 4.4 Why `purpose` instead of `prompt`

In v3, the `"prompt"` field served double duty:
1. Instruction to the vision model about what to look for in the image
2. Record of the agent's intent for later reasoning

In v4, these are split:
- **`purpose`** (written by OLMo) — why this view is needed, what you expect to learn.
  Stored in history for OLMo to reference later.
- **Interpret prompt** (generated by the system) — the actual prompt sent to Molmo2
  with the captured image. Constructed programmatically from the purpose + question +
  spatial context.

This separation prevents the 32B model from wasting tokens trying to craft a
"good vision prompt" — it just states its intent, and the system handles the rest.

---

## 5. Multi-View Planning Prompt

### 5.1 Planning prompt template (replaces Phase 2 plan in v3)

```
You have examined a 3D volume and received this initial description:
"{first_look_finding}"

QUESTION: "{question}"

VOLUME:
{volume_info}

Based on this initial view, plan your investigation strategy.
Return an actions array with 1-{max_actions_per_plan} views to capture.

Consider:
- What spatial regions need examination to answer the question?
- Would different layouts (xy vs xz vs yz) reveal different information?
- Would a scan (video sweep) show spatial distribution better than static views?
- Would toggling layer visibility reveal alignment, segmentation quality, etc.?
- Is the question quantitative (need count action) or qualitative (scan/screenshot)?

Do NOT plan all views at once — plan the most informative first batch.
You will see the results and can plan more views afterward.
```

### 5.2 Decision prompt template (iteration N, replaces `build_decision_prompt`)

```
QUESTION: "{question}"

VOLUME:
{volume_info}

{action_schema}

INVESTIGATION SO FAR:

{structured_findings}

Iteration {N}/{max}. Plan your next actions.

If you have enough evidence to answer the question confidently, use the
answer action. If findings are contradictory or incomplete, plan views
that would resolve the uncertainty.
```

### 5.3 Structured findings format

Replace the flat `format_history_entry()` with structured blocks that OLMo can
reference by iteration number:

```
── Iteration 0 (first look) ──────────────────────────────────
  View: screenshot, 4panel, center, full zoom
  Finding: "Dense field of fluorescent neurons visible in xy plane.
    Approximately 20-30 bright cell bodies in this cross-section..."
  FOV: x=[0..497], y=[0..497]

── Iteration 1 (batch of 3 views) ────────────────────────────
  View 1a: screenshot, xy, z=50, full zoom
  Purpose: "Check neuron density in first quarter of volume"
  Finding: "Sparse — only 5-8 neurons visible at this depth..."
  FOV: x=[0..497], y=[0..497]

  View 1b: screenshot, xy, z=150, full zoom
  Purpose: "Check neuron density in middle of volume"
  Finding: "Dense cluster of ~25 neurons near center..."
  FOV: x=[0..497], y=[0..497]

  View 1c: scan, z_sweep, z=0..220, 50 frames
  Purpose: "Survey full z-depth for neuron distribution"
  Finding: "Neurons concentrated in z=80-180 range, sparse at edges..."

  Reasoning: "Most neurons are in the middle third of the volume (z=80-180).
  Need to count more carefully in that region."

── Iteration 2 (count action) ────────────────────────────────
  View 2a: count, z_sweep, z=80..180, target=neurons, keyframe_interval=3
  Purpose: "Get grounded count in high-density region"
  Finding: "DETECTED: 187 instances across 12/17 keyframes.
    Per keyframe: min=3, max=24, median=15."

  Reasoning: "187 raw detections with keyframe_interval=3 and ~5µm spacing
  means significant double-counting of neurons spanning multiple slices."
```

This format:
- Numbers iterations and sub-views for easy reference
- Preserves `purpose` alongside `finding` so OLMo can assess whether its investigation
  strategy is working
- Includes FOV context so OLMo knows what spatial region each view covered
- Includes the reasoning output from the previous OLMo pass

---

## 6. Reasoning Phase Prompt (post-findings, pre-decision)

After Molmo2 interprets all views from a batch, OLMo reasons over them:

```
QUESTION: "{question}"

You planned {N} views this iteration. Here are the results:

{new_findings_this_iteration}

Combined with your prior investigation:
{compressed_prior_findings}

Analyze these findings and decide your next step:

1. If the evidence is sufficient, provide your answer.
2. If findings conflict, identify the contradiction and plan views to resolve it.
3. If critical regions remain unexplored, plan the next batch of views.
4. If you need to perform quantitative analysis on existing data, use reason.

Respond with your next action (actions array for more views, or answer/reason).
```

---

## 7. Think Token Strategy

### 7.1 Token budgets by prompt type

Think models produce `<think>` blocks (often 1000-2500 tokens) BEFORE the actual
response. The budget must accommodate both. If `max_new_tokens` is too small, the
model exhausts its budget mid-think and never emits the JSON — a silent total failure.

| Prompt type | max_new_tokens | Think budget | Response budget | Rationale |
|---|---|---|---|---|
| Strategy planning | 4096 | ~1500-2000 | ~500-800 (JSON) | Plans 1-4 actions with reasoning |
| Action decision | 4096 | ~1500-2000 | ~500-800 (JSON) | Same as planning, iterative |
| Reasoning step | 6144 | ~2000-3000 | ~500-1000 (text) | Deeper chains for synthesis |
| Final answer | 8192 | ~2000-4000 | ~1000-2000 (text) | Full synthesis, longest think |
| Count interpret | 4096 | ~1000-2000 | ~500-800 (text) | Statistical reasoning |

**Why the old 2048/4096 budgets were too low:** A 2048 budget with 1500 think tokens
leaves only 548 tokens for the JSON actions array — barely enough for 2 actions,
and truncation mid-JSON is a parse failure requiring a retry (which wastes a full
swap cycle of ~40-70s).

### 7.2 Think token handling in code

```python
def strip_think_tokens(text: str) -> tuple[str, str]:
    """Extract and separate think blocks from OLMo output.

    Returns (clean_text, think_content).
    clean_text: the response with <think> blocks removed.
    think_content: concatenated thinking for transcript logging.
    """
    think_blocks = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return clean, '\n'.join(think_blocks)
```

**Logging:** Full output (including `<think>` blocks) goes to `transcript.md`.
Only `clean_text` is parsed for JSON actions and stored in history.

### 7.3 Prompting for good thinking

Do NOT add "think step by step" — OLMo Think already does this. Instead, include
domain-specific reasoning anchors in the system prompt:

```
When planning views, consider:
- Spatial coverage: what fraction of the volume has been examined?
- Resolution: are objects large enough to count at the current zoom?
- Layer interactions: do different layers need separate visibility states?
- Anisotropy: if z-resolution is coarser, xy views may show more detail

When synthesizing a count:
- Account for double-counting: objects spanning multiple z-slices appear
  in adjacent keyframes
- Keyframe spacing vs object size: if keyframe spacing < object diameter,
  expect ~2-3x overcounting from overlap
- Detection confidence: low-contrast or partial objects may be missed
```

These "reasoning anchors" guide the `<think>` chain toward domain-relevant analysis
without prescribing the exact reasoning steps.

---

## 8. Molmo2 Vision Prompts (unchanged model, refined prompts)

Molmo2 continues to handle all vision calls. These prompts are generated
programmatically from the action metadata — Molmo doesn't see the action schema.

### 8.1 Screenshot interpretation

```
Question: "{question}"

{purpose_from_olmo}

This is a {layout} view at position ({x}, {y}, {z}), zoom={zoom_name}.
Visible layers: {visible_layer_list}
Field of view: {fov_description}

Describe what you see. Report:
- What structures are present (type, shape, intensity)
- Approximate counts if objects are discrete and countable
- Spatial distribution (clustered, uniform, sparse/dense regions)
- Anything unusual or noteworthy
```

### 8.2 Scan interpretation

```
Question: "{question}"

{purpose_from_olmo}

Scan: {num_frames} frames along {axis}, {start} → {end}
Frame spacing: ~{spacing}µm, total distance: {total}µm
Layout: {layout}, zoom: {zoom_name}

Describe what you observe across the frames:
- How does the content change along the scan axis?
- Where are structures most dense vs sparse?
- Are there boundaries, transitions, or abrupt changes?
- Estimate the spatial extent of notable features
```

### 8.3 Count pointing (unchanged — Molmo2's pointing format)

```
Point to the {target}.
Each {target_singular} is approximately {neuron_pixels} pixels across.
```

---

## 9. Expanded State Manipulation Options

### 9.1 New capabilities to expose in the action schema

v3's `build_clean_state()` already supports these view_spec keys, but the action
schema doesn't expose them all to the model. v4 should document:

| Key | Type | What it does | When to use |
|---|---|---|---|
| `show` | `[int, ...]` | Layer visibility by number | Compare layers, isolate channels |
| `shaderRange` | `[vmin, vmax]` | Brightness/contrast | Enhance faint structures, reduce overexposure |
| `layout` | string | Panel configuration | `4panel` for overview, single-axis for detail |
| `zoom` | string | Named zoom level | `full` for survey, `region`/`close-up` for detail |
| `crossSectionOrientation` | `[x,y,z,w]` | Oblique slice quaternion | Custom cutting planes |

### 9.2 Actions NOT to add (keep it simple)

- **No `navigate` action** — position changes are implicit in each view's coordinates.
  Adding a separate "move" action would require an extra swap cycle.
- **No `annotate` action** — the system handles annotations internally (count results).
  Letting the model create arbitrary annotations risks state pollution.
- **No `configure_layer` action** — shader range changes are done per-view, not
  as a persistent state change. This avoids state management complexity.
- **No diagonal/custom-path scans** — the current `start`/`end` interpolation already
  supports arbitrary paths. Adding named diagonal types would complicate the schema
  without clear benefit.

### 9.3 Layer visibility patterns to suggest

For common multi-layer scenarios, the system prompt can include guidance:

```
LAYER COMPARISON STRATEGIES:
When the volume has multiple image layers (e.g., fixed + moving channels):
  - Show each layer alone to inspect channel quality
  - Show both overlaid to assess alignment/co-localization
  - Toggle layers across a scan to compare spatial patterns

When the volume has segmentation overlays:
  - Show image + segmentation together to assess boundary accuracy
  - Show segmentation alone to check for fragmentation
  - Show image alone to inspect raw data quality
```

---

## 10. Context Budget Management

### 10.1 OLMo 3.1 context limit: 32K tokens

The decision prompt must stay under 32K tokens. Budget allocation:

| Component | Est. tokens | Notes |
|---|---|---|
| System prompt | ~400 | Role, capabilities, reasoning anchors |
| Action schema | ~600 | Schema with placeholders (shorter than v3's concrete examples) |
| Volume info | ~200 | Dimensions, layers, zoom table |
| Current findings | ~200-500/iteration | Structured findings, 5 iterations max before compression |
| Question + iteration metadata | ~50 | Short |
| **Total at iteration 5** | **~3500-5000** | Well within 32K |
| **Reserve for generation** | ~4000 | Think tokens + JSON output |

### 10.2 History compression strategy

After 5 iterations with batched views, findings can accumulate. Strategy:

1. **Iterations 1-3:** Full findings with FOV, purpose, reasoning
2. **Iterations 4+:** Compress older iterations to one-line summaries:
   ```
   [Iter 1: 3 views — neurons concentrated z=80-180, sparse at edges]
   [Iter 2: count z=80-180 — 187 raw detections, estimated ~60-80 unique]
   ```
3. **Keep all count data verbatim** — quantitative results never compress
4. **Keep most recent 2 iterations full** — the model needs detailed context for
   its latest reasoning

### 10.3 Token counting

Add lightweight token estimation (4 chars ≈ 1 token) to enforce the budget:

```python
def estimate_tokens(text: str) -> int:
    """Rough token estimate for budget checking."""
    return len(text) // 4

def build_olmo_prompt(question, volume_info, history, config, iteration):
    """Build decision prompt with token budget enforcement."""
    budget = config.get("max_olmo_context_tokens", 32000)
    generation_reserve = 4096

    # Build full prompt, then compress history if over budget
    prompt = _build_full_prompt(question, volume_info, history, iteration)
    while estimate_tokens(prompt) > budget - generation_reserve and len(history) > 2:
        history = _compress_oldest(history)
        prompt = _build_full_prompt(question, volume_info, history, iteration)
    return prompt
```

---

## 11. No Compact Profile — L40S Only

v4 drops T4/compact profile support entirely (PLAN_v4.md D7). This simplifies
the prompting design:

- **One prompt path:** Schematic placeholders + batched actions. No fallback to
  concrete examples or single-action format.
- **One config:** No `max_actions_per_plan=1` special case. Always 4.
- **No image downscaling:** `max_image_side` is gone. Molmo always gets full-res images.
- **OLMo always available:** No "skip swaps, use Molmo for text" fallback. Every
  text call goes through OLMo 32B.
- **GPU rendering always on:** Chromium always launches with `--use-gl=egl`.

This means there is exactly one code path to test and debug.

---

## 12. Implementation Checklist

### New functions to add to `molmo_glancer.py`:

1. `strip_think_tokens(text)` — separate `<think>` blocks, detect truncation (§16.3)
2. `ask_text_olmo(manager, system_prompt, user_prompt, max_new_tokens, sampling)` — OLMo generation via ChatML system+user roles, think handling, truncation recovery (§16.1, §16.3)
3. `build_olmo_system_prompt(volume_info)` — system prompt with reasoning + spatial anchors (§3.2, §15.3, §7.3)
4. `build_olmo_decision_prompt(question, volume_info, history, config, iteration)` — replaces `build_decision_prompt` for OLMo calls
5. `build_vision_interpret_prompt(action, question, volume_info)` — generates Molmo2 interpret prompts with imaging context block (§15.6)
6. `format_structured_findings(history)` — replaces `format_history_entry` with richer format including FOV (§15.2)
7. `compress_findings(history, max_entries)` — token-aware history compression
8. `parse_action_batch(model_output)` — extends `parse_action` to handle `{"actions": [...]}`
9. `estimate_tokens(text)` — lightweight token counting for budget enforcement
10. `assess_think_confidence(think_content)` — mine think blocks for uncertainty signals (§16.5)

### New functions to add to `volume_info.py`:

11. `_infer_layer_role(layer)` — heuristic layer role from name/source (§15.5)
12. `pixel_to_physical(px, py, center, scale, layout)` — canvas→µm coords (§15.8)
13. `summarize_spatial_distribution(phys_points, volume_info)` — spatial summary for OLMo (§15.8)

### Config: single `CONFIG` dict in `gpu_config.py` (replaces `GPU_PROFILES`):

```python
CONFIG = {
    # Molmo2-O-7B (vision)
    "torch_dtype": torch.float16,
    "max_scan_frames": 50,
    "max_context_tokens": 55000,
    # OLMo 3.1 32B Think (text reasoning)
    "max_olmo_context_tokens": 32000,
    "max_actions_per_plan": 4,
    # OLMo generation budgets (§7.1 — includes think + response headroom)
    "olmo_max_new_tokens_decision": 4096,
    "olmo_max_new_tokens_synthesis": 8192,
    "olmo_max_new_tokens_retry": 2048,
    "olmo_max_new_tokens_hard_cap": 16384,
    # OLMo sampling presets (§16.2)
    "olmo_sampling_structured": {
        "temperature": 0.2, "top_p": 0.9,
        "repetition_penalty": 1.1, "do_sample": True,
    },
    "olmo_sampling_synthesis": {
        "temperature": 0.5, "top_p": 0.9,
        "repetition_penalty": 1.1, "do_sample": True,
    },
    "olmo_sampling_retry": {
        "temperature": 0.1, "top_p": 0.9,
        "repetition_penalty": 1.1, "do_sample": True,
    },
    # Agent loop
    "max_agent_iterations": 20,
}
```

### Prompt templates to update:

1. **Action schema** — replace concrete examples with schematic placeholders
2. **Decision prompt** — restructure for batched actions + structured findings
3. **System prompt** — new OLMo-specific system prompt (not sent to Molmo2)
4. **Reasoning prompt** — new post-findings reasoning prompt
5. **Interpret prompts** — add `purpose` context from OLMo's plan

---

## 13. Example: Full Iteration Walkthrough

### Setup
- Volume: 497x497x220µm, 2 image layers (fixed green, moving magenta), 4panel
- Question: "How well are neurons aligned between fixed and moving volumes?"

### Iteration 1 — OLMo plans

OLMo receives: first look finding + question + schema

OLMo thinks:
```
<think>
The question is about alignment quality between two channels. I need to:
1. See each channel individually to understand what's in each
2. See them overlaid to check for systematic shifts
3. Check multiple z-depths since alignment may vary through the volume

I should start with visibility toggling at the center, then scan through z.
</think>
```

OLMo responds:
```json
{"actions": [
  {"action": "screenshot",
   "view": {"x": 249, "y": 249, "z": 110, "layout": "xy", "zoom": "full"},
   "show": [1],
   "purpose": "View fixed channel alone at volume center to see reference neuron positions"},
  {"action": "screenshot",
   "view": {"x": 249, "y": 249, "z": 110, "layout": "xy", "zoom": "full"},
   "show": [2],
   "purpose": "View moving channel alone at same position to compare against fixed"},
  {"action": "screenshot",
   "view": {"x": 249, "y": 249, "z": 110, "layout": "xy", "zoom": "full"},
   "show": [1, 2],
   "purpose": "Overlay both channels to see alignment — green/magenta overlap = good alignment"}
],
 "reasoning": "Start with channel comparison at center. If well-aligned, green+magenta overlap produces white/bright regions. Misalignment shows as separated green and magenta spots."}
```

### System captures all 3 screenshots, Molmo2 interprets each

Finding 1a: "Single green channel showing ~15 bright neurons scattered across the field..."
Finding 1b: "Single magenta channel with similar neuron pattern, slightly shifted..."
Finding 1c: "Overlay shows mostly yellow/white overlap with 2-3 neurons showing green-magenta separation in the lower-left quadrant..."

### Iteration 1 — OLMo reasons over findings

OLMo receives all 3 findings and reasons:
```
<think>
Good alignment overall (most neurons overlap → white), but some separation in
lower-left. This could be a local registration error. I should check if this
misalignment is consistent across z-depths or just local. A z-scan with both
channels visible would show this.
</think>
```

Decides to plan more views targeting the misaligned region.

---

## 15. Metadata Enrichment — Layer-Aware Context and Spatial Coordinates

### 15.1 The core problem

Two models each have blind spots that better metadata can compensate for:

- **OLMo 32B** reasons about findings from prior views but doesn't know *where* each view
  was geographically — what 3D region it covered. Without this it can't track what's been
  explored or reason spatially across findings.
- **Molmo2** was not fine-tuned on neural microscopy data. It sees a dark Neuroglancer
  canvas with colored overlays and bright spots, with no training context for what any of
  it means. It needs explicit scaffolding to interpret the imaging modality, layer types,
  and common artifacts correctly.

---

### 15.2 Per-view FOV in every finding (coordinates spanning the view)

**Problem:** `format_fov_feedback()` already computes the visible window per view, but only
`print()`s it to the console. It is never stored in history, so OLMo never knows what
spatial region each view covered.

**Fix:** Store `fov_feedback` in every history entry, and include it in the structured
findings block (Section 5.3). This one change unlocks 3D spatial reasoning for OLMo.

Current state (history entry, line ~703):
```python
history.append({
    "iteration": ...,
    "action_data": {...},
    "finding": finding,
    "fov_feedback": "[user's original view — default zoom and position]",
})
```

The `fov_feedback` field exists but uses a static placeholder for first-look and is
set only for screenshot actions. It must be populated for all captured views.

**Required changes:**

1. **Screenshot:** after computing `fov_feedback`, store it in history entry.
2. **Scan:** store the swept axis range + visible window at start/end frame.
3. **First look:** compute `format_fov_feedback` from the actual center position and
   fit scale used, not the static string.

**Formatted output for OLMo findings block:**
```
── Iteration 1 ──────────────────────────────────────────────
  View 1a: screenshot, xy, z=110µm, zoom=full
  Purpose: "Check cell density at volume center"
  FOV: x=[0..497]µm, y=[0..497]µm  (slice at z=110µm)
  Finding: "Dense field of ~20-30 fluorescent cell bodies..."

  View 1b: scan, z_sweep, 50 frames
  Purpose: "Survey depth distribution of cells"
  Swept: z=[0..220]µm  (full depth), FOV per frame: x=[0..497]µm, y=[0..497]µm
  Finding: "Cells concentrated in z=[80..160]µm..."
```

For a 4panel view, report the shared position and all three visible cross-sections:
```
  FOV: position=(249, 249, 110)µm, showing xy@z=110, xz@y=249, yz@x=249 planes
```

For "close-up" or "region" zoom that crops the data, the window differs from the
full volume — this is when FOV reporting is most critical (OLMo must know it's only
seeing part of the data).

**Also fix the first look prompt:** line 690 says `"voxels"` but the values are in µm
(they come from `volume_info.shape` which is physical extent). Change to µm.

---

### 15.3 OLMo 3D spatial reasoning anchors

Add to the OLMo system prompt (Section 3.2) a spatial tracking block:

```
SPATIAL REASONING:
You are reasoning about a 3D physical volume measured in micrometers (µm).
All coordinates in findings are physical positions in this space — use them
to reason about what has and hasn't been examined.

When planning:
- Track which 3D regions are covered by prior views. If most views cluster
  near z=110µm, the volume edges (z≈0 and z≈220µm) are unexplored.
- Reference specific coordinates: say "neurons in z=[80..140]µm" not "in the middle".
- FOV at 'full' zoom covers the whole xy plane. Cropped zooms miss the edges —
  account for this when stating what you've confirmed.

When synthesizing:
- Spatially anchor each claim: "the 3 dense clusters found in z=[100..130]µm..."
- Note where you have NOT looked — spatial gaps in coverage are limits on confidence.
- If anisotropy is high (z coarser than xy), treat z-extent estimates as approximate.
```

**Why this matters:** Without explicit instruction, language models tend to reason about
findings abstractly ("neurons are present") rather than spatially ("neurons found in
z=80-160, not seen in z=0-80 or z=160-220 — 2/3 of the volume is unexamined").

---

### 15.4 Layer-type-specific metadata enrichment

Different layer types carry fundamentally different semantic content. The prompts
should reflect this.

#### Current state

`format_for_prompt()` lists layers as:
```
1. green_channel (image, 497×497×220µm) [visible]
2. cell_segmentation (segmentation, 497×497×220µm) [visible]
3. atlas_overlay (annotation, extent unknown) [visible]
```

No type-specific detail. Molmo gets the same generic "describe what you see" prompt
regardless of what layer types are present.

#### Proposed: enriched layer listing

```
LAYERS:
  1. green_channel  [image, visible]
     Size: 497×497×220µm  |  Voxel: 0.259×0.259×1.0µm  |  Contrast range: [0.10, 0.80]
     Role: fluorescence channel — bright regions = labeled structures

  2. cell_segmentation  [segmentation, visible]
     Size: 497×497×220µm  |  Voxel: 0.259×0.259×1.0µm
     Role: object segmentation — distinct colors = distinct labeled objects

  3. atlas_overlay  [annotation]
     Role: annotation layer — points, lines, or polygons placed by an analyst
```

Changes to `LayerInfo` and `format_for_prompt()`:
- Show voxel size per layer (from `voxel_scales`, converted to µm)
- Show shader range when present ("Contrast range: [0.10, 0.80]")
- Add a `role_hint` string derived from layer type + name heuristics (see §15.5)

#### Proposed: layer-aware description in `VolumeInfo.format_for_prompt()`

```python
for i, l in enumerate(self.layers, 1):
    vis = "visible" if l.visible else "hidden"
    vx, vy, vz = [s * 1e6 for s in self.voxel_scales[:3]]  # m → µm
    ext = f"{l.extent[0]:.0f}×{l.extent[1]:.0f}×{l.extent[2]:.0f}µm" if l.extent else "extent unknown"
    role = _infer_layer_role(l)

    layer_line = f"  {i}. {l.name}  [{l.type}, {vis}]\n"
    layer_line += f"       Size: {ext}  |  Voxel: {vx:.3g}×{vy:.3g}×{vz:.3g}µm\n"
    if l.shader_range:
        layer_line += f"       Contrast range: [{l.shader_range[0]:.2f}, {l.shader_range[1]:.2f}]\n"
    layer_line += f"       Role: {role}"
    lines.append(layer_line)
```

---

### 15.5 Layer role inference from name and source

Parse layer name and source URL to infer a semantic role hint. This is purely heuristic
— no ground truth — but it meaningfully guides both models.

```python
def _infer_layer_role(layer: LayerInfo) -> str:
    """Infer a human-readable role hint from layer name and source."""
    name = layer.name.lower()
    src = (layer.source or "").lower()

    if layer.type == "segmentation":
        if any(k in name for k in ("cell", "nuc", "soma", "neuron", "seg")):
            return "object segmentation — distinct colors = distinct labeled objects"
        return "segmentation overlay — distinct colors = distinct labeled objects"

    if layer.type == "annotation":
        if any(k in name for k in ("point", "dot", "mark")):
            return "point annotations placed by an analyst"
        if any(k in name for k in ("line", "path", "tract")):
            return "line/path annotations"
        return "annotation layer — points, lines, or polygons"

    # Image layer
    if any(k in name for k in ("dapi", "hoechst", "nuc", "nuclei")):
        return "nuclear stain — bright spots = all cell nuclei"
    if any(k in name for k in ("gfp", "green", "cfos", "neuron", "label")):
        return "fluorescence channel — bright spots = labeled structures (neurons/axons)"
    if any(k in name for k in ("dapi", "blue")):
        return "nuclear stain channel"
    if any(k in name for k in ("moving", "target", "r2r")) or "moving" in src:
        return "moving/target volume (registration partner)"
    if any(k in name for k in ("fixed", "ref", "reference")) or "fixed" in src:
        return "fixed/reference volume"
    if "ccf" in name or "ccf" in src or "atlas" in name:
        return "reference atlas (e.g. Allen CCF) — shows anatomical regions"
    if any(k in name for k in ("align", "registered", "warp")):
        return "registered/aligned channel"

    return "fluorescence channel — bright regions = signal, black = background"
```

This runs at `discover_volume()` time and stores the result in `LayerInfo`. No network
calls, no added latency.

---

### 15.6 Molmo2 domain scaffolding for neural imaging

Molmo2 has never been trained on:
- Fluorescence widefield/confocal/lightsheet microscopy images
- Brain tissue with sparse vs dense cell labeling
- Neuroglancer's specific rendering (dark canvas, colored segmentation overlays)
- Stitching artifacts, z-blur, saturation halos common in this data

The current interpret prompt gives Molmo2 no context about what it's looking at. Every
prompt should include an **imaging context block** placed before the description request:

```
IMAGING CONTEXT:
This is a fluorescence microscopy image of brain tissue rendered in Neuroglancer.
Rendering conventions:
- Background is black (zero signal).
- Bright regions = fluorescent signal (labeled cells, axons, or other structures).
- {segmentation_note}
- {annotation_note}
Visible layers: {layer_list_with_roles}
Scale: at this zoom, each pixel covers ~{um_per_pixel:.2f}µm.
{artifact_note}
```

Where the variables are filled per-view:

**`segmentation_note`** (only if a segmentation layer is visible):
```
A colored segmentation overlay is visible — each distinct color marks a
different labeled object. Do not confuse overlay colors with fluorescence
signal. The underlying image channel may or may not also be shown.
```

**`annotation_note`** (only if an annotation layer is visible):
```
Analyst-placed annotations are visible (dots, lines, or polygons).
These mark objects of interest identified in prior analysis.
```

**`artifact_note`** (always, as a calibration hint):
```
Common artifacts to be aware of: grid-line tiling boundaries (straight
dark lines at regular intervals), out-of-focus blur at z-edges, and
saturation halos around very bright structures.
```

**`um_per_pixel`** — computed from `crossSectionScale`:
```python
um_per_pixel = view_spec.get("crossSectionScale", 1.0) * 1.0  # already in µm/pixel
```
(since `crossSectionScale` is the physical size per pixel in NG coordinates)

This context block costs ~100-150 tokens per Molmo call but substantially improves
description accuracy for a model that has no prior calibration on this imaging type.

---

### 15.7 Multi-resolution awareness (zoom caveat)

Neuroglancer renders coarser resolution at wider zooms. When the model looks at the full
volume at "wide" or "full" zoom, it may be seeing a 4× or 8× downsampled version — fine
cell processes, thin axons, and small puncta may not be visible.

Add to the zoom table note (currently only warns about cropping):

```
RESOLUTION NOTE:
At 'wide' and 'full' zoom, Neuroglancer renders downsampled data. Fine structures
(thin axons, small puncta) may be invisible at these zooms. Zoom into 'region' or
'close-up' to see fine detail — but remember this crops the field of view.
For counting: use 'region' or 'close-up' zoom so cell bodies are at least 10-20px
across, or the count will miss small/dim objects.
```

This note belongs in the action schema sent to OLMo (alongside the existing crop
warning) so the planner makes better zoom choices.

---

### 15.8 Pixel→physical coordinate translation for count detections

#### Design: pixel space for visuals, physical space for reasoning

Molmo's pointing detections return pixel `(x, y)` coordinates on the 1024×1024 canvas.
These pixel coords serve two consumers with different needs:

1. **Visual annotation** (human-facing) — draw red circles on the image/video at the
   exact pixel positions. This is the v3 behaviour and must be preserved.
2. **Spatial reasoning** (OLMo-facing) — summarize WHERE detections cluster in physical
   µm coordinates so OLMo can reason about spatial distribution across the volume.

Both consumers read from the same `points` list. The pixel→physical translation is a
read-only pass that produces a text summary — it never modifies the pixel coordinates.

```
Molmo pointing → pixel (frame_idx, x, y) per keyframe
                    │
                    ├──→ annotate_scan_frames(frames, points)   ← pixel space (unchanged)
                    ├──→ annotate_screenshot(img, points)        ← pixel space (unchanged)
                    │
                    └──→ pixel_to_physical(points, view_params)  ← NEW: physical space
                         └──→ spatial summary text for OLMo finding
```

#### Translation function

The math is deterministic — we know the view center, scale, and layout at capture time:

```python
HALF_VIEWPORT = VIEWPORT_SIZE / 2  # 512

def pixel_to_physical(
    pixel_x: float,
    pixel_y: float,
    view_center: tuple[float, float, float],
    scale: float,
    layout: str,
) -> tuple[float, float, float]:
    """Convert pixel coords on the 1024×1024 canvas to physical µm coords.

    Parameters
    ----------
    pixel_x, pixel_y : float
        Pixel position on the canvas (0–1024).
    view_center : tuple
        (cx, cy, cz) physical position of the view center in µm.
    scale : float
        crossSectionScale — physical µm per pixel.
    layout : str
        "xy", "xz", or "yz" — determines axis mapping.

    Returns
    -------
    (phys_x, phys_y, phys_z) in µm.
    """
    dx = (pixel_x - HALF_VIEWPORT) * scale
    dy = (pixel_y - HALF_VIEWPORT) * scale
    cx, cy, cz = view_center

    if layout == "xy":
        return (cx + dx, cy + dy, cz)
    elif layout == "xz":
        return (cx + dx, cy, cz + dy)
    elif layout == "yz":
        return (cx, cy + dx, cz + dy)
    return (cx + dx, cy + dy, cz)  # fallback for 4panel (see caveat below)
```

#### Reliable for all single-panel layouts

| Layout | Screen X → | Screen Y → | Fixed axis | Reliable? |
|--------|-----------|-----------|------------|-----------|
| `xy` | phys_x | phys_y | z (slice position) | Yes |
| `xz` | phys_x | phys_z | y (slice position) | Yes |
| `yz` | phys_y | phys_z | x (slice position) | Yes |

For **4panel**: the canvas has 4 sub-panels with different axis mappings. A pixel at
(300, 700) could be in the xy panel or the yz panel depending on the viewport layout.
Rather than parsing Neuroglancer's panel geometry (fragile), **restrict count actions
to single-panel layouts**. The action schema already uses single-panel layouts for
count actions — add a validation check:

```python
# In validate_action(), for count actions:
if action_type == "count":
    layout = action.get("layout", action.get("view", {}).get("layout", "xy"))
    if layout == "4panel":
        print("  WARNING: count action on 4panel — converting to xy for reliable pointing")
        action["layout"] = "xy"  # or action["view"]["layout"] = "xy"
```

#### Spatial distribution summary

After translation, summarize the physical coordinates for OLMo:

```python
def summarize_spatial_distribution(
    phys_points: list[tuple[float, float, float]],
    volume_info: VolumeInfo,
) -> str:
    """Summarize where detections cluster in physical coordinates."""
    if not phys_points:
        return "No detections."

    xs = [p[0] for p in phys_points]
    ys = [p[1] for p in phys_points]
    zs = [p[2] for p in phys_points]

    lines = []
    lines.append(f"Spatial extent of detections:")
    lines.append(f"  x=[{min(xs):.0f}..{max(xs):.0f}]µm  "
                 f"(volume: 0..{volume_info.shape[0]:.0f})")
    lines.append(f"  y=[{min(ys):.0f}..{max(ys):.0f}]µm  "
                 f"(volume: 0..{volume_info.shape[1]:.0f})")
    lines.append(f"  z=[{min(zs):.0f}..{max(zs):.0f}]µm  "
                 f"(volume: 0..{volume_info.shape[2]:.0f})")

    # Centroid
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = sum(zs) / len(zs)
    lines.append(f"  Centroid: ({cx:.0f}, {cy:.0f}, {cz:.0f})µm")

    return "\n".join(lines)
```

#### Integration point in `molmo_glancer.py`

After the existing annotation call (~line 962):

```python
# Pixel-space annotation (visual, unchanged from v3)
if points:
    annotate_scan_frames(frames, points, scan_count)

# Physical-space summary (new, for OLMo reasoning)
view_center = (
    action.get("start", {}).get("x", cx),  # or midpoint of scan
    action.get("start", {}).get("y", cy),
    action.get("start", {}).get("z", cz),
)
# For scans: translate each keyframe's points using that frame's interpolated position
phys_points = []
for frame_idx, px, py in points:
    t = frame_idx / max(len(frames) - 1, 1)
    frame_center = tuple(s_i + t * (e_i - s_i) for s_i, e_i in zip(s, e))
    phys = pixel_to_physical(px, py, frame_center, scale, layout)
    phys_points.append(phys)

spatial_summary = summarize_spatial_distribution(phys_points, volume_info)
```

The `spatial_summary` string is appended to the count finding text that goes into
history, so OLMo sees both the detection statistics AND where they are in 3D space.

#### Scan frame position context for Molmo2

For scan interpretation (not counting — the prose description), tell Molmo what
physical position each frame corresponds to so its descriptions are naturally anchored.
Add to the scan interpret prompt:

```python
# Build frame position legend for scan interpret prompt
n_frames = len(frames)
legend_frames = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]
legend_lines = []
for fi in legend_frames:
    t = fi / max(n_frames - 1, 1)
    pos = s + t * (e - s)
    legend_lines.append(f"  Frame {fi}: {scan_axis}={pos[{'x':0,'y':1,'z':2}[scan_axis]]:.0f}µm")

frame_legend = "Frame positions:\n" + "\n".join(legend_lines)

interpret_prompt = (
    f"Question: \"{question}\"\n\n"
    f"{user_prompt}\n"
    f"Scan: {len(frames)} frames along {scan_axis}, "
    f"~{frame_spacing:.1f}µm between frames, "
    f"{total_dist:.0f}µm total.\n"
    f"{frame_legend}\n"
    f"Describe what you see across the frames. ..."
)
```

This gives Molmo natural anchoring without requiring it to output coordinates:
"Frame 15 (z=66µm): first neurons appear" instead of just "around frame 15."

---

### 15.9 Implementation changes required

| Change | File | Complexity |
|---|---|---|
| Store `fov_feedback` in all history entries | `molmo_glancer.py` | Low |
| Populate `fov_feedback` for scan + first-look | `molmo_glancer.py` | Low |
| Fix "voxels" → "µm" in first-look prompt | `molmo_glancer.py:690` | Trivial |
| Add `_infer_layer_role()` to `LayerInfo` | `volume_info.py` | Low |
| Enrich `format_for_prompt()` (voxel size, shader range, role) | `volume_info.py` | Low |
| Add spatial reasoning anchors to OLMo system prompt | `molmo_glancer.py` | Low |
| Add imaging context block to Molmo2 interpret prompts | `molmo_glancer.py` | Medium |
| Add multi-resolution caveat to zoom table | `volume_info.py` | Low |
| Add `pixel_to_physical()` + `summarize_spatial_distribution()` | `volume_info.py` | Low |
| Convert count detections to physical coords + append to finding | `molmo_glancer.py` | Low |
| Validate count actions reject 4panel layout | `molmo_glancer.py` | Trivial |
| Add frame position legend to scan interpret prompt | `molmo_glancer.py` | Low |

All changes are in existing files. No new modules needed.

---

## 14. Open Design Questions

1. **Should OLMo's reasoning output be included in Molmo2's interpret prompts?**
   Currently no — Molmo2 gets a purpose string and the image. Adding OLMo's full
   reasoning would increase Molmo2's context usage. Recommendation: pass only the
   `purpose` field, not the full reasoning.

2. **Should the actions array allow mixing visual and terminal actions?**
   E.g., `[screenshot, screenshot, answer]`? Recommendation: No. If OLMo is ready
   to answer, it should answer directly. If it needs more views, it plans views.
   Mixing would complicate the execution flow.

3. **Should count interpretation move to OLMo?**
   In v3, `ask_text()` interprets count results using the 7B model. In v4, this
   should definitely use OLMo — statistical reasoning about keyframe overlap,
   double-counting, and extrapolation is exactly where the 32B model shines.
   Recommendation: Yes, route count interpretation to OLMo.

4. **How to handle OLMo JSON parse failures?**
   The 32B model is much more reliable at JSON generation than the 7B, but failures
   still happen. Strategy: same retry with format reminder as v3, but adapt the
   reminder to reference the batch format. If retry fails, treat the text as a
   reason action.

---

## 16. OLMo 3.1 32B Think — Effective Usage

### 16.1 Chat template: use the system role

OLMo 3.x uses ChatML format (`<|im_start|>system/user/assistant<|im_end|>`). The
system role IS natively supported — this is what the model was trained on.

**Do:** Use `apply_chat_template()` with a proper system message:
```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
```

**Don't:** Stuff the system prompt into the user message. This mismatches training
distribution and wastes the system role's special position in the attention pattern.

The system prompt (§3.2) + spatial reasoning anchors (§15.3) + domain reasoning
anchors (§7.3) go in the system role. Volume info, action schema, findings, and
the question go in the user role.

**Note:** Molmo2's chat template does NOT support a system role (it uses
`<|im_start|>user/assistant` only, with images prepended). OLMo and Molmo2 use
different chat template paths — this must be handled in `ask_text_olmo()` vs
`ask_vision()`.

---

### 16.2 Sampling parameters

Not discussed elsewhere in the plan. These matter for JSON reliability and
reasoning quality.

**For structured output (action planning, decisions):**
```python
generation_config = {
    "temperature": 0.2,        # Low — think block handles exploration
    "top_p": 0.9,              # Mild nucleus sampling
    "repetition_penalty": 1.1, # Prevent think loops (see §16.4)
    "do_sample": True,         # Required for temperature to take effect
}
```

**For synthesis/answer:**
```python
generation_config = {
    "temperature": 0.5,        # Slightly higher for natural prose
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,
}
```

**For JSON retry after parse failure:**
```python
generation_config = {
    "temperature": 0.1,        # Near-greedy — just produce valid JSON
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,
}
```

Store these in the CONFIG dict as named presets:
```python
CONFIG = {
    ...
    "olmo_sampling_structured": {"temperature": 0.2, "top_p": 0.9, "repetition_penalty": 1.1},
    "olmo_sampling_synthesis": {"temperature": 0.5, "top_p": 0.9, "repetition_penalty": 1.1},
    "olmo_sampling_retry":     {"temperature": 0.1, "top_p": 0.9, "repetition_penalty": 1.1},
}
```

---

### 16.3 Truncated think block detection

If `max_new_tokens` is exhausted while the model is still inside a `<think>` block,
we get raw text with an unclosed `<think>` tag and no usable response. The current
`strip_think_tokens()` uses `re.findall(r'<think>(.*?)</think>', ...)` which requires
a closing tag — an unclosed block produces empty `clean_text` and a parse failure.

**Detection and recovery:**

```python
def strip_think_tokens(text: str) -> tuple[str, str, bool]:
    """Extract and separate think blocks from OLMo output.

    Returns (clean_text, think_content, was_truncated).
    was_truncated is True if a <think> block was never closed — meaning
    the model ran out of tokens before producing its actual response.
    """
    # Check for unclosed think block
    open_count = text.count("<think>")
    close_count = text.count("</think>")
    was_truncated = open_count > close_count

    # Extract closed blocks
    think_blocks = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # If truncated, also strip the unclosed trailing block
    if was_truncated:
        clean = re.sub(r'<think>(?!.*</think>).*$', '', clean, flags=re.DOTALL)

    return clean.strip(), '\n'.join(think_blocks), was_truncated
```

**When `was_truncated` is True:**
1. Log a warning: "Think block truncated — max_new_tokens too low for this prompt"
2. Retry with 2× `max_new_tokens` (up to a hard cap of 16384)
3. If retry also truncates, fall back to a simplified prompt (drop older history
   entries to shorten the input, giving more room for generation)

This prevents silent failures where the model "thought hard" but produced nothing.

---

### 16.4 Think loop detection

Think models can occasionally enter repetitive reasoning loops, especially on
ambiguous or under-constrained problems. The `<think>` block keeps producing
variations of the same reasoning without converging.

Signs:
- Think block exceeds ~3000 tokens without producing a conclusion
- Repeated phrases or near-identical sentences within the think content

**Mitigation:** `repetition_penalty: 1.1` (§16.2) handles most cases. For
additional safety, after generation completes, check think content length:

```python
clean_text, think_content, truncated = strip_think_tokens(raw_output)
if len(think_content) > 3000 and not clean_text.strip():
    print("  WARNING: Long think block with no response — possible loop")
    # Retry with higher repetition_penalty and lower temperature
```

This is a runtime diagnostic, not a prevention mechanism — the sampling params
are the first line of defense.

---

### 16.5 Think content as diagnostic signal

The think block content is logged to transcript.md (§7.2). Beyond logging, it
can be mined at runtime for lightweight quality signals:

**Confidence calibration:**
```python
UNCERTAINTY_MARKERS = ["not sure", "unclear", "uncertain", "contradictory",
                       "hard to tell", "insufficient", "cannot determine"]

def assess_think_confidence(think_content: str) -> str:
    """Check think block for uncertainty signals."""
    lower = think_content.lower()
    matches = [m for m in UNCERTAINTY_MARKERS if m in lower]
    if len(matches) >= 2:
        return "low"   # multiple uncertainty signals
    elif matches:
        return "medium"
    return "high"
```

If the model's answer says `"confidence": "high"` but the think block contains
multiple uncertainty markers, flag this in the transcript as a potential
miscalibration. Don't override the model's stated confidence — just log it for
human review.

**Spatial reasoning quality check:**
```python
def check_spatial_grounding(think_content: str) -> bool:
    """Check if think block references physical coordinates."""
    # Look for coordinate-like patterns: "z=110", "x=[80..200]", "(249, 249, 110)"
    return bool(re.search(r'[xyz]\s*[=\[]\s*\d', think_content))
```

If the think block doesn't reference any coordinates despite having FOV data in the
findings, the spatial reasoning anchors (§15.3) aren't being effective. Log this for
prompt tuning.

These checks cost ~0.1ms per call (regex on a string) — negligible.

---

### 16.6 INT8 quantization and think chain quality

INT8 via bitsandbytes is <1% degradation on standard benchmarks (MMLU, MATH). But
think chains are long autoregressive sequences where small per-token errors can
compound — 2000 tokens of chained reasoning is more sensitive than a 50-token
factual answer.

**Monitoring strategy for L40S testing (implementation step 9):**

1. Run 3-5 representative questions at INT8 on L40S
2. If possible, run the same questions at BF16 on a larger instance (one-time reference)
3. Compare:
   - Think block coherence (does it stay on topic or drift?)
   - JSON action validity rate (parse failures at INT8 vs BF16)
   - Answer quality and coordinate grounding
4. If INT8 think quality degrades noticeably:
   - Try pre-quantized INT8 checkpoints (e.g. from unsloth) — these use better
     calibration than on-the-fly bnb quantization
   - Or increase `repetition_penalty` slightly (1.15) to counteract drift

Most likely outcome: INT8 is fine. But the monitoring step is cheap and catches
problems before they become mysterious answer quality issues.

---

### 16.7 Think vs Instruct: when to skip thinking

The plan downloads only OLMo 3.1 32B Think. The Instruct variant (no `<think>`
blocks, faster inference) exists but would require a separate download and swap
management for two OLMo variants — not worth the complexity for v4.

However, if testing reveals that some call types don't benefit from thinking:
- **JSON retry** — after a parse failure, the retry prompt is just "output valid JSON."
  Think adds latency here for minimal benefit.
- **Count interpretation** — straightforward statistical summary may not need
  deep reasoning chains.

**v4 approach:** Use Think everywhere. If think overhead on fast calls is
problematic, the Instruct variant is a v5 optimization — same architecture,
same weights download path, just a different HF model ID.

**Alternative for v4:** For calls where thinking isn't needed, we can't disable
the `<think>` mechanism, but we can use lower `max_new_tokens` (2048 instead of
4096) and higher `temperature` to make the think block shorter and less
deliberate. This is an imprecise lever but avoids the two-model complexity.

---

### 16.8 Updated CONFIG entries

The following entries should be added/updated in the `CONFIG` dict (§12):

```python
CONFIG = {
    # ... existing entries ...

    # OLMo generation budgets (§7.1 revised)
    "olmo_max_new_tokens_decision": 4096,
    "olmo_max_new_tokens_synthesis": 8192,
    "olmo_max_new_tokens_retry": 2048,     # tight budget, minimal think expected
    "olmo_max_new_tokens_hard_cap": 16384,  # absolute max for truncation recovery

    # OLMo sampling presets (§16.2)
    "olmo_sampling_structured": {
        "temperature": 0.2, "top_p": 0.9,
        "repetition_penalty": 1.1, "do_sample": True,
    },
    "olmo_sampling_synthesis": {
        "temperature": 0.5, "top_p": 0.9,
        "repetition_penalty": 1.1, "do_sample": True,
    },
    "olmo_sampling_retry": {
        "temperature": 0.1, "top_p": 0.9,
        "repetition_penalty": 1.1, "do_sample": True,
    },
}
```
