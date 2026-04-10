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

### 1.2 Single-action-per-iteration swap overhead

Each iteration plans one action, captures one view, interprets it, and then decides the
next. With model swapping this becomes 2 swap cycles per view — ~40-70s of overhead per
view. This is the cost of the 7B→32B quality jump.

> **Future optimization:** Batched multi-view planning (N actions per iteration to
> amortize swap overhead) is designed in `PLAN_batched_planning.md` for implementation
> after v4's single-action OLMo flow is validated.

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

Each iteration makes 4 OLMo calls and 1 Molmo call, with only 2 physical swaps.
OLMo stays loaded across the step 6 → step 1 boundary between iterations.

```
┌─────────────── Iteration N ───────────────────────────────────┐
│                                                                │
│  ┌─ OLMo 3.1 32B Think — 3 calls, stays loaded ────────────┐  │
│  │                                                           │  │
│  │  Step 1: Plan — natural language investigation strategy   │  │
│  │  "The lower-left showed misalignment; I need to check     │  │
│  │   whether this persists at other z-depths..."             │  │
│  │                                                           │  │
│  │  Step 2: Action — strict JSON from schema                 │  │
│  │  {"action": "screenshot", "view": {...}, "purpose": "..."}│  │
│  │  → validate_action() resolves coords/zoom/layers          │  │
│  │                                                           │  │
│  │  Step 3: Vision instructions — craft Molmo2 guidance      │  │
│  │  "Focus on channel overlap in lower-left quadrant.        │  │
│  │   Compare green-magenta separation vs upper-right..."     │  │
│  │  (count: refine target description for pointing instead)  │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│              ↓ swap_to_molmo (~5s)                               │
│  ┌─ Molmo2-O-7B (vision) ──────────────────────────────────┐  │
│  │                                                           │  │
│  │  Step 4: Capture screenshot/scan/count frames             │  │
│  │  Step 5: Interpret with OLMo-crafted instructions         │  │
│  │          + auto-generated template (imaging context, FOV)  │  │
│  │  (count: per-keyframe pointing, not prose interpretation)  │  │
│  │  → finding                                                │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│              ↓ swap_to_olmo (~30-60s)                            │
│  ┌─ OLMo 3.1 32B Think — reasoning ─────────────────────────┐  │
│  │                                                           │  │
│  │  Step 6: Reason over finding + prior evidence             │  │
│  │  → continue (next iteration) or answer (end loop)         │  │
│  │  (count: interprets pointing statistics here)             │  │
│  │  OLMo stays loaded → step 1 of next iteration             │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Short-circuit:** If step 2 outputs `reason` or `answer`, steps 3-5 are
skipped — no Molmo swap needed. OLMo handles it directly.

### 2.2 Prompt routing

| Step | Prompt type | Model | Why |
|---|---|---|---|
| — | First look interpret | Molmo2 | Needs vision (Phase 1, before loop) |
| 1 | Investigation plan | OLMo 32B | Strategic reasoning, benefits from <think> |
| 2 | Action decision (JSON) | OLMo 32B | Structured output from schema |
| 3 | Vision instructions | OLMo 32B | Context-aware guidance for Molmo2 |
| 3' | Count target refinement | OLMo 32B | Variant: refine target noun/size for pointing |
| 5 | Screenshot interpret | Molmo2 | Needs vision |
| 5 | Scan interpret | Molmo2 | Needs vision (video frames) |
| 5' | Count pointing | Molmo2 | Needs vision (per-keyframe pointing) |
| 6 | Reasoning / synthesis | OLMo 32B | Evidence synthesis, decide next step |
| 6' | Count interpret | OLMo 32B | Statistical reasoning about detections |
| — | Final answer | OLMo 32B | Quality-critical synthesis (post-loop) |
| — | Forced answer | OLMo 32B | Last-resort synthesis (max iterations) |

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

2. **Expanded action vocabulary** — More fine-grained control over the Neuroglancer
   state, reflecting the full `build_clean_state()` API surface.

3. **Purpose-driven prompts** — Each action carries a `purpose` field explaining
   WHY this view is needed, not just what to look for.

### 4.2 New action schema

```
ACTION SCHEMA:

Respond with a JSON object representing your next action.

──── Visual Actions ─────────────────────────────────────────────

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
| Response format | Single `{"action": ...}` | Single `{"action": ..., "purpose": "..."}` |
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

## 5. Per-Iteration OLMo Prompts (Steps 1-3)

Each iteration has three OLMo calls before the Molmo vision phase. All three
run while OLMo is continuously loaded — no swaps between them.

### 5.1 Step 1 — Investigation plan (natural language)

Produces a natural language reasoning trace about what to look at next and why.
This output is stored in history and passed to step 2 as context. No JSON, no
action schema — pure strategic reasoning.

**First iteration (after first look):**
```
You have examined a 3D volume and received this initial description:
"{first_look_finding}"

QUESTION: "{question}"

VOLUME:
{volume_info}

Plan your investigation strategy. What should you look at first, and why?

Consider:
- What spatial regions need examination to answer the question?
- Would a different layout (xy vs xz vs yz) reveal different information?
- Would a scan (video sweep) show spatial distribution better than a static view?
- Would toggling layer visibility reveal alignment, segmentation quality, etc.?
- Is the question quantitative (need count action) or qualitative (scan/screenshot)?
```

**Subsequent iterations:**
```
QUESTION: "{question}"

INVESTIGATION SO FAR:

{structured_findings}

Iteration {N}/{max}. What should you investigate next, and why?

Consider what spatial regions remain unexplored, whether findings are
consistent, and whether you have enough evidence to answer.
```

### 5.2 Step 2 — Action decision (strict JSON)

Receives step 1's plan and outputs a valid JSON action from the schema.
This is the only call that includes the action schema.

```
YOUR INVESTIGATION PLAN:
{step1_plan_output}

{action_schema}

Output the JSON action that executes your plan. Include a "purpose" field
explaining what you expect to learn from this view.

If you already have enough evidence, use the answer action instead.
```

### 5.3 Step 3 — Vision instructions for Molmo2

Receives the validated action JSON and crafts specific instructions for the
vision model. This output is prepended to the auto-generated interpret template
(§8.1-8.3), which provides the structural scaffolding (imaging context, spatial
metadata, FOV). OLMo adds investigation-specific focus.

**For screenshot/scan actions:**
```
You have planned this view:
{action_json_summary}

PURPOSE: {purpose}
QUESTION: "{question}"

RECENT FINDINGS:
{last_2_findings}

Write specific instructions for the vision model that will interpret this view.
Tell it:
- What specific features or structures to focus on
- What region of the image matters most for this investigation
- What to compare against prior findings (if any)
- Any artifacts or confounds to watch for

Keep it concise (2-4 sentences). The vision model also receives standard
imaging context and spatial metadata automatically.
```

**Count variant — refine target description for pointing:**
```
You have planned a count action:
{action_json_summary}

PURPOSE: {purpose}
TARGET: "{target}"

The vision model will point to each instance of the target on sampled keyframes.
Refine the target description to help the model identify the right objects:
- What size and shape are the targets?
- What intensity or color distinguishes them from background?
- Should the model ignore any similar-looking artifacts?

Output a refined pointing instruction (1-2 sentences).
```

### 5.4 Structured findings format

Replace the flat `format_history_entry()` with structured blocks that OLMo can
reference by iteration number:

```
── Iteration 0 (first look) ──────────────────────────────────
  View: screenshot, 4panel, center, full zoom
  Finding: "Dense field of fluorescent neurons visible in xy plane.
    Approximately 20-30 bright cell bodies in this cross-section..."
  FOV: x=[0..497], y=[0..497]

── Iteration 1 ───────────────────────────────────────────────
  View: screenshot, xy, z=50, full zoom
  Purpose: "Check neuron density in first quarter of volume"
  Finding: "Sparse — only 5-8 neurons visible at this depth..."
  FOV: x=[0..497], y=[0..497]

── Iteration 2 ───────────────────────────────────────────────
  View: scan, z_sweep, z=0..220, 50 frames
  Purpose: "Survey full z-depth for neuron distribution"
  Finding: "Neurons concentrated in z=80-180 range, sparse at edges..."
  Swept: z=[0..220]µm, FOV per frame: x=[0..497], y=[0..497]

── Iteration 3 ───────────────────────────────────────────────
  View: count, z_sweep, z=80..180, target=neurons, keyframe_interval=3
  Purpose: "Get grounded count in high-density region"
  Finding: "DETECTED: 187 instances across 12/17 keyframes.
    Per keyframe: min=3, max=24, median=15."
  Reasoning: "187 raw detections with keyframe_interval=3 and ~5µm spacing
  means significant double-counting of neurons spanning multiple slices."
```

This format:
- Numbers iterations for easy reference
- Preserves `purpose` alongside `finding` so OLMo can assess whether its investigation
  strategy is working
- Includes FOV context so OLMo knows what spatial region each view covered
- Includes the reasoning output from the OLMo reason phase

---

## 6. Reasoning Phase Prompt (Step 6)

After Molmo2 interprets the captured view, OLMo reasons over the new finding
combined with prior history. This is an explicit separate call (step 6) — not
folded into the decision prompt. OLMo stays loaded from this step into the next
iteration's step 1, so the reasoning flows directly into planning.

### 6.1 Post-finding reasoning prompt

```
QUESTION: "{question}"

NEW FINDING (iteration {N}):
{latest_finding_with_fov}

INVESTIGATION SO FAR:
{structured_findings}

Analyze the new finding in context of your prior investigation:

1. Does this finding confirm, contradict, or extend previous findings?
2. What spatial regions remain unexplored?
3. Do you have sufficient evidence to answer the question confidently?

If you have enough evidence, respond with your answer using the answer action:
{"action": "answer", "answer": "...", "confidence": "...", "evidence_summary": "..."}

Otherwise, summarize your current understanding and what remains uncertain.
This reasoning will inform your next investigation step.
```

### 6.2 Count interpretation variant (step 6')

When the action was `count`, step 6 receives the raw pointing statistics instead
of a prose finding. OLMo interprets the detection data:

```
QUESTION: "{question}"

COUNT RESULTS (iteration {N}):
Target: "{target}"
{pointing_statistics}
{spatial_distribution_summary}

INVESTIGATION SO FAR:
{structured_findings}

Interpret these detection results:
- Account for double-counting (objects spanning multiple z-slices appear
  in adjacent keyframes)
- Keyframe spacing vs object size: if spacing < diameter, expect overcounting
- Detection confidence: low-contrast or partial objects may be missed

Then decide: do you have sufficient evidence to answer, or do you need
more investigation?
```

### 6.3 Explicit reason action (short-circuit from step 2)

When step 2 outputs a `reason` action (no new view needed), steps 3-5 are
skipped and OLMo reasons directly:

```
QUESTION: "{question}"

INVESTIGATION SO FAR:
{structured_findings}

Your reasoning request: "{reason_question}"

Analyze the evidence and decide your next step:

1. If the evidence is sufficient, provide your answer.
2. If findings conflict, identify the contradiction and plan a view to resolve it.
3. If critical regions remain unexplored, describe what to investigate next.
```

---

## 7. Think Token Strategy

### 7.1 Token budgets by prompt type

Think models produce `<think>` blocks (often 1000-2500 tokens) BEFORE the actual
response. The budget must accommodate both. If `max_new_tokens` is too small, the
model exhausts its budget mid-think and never emits the JSON — a silent total failure.

| Step | Prompt type | max_new_tokens | Think budget | Response budget | Rationale |
|---|---|---|---|---|---|
| 1 | Investigation plan | 4096 | ~1500-2000 | ~500-1000 (text) | Strategic reasoning about what to look at |
| 2 | Action decision (JSON) | 4096 | ~1500-2000 | ~500-800 (JSON) | Structured action output |
| 3 | Vision instructions | 2048 | ~500-1000 | ~200-400 (text) | Light call — targeted guidance |
| 3' | Count target refinement | 2048 | ~500-1000 | ~100-200 (text) | Light call — refine target noun |
| 6 | Post-finding reasoning | 6144 | ~2000-3000 | ~500-1000 (text) | Deeper chains for synthesis |
| 6' | Count interpretation | 6144 | ~2000-3000 | ~500-1000 (text) | Statistical reasoning about detections |
| — | Final answer | 8192 | ~2000-4000 | ~1000-2000 (text) | Full synthesis, longest think |

**Why the old 2048/4096 budgets were too low:** A 2048 budget with 1500 think tokens
leaves only 548 tokens for the JSON action — barely enough for a single action with
purpose, and truncation mid-JSON is a parse failure requiring a retry (which wastes
a full swap cycle of ~40-70s).

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

Molmo2 continues to handle all vision calls. The interpret prompt has two parts:

1. **OLMo-crafted instructions** (from step 3) — investigation-specific guidance
   that tells Molmo2 what to focus on, what to compare, what to ignore.
2. **Auto-generated template** (below) — structural scaffolding with imaging
   context, spatial metadata, and FOV. Always present.

`build_vision_interpret_prompt()` concatenates: step-3 instructions + template.

### 8.1 Screenshot interpretation template

```
{olmo_vision_instructions}

---

Question: "{question}"

This is a {layout} view at position ({x}, {y}, {z}), zoom={zoom_name}.
Visible layers: {visible_layer_list}
Field of view: {fov_description}

{imaging_context_block}

Describe what you see. Report:
- What structures are present (type, shape, intensity)
- Approximate counts if objects are discrete and countable
- Spatial distribution (clustered, uniform, sparse/dense regions)
- Anything unusual or noteworthy
```

### 8.2 Scan interpretation template

```
{olmo_vision_instructions}

---

Question: "{question}"

Scan: {num_frames} frames along {axis}, {start} → {end}
Frame spacing: ~{spacing}µm, total distance: {total}µm
Layout: {layout}, zoom: {zoom_name}

{imaging_context_block}

Describe what you observe across the frames:
- How does the content change along the scan axis?
- Where are structures most dense vs sparse?
- Are there boundaries, transitions, or abrupt changes?
- Estimate the spatial extent of notable features
```

### 8.3 Count pointing (OLMo-refined target description)

The pointing prompt uses OLMo's refined target description from step 3 (count
variant) instead of the generic `{target}` noun. The refined description
helps Molmo2 identify the right objects and avoid false detections.

```
{olmo_refined_pointing_instruction}
Each {target_singular} is approximately {neuron_pixels} pixels across.
```

If step 3 produced no refinement (or for backward compatibility), falls back to:
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

After many iterations, findings can accumulate. Strategy:

1. **Iterations 1-3:** Full findings with FOV, purpose, reasoning
2. **Iterations 4+:** Compress older iterations to one-line summaries:
   ```
   [Iter 1: screenshot xy z=50 — sparse, only 5-8 neurons]
   [Iter 2: scan z=0..220 — neurons concentrated z=80-180, sparse at edges]
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

- **One prompt path:** Schematic placeholders + purpose-driven actions. No fallback to
  concrete examples.
- **No image downscaling:** `max_image_side` is gone. Molmo always gets full-res images.
- **OLMo always available:** No "skip swaps, use Molmo for text" fallback. Every
  text call goes through OLMo 32B.
- **GPU rendering always on:** Chromium always launches with `--use-gl=egl`.

This means there is exactly one code path to test and debug.

---

## 12. Implementation Checklist

### New functions to add to `molmo_glancer.py`:

**OLMo generation:**
1. `strip_think_tokens(text)` — separate `<think>` blocks, detect truncation (§16.3)
2. `ask_text_olmo(manager, system_prompt, user_prompt, max_new_tokens, sampling)` — OLMo generation via ChatML system+user roles, think handling, truncation recovery (§16.1, §16.3). Returns `(text, token_counts)` matching `ask_text()` shape.

**Per-iteration prompt builders (steps 1-3, 6):**
3. `build_olmo_system_prompt(volume_info)` — system prompt with reasoning + spatial anchors (§3.2, §15.3, §7.3)
4. `build_plan_prompt(question, volume_info, history, iteration)` — step 1: natural language investigation plan (§5.1)
5. `build_action_prompt(plan_output, action_schema)` — step 2: strict JSON action from schema (§5.2)
6. `build_vision_instructions_prompt(action, question, recent_findings)` — step 3: craft targeted Molmo2 instructions (§5.3). Count variant: refine target description for pointing.
7. `build_reasoning_prompt(question, finding, history)` — step 6: post-finding reasoning (§6.1). Count variant: includes pointing statistics interpretation (§6.2).

**Vision prompt assembly:**
8. `build_vision_interpret_prompt(action, question, volume_info, olmo_instructions)` — combines OLMo-crafted instructions (step 3) with auto-generated template + imaging context block (§8, §15.6)

**History and context:**
9. `format_structured_findings(history)` — replaces `format_history_entry` with richer format including FOV (§5.4)
10. `compress_findings(history, max_entries)` — token-aware history compression
11. `estimate_tokens(text)` — lightweight token counting for budget enforcement

**Diagnostics:**
12. `assess_think_confidence(think_content)` — mine think blocks for uncertainty signals (§16.5)

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
    # OLMo generation budgets (§7.1 — includes think + response headroom)
    "olmo_max_new_tokens_plan": 4096,          # step 1: investigation plan
    "olmo_max_new_tokens_decision": 4096,      # step 2: JSON action
    "olmo_max_new_tokens_vision_instr": 2048,  # step 3: vision instructions (light call)
    "olmo_max_new_tokens_reasoning": 6144,     # step 6: post-finding reasoning
    "olmo_max_new_tokens_synthesis": 8192,      # final answer
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

1. **Action schema** — replace concrete examples with schematic placeholders + `purpose` field
2. **Step 1 plan prompt** — natural language investigation planning (§5.1)
3. **Step 2 action prompt** — strict JSON with schema (§5.2)
4. **Step 3 vision instructions prompt** — craft Molmo2 guidance (§5.3) + count variant
5. **Step 6 reasoning prompt** — post-finding reasoning (§6.1) + count variant (§6.2)
6. **System prompt** — new OLMo-specific system prompt (not sent to Molmo2)
7. **Interpret prompts** — combine OLMo instructions + auto-generated template (§8)
8. **First look prompt** — add imaging context block, fix "voxels" → "µm"
9. **`save_prompt_templates()`** — update to reflect all new prompt formats

---

## 13. Example: Iteration Walkthrough

### Setup
- Volume: 497x497x220µm, 2 image layers (fixed green, moving magenta), 4panel
- Question: "How well are neurons aligned between fixed and moving volumes?"

### Iteration 1

**Step 1 — OLMo plans** (natural language)

OLMo receives: first look finding + question + volume info
```
<think>
The question is about alignment quality between two channels. I need to see
them overlaid to check for systematic shifts. Starting with both channels
visible at the center — if well-aligned, green+magenta overlap produces
white/bright regions. Misalignment shows as separated spots.
</think>
```
Plan: "I'll overlay both channels at the volume center to assess overall
alignment quality. Green-magenta overlap should produce yellow/white if
well-aligned; separated colors indicate misalignment."

**Step 2 — OLMo action** (strict JSON)

OLMo receives: step 1 plan + action schema
```json
{"action": "screenshot",
 "view": {"x": 249, "y": 249, "z": 110, "layout": "xy", "zoom": "full"},
 "show": [1, 2],
 "purpose": "Overlay both channels to see alignment quality at center"}
```

**Step 3 — OLMo crafts vision instructions**

OLMo receives: validated action JSON + purpose + question
```
"Look for regions where green and magenta signals overlap to form yellow/white
 — this indicates good alignment. Pay special attention to individual neurons:
 do they appear as single merged spots or as separated green-magenta pairs?
 Note any regions where channels are consistently offset."
```

**Steps 4-5 — Capture + Molmo2 interprets**

Molmo2 receives: OLMo's vision instructions + auto-generated template (imaging
context block, spatial metadata, FOV).

Finding: "Overlay shows mostly yellow/white overlap indicating good alignment.
2-3 neurons in the lower-left quadrant show green-magenta separation suggesting
local misalignment..."

**Step 6 — OLMo reasons**

OLMo receives: the new finding + full investigation history
```
<think>
Good alignment overall (most neurons overlap → white), but some separation in
lower-left. This could be a local registration error. I should check if this
misalignment is consistent across z-depths or just at this z-slice.
</think>
```
Reasoning output: "Good alignment confirmed across most of the center FOV.
Local misalignment detected in lower-left quadrant (2-3 neurons show
green-magenta separation). Next: check whether this offset persists at
different z-depths to determine if it's a systematic or slice-specific error."

→ Stored in history. Next iteration begins at step 1 (OLMo still loaded).

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
  View: screenshot, xy, z=110µm, zoom=full
  Purpose: "Check cell density at volume center"
  FOV: x=[0..497]µm, y=[0..497]µm  (slice at z=110µm)
  Finding: "Dense field of ~20-30 fluorescent cell bodies..."

── Iteration 2 ──────────────────────────────────────────────
  View: scan, z_sweep, 50 frames
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

2. **Should count interpretation move to OLMo?**
   In v3, `ask_text()` interprets count results using the 7B model. In v4, this
   should definitely use OLMo — statistical reasoning about keyframe overlap,
   double-counting, and extrapolation is exactly where the 32B model shines.
   Recommendation: Yes, route count interpretation to OLMo. This means:
   - After Molmo2 does per-keyframe pointing, swap to OLMo for interpretation
   - The count action dispatch must handle this extra swap within the iteration
   - OLMo is already on GPU for the reason phase, so this is natural flow

3. **How to handle OLMo JSON parse failures?**
   The 32B model is much more reliable at JSON generation than the 7B, but failures
   still happen. Strategy: same retry with format reminder as v3. If retry fails,
   treat the text as a reason action.

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
