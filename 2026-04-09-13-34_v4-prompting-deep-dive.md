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

| Prompt type | max_new_tokens | Rationale |
|---|---|---|
| Strategy planning | 2048 | Needs space for <think> + action batch |
| Action decision | 2048 | <think> reasoning + JSON actions array |
| Reasoning step | 3072 | Longer chains for synthesis |
| Final answer | 4096 | Full synthesis, may have long <think> |
| Count interpret | 2048 | Statistical reasoning over detection data |

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

1. `strip_think_tokens(text)` — separate `<think>` blocks from response
2. `ask_text_olmo(manager, prompt, max_new_tokens)` — OLMo generation with think handling
3. `build_olmo_system_prompt(volume_info)` — system prompt with reasoning anchors
4. `build_olmo_decision_prompt(question, volume_info, history, config, iteration)` — replaces `build_decision_prompt` for OLMo calls
5. `build_vision_interpret_prompt(action, question, volume_info)` — generates Molmo2 interpret prompts from action metadata
6. `format_structured_findings(history)` — replaces `format_history_entry` with richer format
7. `compress_findings(history, max_entries)` — token-aware history compression
8. `parse_action_batch(model_output)` — extends `parse_action` to handle `{"actions": [...]}`
9. `estimate_tokens(text)` — lightweight token counting for budget enforcement

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
    "olmo_max_new_tokens_decision": 2048,
    "olmo_max_new_tokens_synthesis": 4096,
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
