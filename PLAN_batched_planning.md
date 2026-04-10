# Future Plan — Batched Multi-View Planning

> **Prerequisite:** v4 (OLMo 3.1 32B Think model swap) must be complete and validated.
> This plan builds on the single-action OLMo agent loop established in v4.

## Goal

Change the agent loop from one action per iteration to N actions per iteration.
OLMo plans multiple views in a single call; Molmo2 interprets all of them before
OLMo reasons over the batch. This amortizes swap overhead from 2 swap cycles per
view down to 2 swap cycles per batch.

## Why

With v4's asymmetric swap (~40-70s per full cycle), each iteration pays a fixed cost
regardless of how many views it covers. Planning multiple views per OLMo call spreads
that cost across more data collection:

| Scenario             | Swaps per iter | 5-iter run | 10-iter run |
|----------------------|:--------------:|:----------:|:-----------:|
| v4 (single-action)   | 2              | 10         | 20          |
| **Batched (N views)** | **2**          | **10**     | **20**      |

Same swap count, but each iteration covers N views instead of 1. A 10-view investigation
that takes 10 iterations in v4 could finish in 3-4 batched iterations.

## Design (extracted from v4 planning)

### Batched action format

v4 decision output (single action):
```json
{"action": "screenshot", "view": {...}, "purpose": "..."}
```

Batched decision output (action batch):
```json
{"actions": [
    {"action": "screenshot", "view": {...}, "purpose": "..."},
    {"action": "screenshot", "view": {...}, "purpose": "..."},
    {"action": "scan", "scan": {...}, "purpose": "..."}
],
 "reasoning": "I want to compare these three regions because..."}
```

### Batched agent loop

```
Phase 2 loop (multiple views per iteration, 2 swaps per iteration):
  ┌─ OLMo phase (swap_to_olmo) ─────────────────────────────────┐
  │  ask_text_olmo: plan N actions from accumulated history       │
  │  → returns [{action, view, purpose}, ...]                     │
  └──────────────────────────────────────────────────────────────┘
  ┌─ Molmo phase (swap_to_molmo) ────────────────────────────────┐
  │  for each planned action:                                     │
  │    capture screenshot/scan                                    │
  │    ask_vision(Molmo2): interpret → finding                   │
  │  collect [finding_1, finding_2, ..., finding_N]              │
  └──────────────────────────────────────────────────────────────┘
  ┌─ OLMo phase (swap_to_olmo) ─────────────────────────────────┐
  │  ask_text_olmo: reason over N findings                        │
  │  → decide: plan more views, or answer                        │
  └──────────────────────────────────────────────────────────────┘
```

### Guardrails

- Max actions per batch: `max_actions_per_plan` config key (default 4, cap at 6)
- Total VRAM per batch must fit — large scans eat more VRAM than screenshots
- `answer` and `reason` actions are still single (they don't involve vision)
- If OLMo returns a single action, treat it as a batch of 1 — backward compatible

### Action schema changes for batched mode

Batched response format section added to the action schema:
```
──── Batched Response Format ─────────────────────────────────

For visual actions, wrap in an actions array:
  {"actions": [
     <action_1>,
     <action_2>,
     ...
   ],
   "reasoning": "<why these specific views, what you expect to learn>"}

For reason/answer, return the action directly (no array).
```

### Multi-view planning prompt

Replaces the single-action decision prompt when batching is enabled:

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

### Reasoning phase prompt (post-batch)

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

### Structured findings format (sub-views within iterations)

```
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

  Reasoning: "Most neurons are in the middle third of the volume..."
```

### Example: batched iteration walkthrough

**Setup:** 497x497x220µm, 2 image layers, question about alignment quality.

**Iteration 1 — OLMo plans batch of 3:**
```json
{"actions": [
  {"action": "screenshot", "view": {..., "z": 110}, "show": [1],
   "purpose": "View fixed channel alone at center"},
  {"action": "screenshot", "view": {..., "z": 110}, "show": [2],
   "purpose": "View moving channel alone at same position"},
  {"action": "screenshot", "view": {..., "z": 110}, "show": [1, 2],
   "purpose": "Overlay both channels to see alignment"}
], "reasoning": "Compare channels individually then overlaid..."}
```

**Molmo phase:** Captures and interprets all 3 screenshots without swapping.

**OLMo reasons:** Receives 3 findings, identifies misalignment in lower-left,
plans next batch targeting that region.

---

## Open Design Questions

These must be resolved before implementing batched planning:

### OD1. History structure for batched actions (from gap analysis G1)

Does a batch of 3 actions become:
- **Option A:** One history entry with a list of sub-findings (matches §5.3 format)
- **Option B:** Three separate history entries tagged to the same iteration

Option A is cleaner for `format_structured_findings()` and compression. Option B is
simpler for `count_prior_matches()` and backward compatibility.

### OD2. `validate_action()` for batched format (from gap analysis G2)

Each action in the `{"actions": [...]}` array needs zoom resolution, coordinate
clamping, and `_resolve_show()`. Options:
- Loop over batch, call existing `validate_action()` per action
- New `validate_action_batch()` wrapper

### OD3. Duplicate detection in batched mode (from gap analysis G3)

How does `count_prior_matches()` / `_action_fingerprint()` work for batches?
- Fingerprint whole batch as a unit?
- Fingerprint individual actions within the batch?
- Both? (block individual duplicate actions while allowing the batch)

How does `max_consecutive_duplicates` translate to batched iterations?

### OD4. `count` action dispatch in batched Molmo phase (from gap analysis G8)

`count` requires a multi-step inner loop (scan → per-keyframe pointing → OLMo
interpretation). This dispatch path is fundamentally different from screenshot/scan.
If count interpretation should go to OLMo (per deep-dive §14 Q3), this means a
brief OLMo swap mid-batch. Options:
- Defer count interpretation to the post-batch OLMo reasoning phase
- Allow a mini-swap for count interpretation within the batch

### OD5. Scan browser session reuse (from gap analysis G7)

`execute_scan()` creates a new Playwright browser per call. At 3-4 scans per batch
this overhead may be significant. Should we refactor to reuse a browser session across
the batch?

### OD6. Counter behavior (`scan_count`, `screenshot_id`)

These increment globally. In a batch of 3 screenshots, they should increment within
the batch. Verify no downstream logic assumes counter == iteration number.

## Implementation checklist (batch-specific)

New/modified functions for batching:

1. `parse_action_batch(model_output)` — parse `{"actions": [...]}` alongside single `{"action": ...}`
2. `validate_action_batch(actions, volume_info)` — validate each action in a batch
3. `format_structured_findings(history)` — render sub-views within iterations
4. Updated `build_decision_prompt()` — include batched action format in schema
5. Updated `run_agent()` inner loop — batch execution + batch reasoning
6. Updated `save_prompt_templates()` — reflect batched prompt formats
7. Config: `max_actions_per_plan` (default 4, cap 6)

## Files affected

All changes are in `code/molmo_glancer.py` (agent loop, prompts, parsing) —
no new modules needed. `gpu_config.py` gets one config key (`max_actions_per_plan`).
