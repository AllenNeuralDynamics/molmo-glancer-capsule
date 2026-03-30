# Neuroglancer Navigation Overhaul — Investigation Findings

## Why Zoom Failed (Root Cause)

The CTRL+scroll zoom code in `patch_env_for_ng` was **never triggered** in the last run.
The model never output `delta_x != 0` in any `scroll_at` action — it always used `delta_x=0.0`.

Two reasons:
1. The last run used `--mode registration`, which doesn't ask for zoom at all. Only `registration_large` does.
2. Even in `registration_large`, the `delta_x` convention is non-standard and unclear to the model. The model was trained to scroll, not to encode zoom intent via the delta_x field.

Additionally, there is a **known Playwright limitation**: `page.keyboard.down("Control") + page.mouse.wheel()` does NOT reliably pass the modifier key to the WheelEvent. Playwright's `mouse.wheel()` fires a WheelEvent without reading current keyboard state. This means even if the model used delta_x, the zoom would likely fail.

---

## MolmoWeb Action Space — Critical Details

**File:** `/root/capsule/code/lib/molmoweb/agent/multimodal_agent.py`

### Coordinate system
- Model outputs **percentages (0–100)** for all x/y/delta coordinates
- Converted to pixels by `_pct_to_px(pct, dim)` and `_pct_to_coord(pct, dim)`
- Viewport: **1280×720**
- Example: `x=50, y=50` → pixel `(640, 360)` (center)
- Example: `delta_y=100` → `720px` scroll delta

### Prompt structure (what model sees)
```
molmo_web_think:
# GOAL
{task description}

# PREVIOUS STEPS
## Step N
THOUGHT: ...
ACTION: ...

# CURRENTLY ACTIVE PAGE
Page 0: {page_title} | {page_url}

# NEXT STEP
```

The model has **no formal action schema description** injected — it infers available actions from training. The `system_message` is just a string prefix `"molmo_web_think"`.

### KeyboardPress ALLOWED_KEYS (hardcoded whitelist)
Only these keys are allowed — anything else becomes `ReportInfeasible`:
```
"Enter", "Escape", "Backspace", "Tab",
"ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
"ControlOrMeta+a", "ControlOrMeta+c", "ControlOrMeta+v",
"F5"
```

**Critical implication**: NG's native keyboard shortcuts (`,` for Z-, `.` for Z+, `Space` for layout cycle) are **BLOCKED** by this whitelist. The model cannot use them without modifying ALLOWED_KEYS or intercepting them in `_ng_step`.

### All available action names model can output
`click`, `dblclick`, `mouse_click`, `hover_at`, `drag_and_drop`, `mouse_drag_and_drop`,
`scroll`, `scroll_at`, `type`, `keyboard_type`, `keypress`, `keyboard_press`,
`gemini_type_text_at`, `goto`, `send_msg_to_user`, `browser_nav`, `noop`, `report_infeasible`

Custom action names (e.g. `ng_navigate`) would fall through to `ReportInfeasible` unless `convert_action_json_to_action_obj` is extended.

### max_past_steps = 3 (default from MolmoWeb client)
The model only sees the last 3 steps of action history. For long explorations, context of earlier steps is lost.

---

## Neuroglancer UI Layout (1280×720 viewport, "xy" layout)

When Neuroglancer loads in `xy` layout (single full-screen XY panel):

```
┌──────────────────────────────────────────────┐
│ [X: 1488 Y: 1905 Z: 896] [t: 0.5]          │  ← top-left: clickable coordinate display
│                                              │  ← click → type "x y z" → Enter to jump
│                                              │
│           MAIN CANVAS (XY slice)             │
│                                              │
│                                              │
│                                              │
├──────────────────────────────────────────────┤
│ [layer1 ▼] [layer2 ▼]   [?] [Share] [⚙]   │  ← bottom bar: layer list + controls
└──────────────────────────────────────────────┘
```

**Coordinate display (top-left)**:
- Shows current voxel position: `X: 1488 Y: 1905 Z: 896`
- **Clickable** — clicking opens an editable text field
- Type new coordinates separated by spaces: `"1200 1800 500"` → Enter → jumps there
- This is the most reliable way for the model to navigate to specific positions

**Layer panel (bottom)**:
- Shows layer names as clickable tabs
- Click layer name → toggles visibility
- Double-click → open layer settings (opacity, shader)

**No visible zoom UI** — zoom is only via CTRL+scroll or URL state

---

## Neuroglancer Keyboard & Mouse Controls (Full Reference)

### Mouse (what works natively in NG canvas)
| Action | Effect |
|--------|--------|
| Left click + drag | Pan XY (translate view in X/Y) |
| Right click | Recenter to clicked position |
| Mouse wheel (no modifier) | Advance/retreat Z slice (1 event = 1 slice) |
| CTRL + mouse wheel | Zoom in/out |
| Shift + mouse wheel | Z-scroll at 10× speed |

### Keyboard (native NG shortcuts — currently BLOCKED by ALLOWED_KEYS)
| Key | Effect |
|-----|--------|
| `.` (period) | Advance Z by 1 slice |
| `,` (comma) | Retreat Z by 1 slice |
| `Space` | Cycle layout modes (xy → xz → yz → 3d → 4panel) |
| `h` or `?` | Show help panel |
| `Z` | Snap to nearest axis-aligned orientation |

### URL State Fields for Programmatic Navigation
| Field | Type | Effect |
|-------|------|--------|
| `position` | [x, y, z] float array | Camera center in voxel coordinates |
| `crossSectionScale` | float | Zoom level for 2D panels (larger = more zoomed out) |
| `projectionScale` | int | Scale for 3D view |
| `layout` | string | Panel layout: "xy", "xz", "yz", "3d", "4panel" |

**zoom relationship**: `crossSectionScale ≈ 9` in the thyme link (overview level). To zoom in by 2×, halve it to `≈ 4.5`. To zoom out, double it.

---

## Architecture Options for Overhaul

### Option A: JavaScript Control Overlay (RECOMMENDED — highest reliability)

Inject a floating HTML/JS overlay into the NG page with labeled buttons visible to the model. Each button fires synthetic DOM events directly on the canvas — bypassing all Playwright keyboard/modifier limitations.

```javascript
// Inject after page loads
const overlay = document.createElement('div');
overlay.id = 'ng-hud';
overlay.style = 'position:fixed;top:8px;right:8px;z-index:9999;...';
overlay.innerHTML = `
  <div class="ng-hud-title">NG NAV</div>
  <button id="ng-zi">ZOOM IN</button>
  <button id="ng-zo">ZOOM OUT</button>
  <button id="ng-zp">Z+20</button>
  <button id="ng-zm">Z-20</button>
`;
document.body.appendChild(overlay);

function ngZoom(direction, ticks=5) {
  // Get the canvas — NG renders to the first canvas
  const canvas = document.querySelector('canvas');
  for (let i = 0; i < ticks; i++) {
    canvas.dispatchEvent(new WheelEvent('wheel', {
      deltaY: direction > 0 ? 100 : -100,
      ctrlKey: true,
      bubbles: true,
      cancelable: true
    }));
  }
}

function ngScrollZ(direction, ticks=20) {
  const canvas = document.querySelector('canvas');
  for (let i = 0; i < ticks; i++) {
    canvas.dispatchEvent(new WheelEvent('wheel', {
      deltaY: direction > 0 ? 100 : -100,
      bubbles: true,
      cancelable: true
    }));
  }
}

document.getElementById('ng-zi').onclick = () => ngZoom(-1, 5);  // zoom in = negative deltaY w/ ctrl
document.getElementById('ng-zo').onclick = () => ngZoom(1, 5);
document.getElementById('ng-zp').onclick = () => ngScrollZ(1, 20);
document.getElementById('ng-zm').onclick = () => ngScrollZ(-1, 20);
```

**Key insight**: `WheelEvent` with `ctrlKey: true` in the constructor is the correct way to synthesize CTRL+scroll — not holding keyboard modifier and then calling `mouse.wheel()`.

**Must re-inject after every `goto()`** since page navigation destroys injected JS/DOM.

### Option B: Fix CTRL+Scroll via page.evaluate()

Replace current `keyboard.down("Control") + mouse.wheel()` with:
```python
page.evaluate("""(ticks, delta) => {
    const canvas = document.querySelector('canvas');
    for (let i = 0; i < ticks; i++) {
        canvas.dispatchEvent(new WheelEvent('wheel', {
            deltaY: delta, ctrlKey: true, bubbles: true
        }));
    }
}""", zoom_ticks, zoom_delta)
```

This is the minimal fix for zoom. No overlay needed, but model still uses the awkward `delta_x` convention.

### Option C: Extend ALLOWED_KEYS + Use Native NG Shortcuts

Add NG keys to the whitelist in `multimodal_agent.py`:
```python
ALLOWED_KEYS = [
    ...,
    ".",          # NG: advance Z
    ",",          # NG: retreat Z
    "Space",      # NG: cycle layout
]
```

Then in `_ng_step`, intercept these and fire them N times (or handle natively). The model may not spontaneously use these without prompting, but can be instructed to.

### Option D: URL State Navigation (100% reliable, zero UI interaction)

For position jumps and zoom: parse current URL → modify state JSON → `goto(new_url)`.

```python
def ng_url_advance_z(current_url: str, voxels: int) -> str:
    """Modify NG URL to shift Z position by N voxels."""
    state = parse_ng_url(current_url)
    state["position"][2] += voxels
    return encode_ng_url(current_url, state)

def ng_url_set_zoom(current_url: str, scale: float) -> str:
    """Set crossSectionScale directly."""
    state = parse_ng_url(current_url)
    state["crossSectionScale"] = scale
    return encode_ng_url(current_url, state)
```

Could expose as `keyboard_press("F1")` → zoom in, `keyboard_press("F2")` → zoom out, by intercepting F-keys in `_ng_step` and doing URL navigation. Or add to `ALLOWED_KEYS` to allow F1/F2.

---

## Recommended Implementation Plan

### Phase 1: Core Reliability Fixes (immediate)
1. **Fix zoom**: Replace `keyboard.down("Control") + mouse.wheel()` with `page.evaluate(WheelEvent ctrlKey:true)` in `_ng_step`
2. **Add NG keys to ALLOWED_KEYS**: Add `"."`, `","` so model can use native Z-navigation shortcuts
3. **Intercept `"."` / `","` in `_ng_step`**: Fire N times (scroll_ticks) for multi-slice advance

### Phase 2: JS Overlay (high autonomy impact)
4. **Add `inject_ng_overlay(page)`** function to `ng_explore.py`
5. **Call inject after initial goto and re-call after any subsequent goto** in `_ng_step`
6. **Update task prompts** to tell model about the overlay buttons and how to click them

### Phase 3: Prompt Engineering (model knowledge)
7. **Add NG UI guide section** to all task prompts:
   - Location of coordinate display (top-left ~5%, ~3% of viewport)
   - How to click it and type coordinates to jump
   - Describe overlay buttons and their purpose
   - Describe layer list location

### Phase 4: Optional Enhancements
8. **URL state navigation utility** (`ng_parse_url`, `ng_modify_url`) for programmatic jumps
9. **Custom `NgJumpAction`** if model fine-tuning is planned
10. **Extend `max_past_steps`** beyond 3 for longer exploration context

---

## Files to Modify

| File | Changes |
|------|---------|
| `code/ng_explore.py` | inject_ng_overlay(), fix zoom in _ng_step, update prompts |
| `code/lib/molmoweb/agent/multimodal_agent.py` | Extend ALLOWED_KEYS with `.`, `,`, `F1`, `F2` |

Optionally:
| `code/lib/molmoweb/agent/actions.py` | Add NgJumpAction (low priority) |
| `code/lib/molmoweb/utils/envs/action_executor.py` | Handle NgJumpAction (low priority) |

---

## Key Numbers (for overlay button placement)
- Viewport: 1280×720
- Overlay target position: top-right, ~x=1100-1250, y=10-200 (avoids coordinate display at top-left)
- Model outputs percentages: x≈86-98%, y≈1-28%
- Coordinate display: approximately x=0-300px (top-left), ~3% from top
