# v3 Implementation Tasks

Source of truth: v3_plan.md
Status key: [ ] todo, [x] done, [~] in progress, [-] skipped

---

## Phase 1: Infrastructure

### 1.1 GPU config module
- [ ] Create `code/gpu_config.py`
- [ ] `detect_gpu_profile()` — read VRAM, return "compact" / "medium" / "full"
- [ ] `GPU_PROFILES` dict — quantization, max_image_side, max_crops, max_scan_frames, max_agent_iterations, max_context_tokens
- [ ] `load_model(checkpoint, config)` — fp16 or 4-bit based on profile
- [ ] Verify model loads on T4 (4-bit) without crash
Files: code/gpu_config.py

### 1.2 Volume metadata discovery
- [ ] Create `code/volume_info.py`
- [ ] `VolumeInfo` dataclass — shape, voxel_scales, axis_names, layers, canonical_factors, anisotropy_ratio
- [ ] `discover_volume(ng_state)` — parse NG state JSON for dimensions, layers
- [ ] `read_shape_from_source(source_url)` — read zarr .zarray for shape
- [ ] Test with example_ng_link.txt data source
Files: code/volume_info.py

### 1.3 Clean state builder
- [ ] Create `code/visual_capture.py`
- [ ] `build_clean_state(base_state, view_spec)` — applies full view spec + overlay hiding
- [ ] Set showAxisLines=false, showScaleBar=false, showDefaultAnnotations=false
- [ ] Set crossSectionBackgroundColor="#000000"
- [ ] Set selectedLayer.visible=false, statistics.visible=false
- [ ] Apply crossSectionScale, projectionScale, projectionOrientation from view spec
- [ ] Apply layerVisibility, shaderRange from view spec
- [ ] Clamp position to volume bounds
- [ ] Test: generated URL opens in browser with clean rendering
Files: code/visual_capture.py

### 1.4 Clean screenshot capture
- [ ] `capture_screenshot(page, url)` in visual_capture.py
- [ ] CSS injection to hide toolbar, layer panel, side panel, statistics
- [ ] `viewer.isReady()` polling with timeout (fallback to sleep)
- [ ] Canvas-only screenshot via `page.locator("canvas").first`
- [ ] Fixed 1024×1024 square viewport
- [ ] Test: screenshot has no UI chrome, 100% data pixels
Files: code/visual_capture.py

### 1.5 Update dev startup
- [ ] Update `code/_dev_startup.sh` with new deps: zarr, s3fs, imageio
- [ ] Verify all deps install cleanly on T4
Files: code/_dev_startup.sh

### 1.6 Shell entry point
- [ ] Create `code/run_v3` — sets env vars (PLAYWRIGHT_BROWSERS_PATH, HF_HOME), runs molmo_glancer.py, tees to output.log
Files: code/run_v3

### 1.7 Integration test
- [ ] Model loads on T4 (4-bit)
- [ ] Screenshot of example_ng_link is clean (no UI chrome)
- [ ] NG state roundtrip: build_clean_state → to_url → screenshot matches expected view

---

## Phase 2: Agent Loop

### 2.1 Action schemas
- [ ] Define action JSON schemas (screenshot, scan, think, answer)
- [ ] `parse_action(model_output)` — extract JSON from model text, validate action type and required fields
- [ ] Handle malformed JSON — retry once with format reminder
Files: code/molmo_glancer.py

### 2.2 Prompt construction
- [ ] `build_decision_prompt(question, volume_info, history, config)` — constructs user message with instructions, volume info, coverage summary, recent findings
- [ ] No system role — all instructions in user message
- [ ] Include action schemas with examples in prompt
- [ ] Include FOV explanation ("at scale S, you see S·1024 × S·1024 voxels")
Files: code/molmo_glancer.py

### 2.3 Agent loop skeleton
- [ ] Main loop: prompt → model.generate → parse_action → execute → append to context → repeat
- [ ] `ask_text(model, processor, prompt)` — text-only model call (reuse from v2, adapt)
- [ ] `ask_vision(model, processor, image, prompt)` — image+text call (reuse from v2, remove downscaling on L40S)
- [ ] Context accumulation: list of {action, finding, fov_metadata} dicts
- [ ] Loop terminates on "answer" action or max iterations
Files: code/molmo_glancer.py

### 2.4 FOV computation
- [ ] `compute_fov(scale, canonical_factors)` — returns visible voxel extent
- [ ] `compute_visible_window(position, scale, canonical_factors)` — returns [min, max] per axis
- [ ] Append FOV metadata to each finding in context
Files: code/molmo_glancer.py or code/volume_info.py

### 2.5 Guardrails
- [ ] Max iterations from GPU profile config
- [ ] Duplicate detection: skip if new view overlaps >80% with prior view (same layout, position within FOV, similar scale)
- [ ] Position clamping: reject/clamp positions outside volume bounds
- [ ] Scale validation: reject negative or zero crossSectionScale
- [ ] Format retry: if model output isn't valid JSON, re-prompt once
- [ ] Forced answer: at max iterations, force synthesis with accumulated findings
Files: code/molmo_glancer.py

### 2.6 Output saving
- [ ] Save screenshots to results/screenshots/view_NNN.png
- [ ] Save findings.json — per-iteration action + finding + FOV
- [ ] Save ng_states.json — NG state per screenshot
- [ ] Save token_usage.json — per-iteration token counts
- [ ] Save answer.txt — final answer
Files: code/molmo_glancer.py

### 2.7 Integration test
- [ ] Full loop runs on T4 with text-only + screenshot actions
- [ ] Loop terminates (answer or max iterations)
- [ ] Outputs written to results/
- [ ] Action JSON parses correctly from model output

---

## Phase 3: Scans

### 3.1 Video API validation (DO THIS FIRST — gates rest of phase)
- [ ] Create 5 dummy 1024×1024 PIL Images
- [ ] Pass as `{"type": "video", "video": frames}` to processor via apply_chat_template
- [ ] Confirm model generates a response
- [ ] Check: are all 5 frames used? Or does do_sample_frames drop some?
- [ ] If PIL list doesn't work: try numpy array (N, H, W, 3)
- [ ] If numpy doesn't work: write to mp4 with imageio, pass file path
- [ ] Document the working approach
Files: test script (throwaway)

### 3.2 Scan executor
- [ ] `execute_scan(page, base_state, scan_spec, volume_info, config, scan_id)` in visual_capture.py
- [ ] `generate_scan_positions(scan_spec, volume_info)` — np.linspace with clamping
- [ ] Hash-fragment update per frame (faster than page.goto)
- [ ] viewer.isReady() polling per frame
- [ ] Canvas screenshot per frame → list of PIL Images
- [ ] `save_scan_video(frames, scan_id)` — write mp4 via imageio
Files: code/visual_capture.py

### 3.3 Video inference
- [ ] `ask_scan(model, processor, frames, prompt)` in molmo_glancer.py
- [ ] Uses the working input format from 3.1
- [ ] Handle frame sub-sampling override if needed (do_sample_frames=false or max_fps override)
- [ ] Return text + token counts
Files: code/molmo_glancer.py

### 3.4 Scan types
- [ ] z_sweep — vary Z position
- [ ] x_pan / y_pan — vary X or Y position
- [ ] rotation — vary projectionOrientation (interpolate quaternions)
- [ ] zoom_ramp — vary crossSectionScale
- [ ] freeform — arbitrary start/end position vector
Files: code/visual_capture.py

### 3.5 Wire scans into agent loop
- [ ] Agent loop handles "scan" action: execute_scan → ask_scan → append finding
- [ ] Scan findings include frame count, axis swept, range covered
Files: code/molmo_glancer.py

### 3.6 Integration test
- [ ] Scan produces frames on T4
- [ ] Video saved to results/scans/
- [ ] Model interprets video and produces text finding
- [ ] Full loop with scan + screenshot actions works end-to-end

---

## Phase 4: Context & Scale

### 4.1 Context compression
- [ ] `build_history(findings, max_recent, max_summary_tokens)` — sliding window
- [ ] Full detail for last N findings
- [ ] Compressed summary for older findings
- [ ] Summary can be model-generated (via think action) or simple truncation
Files: code/molmo_glancer.py

### 4.2 Scale-aware prompting
- [ ] Volume metadata formatted clearly in decision prompt
- [ ] FOV feedback after each action ("visible window: x=[min..max], y=[min..max]")
- [ ] Explain crossSectionScale in prompt ("at scale S, you see S·1024 voxels")
Files: code/molmo_glancer.py

### 4.3 Detail mode (L40S only)
- [ ] Verify max_crops override via processor kwargs
- [ ] "screenshot" action can specify detail=true → max_crops=24
- [ ] Only enabled in "full" GPU profile
Files: code/molmo_glancer.py, code/gpu_config.py

### 4.4 VRAM monitoring
- [ ] `get_vram_usage()` — returns allocated/reserved/free
- [ ] Check budget before each action
- [ ] Trigger context compression if approaching limit
Files: code/gpu_config.py or code/molmo_glancer.py

### 4.5 L40S integration test
- [ ] Full session with fp16, no quantization
- [ ] Long scan (100+ frames) works
- [ ] Detail mode (max_crops=24) works
- [ ] 15+ iteration session completes without OOM

---

## Phase 5: Validation

### 5.1 Token calibration
- [ ] Log actual vision token counts for 1024×1024 images at max_crops=4, 8, 24
- [ ] Log actual video token counts for 10, 50, 100 frame scans
- [ ] Update VRAM budget estimates in v3_plan.md with real numbers

### 5.2 Functional validation
- [ ] Run on example_ng_link.txt with neuron counting question
- [ ] Run on example_r2r_ng_link.txt
- [ ] Run on thyme_r2r_ng_link.txt
- [ ] Verify: model uses diverse actions (scan + screenshot + think)
- [ ] Verify: model uses diverse view specs (varied x, y, z, layout, scale)
- [ ] Verify: model terminates with answer, not forced at max iterations

### 5.3 v2 vs v3 comparison
- [ ] Same question + data on both v2 and v3
- [ ] Compare: number of unique views (deduplicated)
- [ ] Compare: answer quality (manual assessment)
- [ ] Compare: total tokens used
- [ ] Compare: wall-clock time

---

## Cleanup & Polish (after validation)

- [ ] Update CLAUDE.md for v3 architecture
- [ ] Update README.md
- [ ] Archive v2 code files (run_capsule.py, run.sh, cleanup.sh)
- [ ] Remove any dead code paths
