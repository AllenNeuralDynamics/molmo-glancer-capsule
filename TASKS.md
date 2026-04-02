# v3 Implementation Tasks

Source of truth: v3_plan.md
Status key: [ ] todo, [x] done, [~] in progress, [-] skipped

---

## Phase 1: Infrastructure

### 1.1 GPU config module
- [x] Create `code/gpu_config.py`
- [x] `detect_gpu_profile()` — read VRAM, return "compact" / "full"
- [x] `GPU_PROFILES` dict — quantization, max_image_side, max_crops, max_scan_frames, max_agent_iterations, max_context_tokens
- [x] `load_model(checkpoint, config)` — fp16 or 4-bit based on profile
- [x] `get_vram_usage()` — returns allocated/reserved/free/total
- [x] Verify model loads on T4 (4-bit) without crash
Files: code/gpu_config.py

### 1.2 Volume metadata discovery
- [x] Create `code/volume_info.py`
- [x] `VolumeInfo` dataclass — shape, voxel_scales, axis_names, layers, canonical_factors, anisotropy_ratio
- [x] `discover_volume(ng_state)` — parse NG state JSON for dimensions, layers
- [x] `read_shape_from_source(source_url)` — read zarr .zarray for shape, returns **physical extent** (µm) not raw pixels
- [x] OME-Zarr multiscale transforms: reads axis names + scale from `.zattrs` to compute physical extent
- [x] `compute_fov()`, `compute_visible_window()`, `format_fov_feedback()`
- [x] Test with example_ng_link.txt data source (zarr read) — confirmed 497×497×220 µm, center at 248.5
Files: code/volume_info.py

### 1.3 Clean state builder
- [x] Create `code/visual_capture.py`
- [x] `build_clean_state(base_state, view_spec)` — applies full view spec + overlay hiding
- [x] Set showAxisLines=false, showScaleBar=true, showDefaultAnnotations=true (yellow data border)
- [x] Set crossSectionBackgroundColor="#000000"
- [x] Set selectedLayer.visible=false, statistics.visible=false
- [x] Apply crossSectionScale, projectionScale, projectionOrientation from view spec
- [x] Apply layerVisibility, shaderRange from view spec
- [x] Clamp position to volume bounds
- [x] Auto-fit zoom (2x fit_scale) when no crossSectionScale specified
- [x] Test: generated URL opens in browser with clean rendering, data centered
Files: code/visual_capture.py

### 1.4 Clean screenshot capture
- [x] `capture_screenshot(page, state, config, screenshot_id)` in visual_capture.py
- [x] CSS injection to hide toolbar, layer panel, side panel, statistics
- [x] `viewer.isReady()` polling with `networkidle` + timeout (fallback to short sleep)
- [x] Canvas-only screenshot via `page.locator("canvas").first`
- [x] Fixed 1024×1024 square viewport
- [x] Test: screenshot has data centered with yellow border and scale bar
Files: code/visual_capture.py

### 1.5 Update dev startup
- [x] Update `code/_dev_startup.sh` with new deps: zarr, s3fs, imageio[ffmpeg]
- [x] Verify all deps install cleanly on T4
Files: code/_dev_startup.sh

### 1.6 Shell entry point
- [x] Create `code/run_v3` — sets env vars (PLAYWRIGHT_BROWSERS_PATH, HF_HOME), runs molmo_glancer.py, tees to output.log
Files: code/run_v3

### 1.7 Integration test
- [x] Model loads on T4 (4-bit) — 5.2 GB VRAM
- [x] Screenshot of example_ng_link is clean
- [x] NG state roundtrip: build_clean_state → to_url → screenshot matches expected view

---

## Phase 2: Agent Loop

### 2.1 Action schemas
- [x] Define action JSON schemas (screenshot, scan, think, answer)
- [x] `parse_action(model_output)` — extract JSON from model text, validate action type and required fields
- [x] Handle malformed JSON — retry once with format reminder
Files: code/molmo_glancer.py

### 2.2 Prompt construction
- [x] `build_decision_prompt(question, volume_info, history, config)` — constructs user message
- [x] No system role — all instructions in user message
- [x] Dynamic action schema with volume-appropriate example coordinates
- [x] Named zoom levels ("wide", "full", "region", "neurons", "single-cell") instead of raw floats
- [x] Neuron size reference (30µm) in volume info and interpretation prompts
- [x] Scale bar kept visible for physical size reference
Files: code/molmo_glancer.py

### 2.3 Agent loop skeleton
- [x] Main loop: prompt → model.generate → parse_action → execute → append to context → repeat
- [x] `ask_text(model, processor, prompt)` — text-only model call
- [x] `ask_vision(model, processor, image, prompt)` — image+text call (profile-aware downscaling)
- [x] `ask_scan(model, processor, frames, prompt)` — video call with synthetic VideoMetadata (fps=0.5 to prevent frame dropping)
- [x] Context accumulation: list of {action, finding, fov_metadata} dicts
- [x] Loop terminates on "answer" action or max iterations
- [x] First-look screenshot of user's original NG view before loop starts
Files: code/molmo_glancer.py

### 2.4 FOV computation
- [x] `compute_fov(scale, canonical_factors)` — returns visible voxel extent
- [x] `compute_visible_window(position, scale, canonical_factors)` — returns [min, max] per axis
- [x] Append FOV metadata to each finding in context
Files: code/volume_info.py

### 2.5 Guardrails
- [x] Max iterations from GPU profile config
- [x] Duplicate detection for both screenshots AND scans (fingerprint-based)
- [x] Position clamping: reject/clamp positions outside volume bounds
- [x] Scale validation: reject negative or zero crossSectionScale
- [x] Format retry: if model output isn't valid JSON, re-prompt once
- [x] Forced answer: at max iterations, force synthesis with accumulated findings
- [x] Zoom name resolution: model sends "zoom": "neurons", system resolves to concrete scale
Files: code/molmo_glancer.py

### 2.6 Output saving
- [x] Save screenshots to results/screenshots/view_NNN.png
- [x] Save findings.json — per-iteration action + finding + FOV
- [x] Save token_usage.json — per-iteration token counts
- [x] Save answer.txt — final answer
- [x] Save scan videos to results/scans/scan_NNN.mp4 (with libx264 codec, gif fallback)
Files: code/molmo_glancer.py

### 2.7 Integration test
- [x] Full loop runs on T4 with text-only + screenshot + scan actions
- [x] Loop terminates (answer or max iterations)
- [x] Outputs written to results/
- [x] Action JSON parses correctly from model output
Files: code/molmo_glancer.py

---

## Phase 3: Scans

### 3.1 Video API validation
- [x] PIL Image list works with `{"type": "video", "video": frames}`
- [x] VideoMetadata with synthetic fps=0.5 required (FPS assertion fix)
- [x] Low fps prevents frame sub-sampling (all frames kept)
- [-] numpy array / mp4 path fallback — not needed, PIL list works
Files: code/molmo_glancer.py

### 3.2 Scan executor
- [x] `execute_scan(page, base_state, scan_spec, volume_info, config, scan_id)` in visual_capture.py
- [x] `generate_scan_positions(scan_spec, volume_info)` — np.linspace with clamping
- [x] Hash-fragment update per frame (faster than page.goto)
- [x] viewer.isReady() polling per frame (3s timeout for cached frames)
- [x] Canvas screenshot per frame → list of PIL Images
- [x] `save_scan_video(frames, scan_id)` — write mp4 via imageio (libx264 codec)
- [x] Progress logging every 5 frames
Files: code/visual_capture.py

### 3.3 Video inference
- [x] `ask_scan(model, processor, frames, prompt)` in molmo_glancer.py
- [x] VideoMetadata with fps=0.5 ensures all frames used
- [x] Returns text + token counts
Files: code/molmo_glancer.py

### 3.4 Scan types
- [x] z_sweep — vary Z position (via start/end position interpolation)
- [x] x_pan / y_pan — vary X or Y position (via start/end)
- [ ] rotation — vary projectionOrientation (interpolate quaternions)
- [-] zoom_ramp — replaced by named zoom levels
- [x] freeform — arbitrary start/end position vector
Files: code/visual_capture.py

### 3.5 Wire scans into agent loop
- [x] Agent loop handles "scan" action: execute_scan → ask_scan → append finding
- [x] Duplicate scan detection prevents repeated identical sweeps
Files: code/molmo_glancer.py

### 3.6 Integration test
- [x] Scan produces frames on T4
- [x] Video saved to results/scans/
- [x] Model interprets video and produces text finding
- [~] Full loop with scan + screenshot actions works end-to-end (working, tuning zoom/position)

---

## Phase 4: Context & Scale

### 4.1 Context compression
- [x] Sliding window in `build_decision_prompt` — full detail for last 10, summary for older
- [ ] Summary can be model-generated (via think action) or simple truncation
Files: code/molmo_glancer.py

### 4.2 Scale-aware prompting
- [x] Volume metadata in physical units (µm) in decision prompt
- [x] FOV feedback after each action
- [x] Named zoom levels replace raw crossSectionScale values
- [x] Neuron size reference (30µm) in prompts
- [x] Scale bar visible in screenshots
Files: code/molmo_glancer.py, code/volume_info.py

### 4.3 Detail mode (L40S only)
- [ ] Verify max_crops override via processor kwargs
- [ ] "screenshot" action can specify detail=true → max_crops=24
- [ ] Only enabled in "full" GPU profile
Files: code/molmo_glancer.py, code/gpu_config.py

### 4.4 VRAM monitoring
- [x] `get_vram_usage()` — returns allocated/reserved/free
- [x] VRAM logged at each iteration
- [ ] Trigger context compression if approaching limit
Files: code/gpu_config.py, code/molmo_glancer.py

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
- [~] Run on example_ng_link.txt with neuron counting question (running, iterating on quality)
- [ ] Run on example_r2r_ng_link.txt
- [ ] Run on thyme_r2r_ng_link.txt
- [ ] Verify: model uses diverse actions (scan + screenshot + think)
- [ ] Verify: model uses diverse view specs (varied x, y, z, layout, zoom)
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

---

## Bugs Fixed During Development

- [x] `total_mem` → `total_memory` attribute fix in gpu_config.py
- [x] imageio mp4 codec: explicit `codec="libx264"` required (pyav auto-detect failed)
- [x] Molmo2 video processor FPS assertion: `VideoMetadata(fps=0.5)` required for pre-decoded frames
- [x] Physical vs pixel coordinates: zarr shape must be multiplied by OME-Zarr scale transforms to get NG position units
- [x] Data not centered: position must be in physical units (µm), not voxel indices
- [x] Black background hiding dark data: kept black bg but added scale bar + yellow data border for context
- [x] Scan frame timeouts too long: tightened to 3s for cached frames, `networkidle` for initial load
- [x] Model copying hardcoded example coordinates: made action schema examples dynamic from volume
- [x] Duplicate scan loop: extended fingerprint-based duplicate detection to cover scans, not just screenshots
