---
name: molmo-glancer project
description: Core facts about the molmo-glancer project — repos, models, architecture, key decisions
type: project
---

# Project: molmo-glancer

Goal: autonomous Neuroglancer question answering using a local AI stack (no external APIs).

## Repos (all in /code/)
- `molmo-glancer` — Python library (AIND template). Orchestration, CV analysis, browser automation, pipeline.
- `molmo-glancer-capsule` — CodeOcean capsule / dev workspace / user entry point. Mirrors /capsule at runtime.
- `neuroglancer-chat` — Fork: github.com/seanmcculloch/neuroglancer-chat. Branch `olmo-local` adds OLMo adapter.
- `molmoweb` — Upstream AllenAI MolmoWeb, unmodified.
- `olmo` — Older AllenAI OLMo repo, reference only.
- `OLMo-core` — Active AllenAI OLMo training framework, reference only.

## Fixed models (hardcoded for this project)
- Text LLM: `allenai/OLMo-3-7B-Instruct` (~14 GB bfloat16) — tool calling via XML+Python-syntax format
- Visual agent: `allenai/MolmoWeb-4B` (~8 GB bfloat16) — screenshot-based browser control

## GPU: Tesla T4, 15 GB VRAM
- Models run SEQUENTIALLY — 14+8=22 GB does not fit simultaneously
- Phase 1 (OLMo): task parsing, URL generation (~60-90s cold start)
- Phase 2 (CPU-only): Playwright screenshot sweep + numpy color analysis — no GPU needed
- Phase 3 (MolmoWeb): visual verification of worst-N positions (~30-60s cold start)
- 128 GB system RAM means OS page cache keeps weights warm after first load

## Key architecture decisions
- Two-pass alignment sweep: coarse Playwright sweep → color analysis → refined sweep → MolmoWeb verify
- NeuroglancerState used directly as Python library for deterministic URL generation (no LLM)
- OLMo LLM only invoked for reasoning bookends (task parse, result interpretation)
- Screenshot color analysis: numpy/PIL, pure CPU, milliseconds per image (red/yellow/green fractions)
- neuroglancer-chat fork: only changes are adapters/olmo_adapter.py (new) + adapters/llm.py (2-line switch)

## Example data
- /capsule/example_ng_link.txt: HCR SPIM brain tissue, channel 405nm (nuclear stain), s3://aind-open-data/

## Why: "molmo-glancer" name
- Chosen by user; combines MolmoWeb (AllenAI visual agent) + Neuroglancer
- AIND institutional context makes "molmo" usage appropriate
- Not an official AI2 project — should note in README
