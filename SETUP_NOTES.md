# Setup Notes — Issues & Fixes

Issues found during environment bring-up that require changes to `_dev_startup2.sh` or a `_download_weights.sh` script.

---

## In Progress — run_capsule.py first launch

*(populated as errors appear)*

---

## Known Issues

### molmoweb `agent/` and `utils/` packages not importable after install

**Symptom:** `ModuleNotFoundError: No module named 'utils'` / `No module named 'agent'`

**Root cause:** `pyproject.toml` has no `[tool.setuptools.packages.find]` section. At install time, `agent/` and `utils/` had no `__init__.py`, so setuptools auto-discovery ignored them. The editable-install finder only registered `inference`.

**Fix applied:** Added `agent/__init__.py` and `utils/__init__.py` to the molmoweb repo. Added `sys.path.insert(0, "/code/lib/molmoweb")` to `run_capsule.py`. Added `PYTHONPATH=/code/lib/molmoweb` to the MolmoWeb server subprocess env.

**Action needed in `_dev_startup2.sh`:** After the molmoweb editable install, ensure the two `__init__.py` files exist:
```bash
touch /code/lib/molmoweb/agent/__init__.py
touch /code/lib/molmoweb/utils/__init__.py
```
Or: add `[tool.setuptools.packages.find]` to molmoweb's `pyproject.toml` (preferred long-term fix via the fork).

---

### torchvision/torch version mismatch — FATAL for native backend

**Symptom:** `RuntimeError: operator torchvision::nms does not exist` — crashes MolmoWeb server when `PREDICTOR_TYPE=native`.

**Root cause:** Dockerfile runs `pip3 install -U ai2-olmo-core==2.4.0` which upgrades torch from 2.4.0 → **2.11.0+cu130**, but leaves torchvision at 0.19.0 (built for 2.4.0). Import chain: `NativeActionPredictor` → `olmo` → `torchmetrics` → `torchvision` → crash.

**Resolution:** Upgraded torchvision to `0.26.0+cu130` (see fix below). Reverted to `PREDICTOR_TYPE=native`.

**HF backend also does not work** — `AutoProcessor.from_pretrained` fails because the checkpoint has no `processor_config.json`. The checkpoint's `config.json` only has `auto_map` entries for `AutoConfig` and `AutoModelForImageTextToText`, not `AutoProcessor`. `HFActionPredictor` is not usable with this checkpoint.

**Fix applied at runtime:** `pip install --upgrade torchvision --extra-index-url https://download.pytorch.org/whl/cu130` → upgraded to `torchvision 0.26.0+cu130`. Confirmed working.

**Action needed in `_dev_startup2.sh`:** Add after the other pip installs:
```bash
"$PIP" install --no-cache-dir torchvision --upgrade \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

**Root cause in Dockerfile:** The line `pip3 install -U ai2-olmo-core==2.4.0` uses `-U` which upgrades torch (via ai2-olmo-core's deps) from 2.4.0 → 2.11.0+cu130 but leaves torchvision at 0.19.0. Options:
1. Add `torchvision` upgrade to `_dev_startup2.sh` (done above) — quick fix
2. Remove `-U` from the Dockerfile pip3 line — keeps torch at 2.4.0, avoids the mismatch entirely but may conflict with newer ai2-olmo-core requirements

---

---

### vllm installed in /scratch/vllm-venv (not conda env)

**Note:** `_dev_startup2.sh` installs vllm into a separate venv at `/scratch/vllm-venv --system-site-packages` to avoid filling the 5 GB root overlay. `run_capsule.py` must invoke it as `/scratch/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server` (not `/opt/conda/bin/python`). Fixed in `run_capsule.py`.

---

---

### NativeActionPredictor loads model in float32 → OOM on T4

**Symptom:** `torch.OutOfMemoryError: CUDA out of memory` during MolmoWeb server startup. The model allocates 13.67 GB before failing to allocate a further 1.45 GB.

**Root cause:** `NativeActionPredictor.__init__` calls `model_cfg.build_model()` with PyTorch's default dtype (float32). The 4.46B-parameter model = **17.8 GB in float32**, exceeding the T4's 14.56 GB.

**Fix applied** in `code/lib/molmoweb/agent/model_backends.py`: set `torch.set_default_dtype(torch.bfloat16)` around the `build_model()` call. In bfloat16, the model is **8.9 GB** — fits with headroom.

**No `_dev_startup2.sh` change needed** — this is a code fix in the molmoweb fork.

### MolmoWeb server load time > 180s — timeout bumped to 300s

Model loading (4.46B params from safetensors) takes >3 min on cold start. Health-check timeout in `run_capsule.py` bumped from 180s → 300s.

---

---

### MolmoWeb-4B native checkpoint download incomplete

**Symptom:** `FileNotFoundError: .../model_and_optim/__48_6.distcp` — native backend fails because `model_and_optim/` only contains shards up to `__37_*` (494 of ~784 expected `.distcp` files).

**Root cause:** Previous `huggingface-cli download` was interrupted mid-download.

**Fix:** `_download_weights.sh` was updated to remove the "already exists → skip" guard. `huggingface-cli download` is idempotent — re-running it resumes and fills in the missing files. Run: `bash /code/_download_weights.sh --skip-olmo`

---

### OLMo vllm server OOM during torch.compile autotuning

**Symptom:** `InductorError: RuntimeError: Failed to run autotuning code block: CUDA out of memory` during vllm KV cache initialization.

**Root cause:** OLMo-3-7B-Instruct in float16 = **14.41 GiB**, nearly the entire T4 (14.56 GiB). vllm's torch.compile piecewise backend runs autotuning that needs ~16 MiB of workspace → no room. Also the default `max_model_len=65536` would require huge KV cache that can't be allocated.

**Fix applied in `run_capsule.py`:** Added `--enforce-eager` (skip torch.compile) and `--max-model-len 2048` (cap KV cache to what synthesis actually needs) to the vllm server command. Both flags are for T4 compatibility.

**No `_dev_startup2.sh` change needed** — runtime flag only.

---

*(further entries added below as the run proceeds)*
