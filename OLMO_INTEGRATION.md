# OLMo Integration Analysis: Completing the Local Stack

> Updated after investigating `/code/OLMo-core` and the released HuggingFace model cards.

## OLMo-core findings: tool calling exists, but with a custom format

The short answer is: **OLMo-3-7B-Instruct has native function calling** — but it uses a custom XML + Python-syntax format that is not OpenAI-compatible. This is the critical nuance.

### How it works

The `allenai/olmo-toolu-*` datasets in the SFT mix (the `toolu` = "tool use") trained OLMo-3 Instruct and Think variants on explicit function-calling trajectories. The result is a model with dedicated special tokens and a structured format:

**Tool definitions** go in `<functions>` tags as JSON schema:
```
<functions>
[{"name": "ng_set_view", "description": "...", "parameters": {...}}, ...]
</functions>
```

**Tool calls** come back in Python-style syntax inside `<function_calls>` tags:
```
<function_calls>
ng_set_view(center={"x": 100, "y": 200, "z": 50}, zoom=1.5)
</function_calls>
```

**Tool results** go in `<|im_start|>environment` turns.

Special tokens (with dedicated IDs in the tokenizer):
- `<functions>` / `</functions>` → token IDs 100266 / 100267
- `<function_calls>` / `</function_calls>` → token IDs 100268 / 100269

The default system prompt for tool-use mode: `"You are a helpful function-calling AI assistant."`

### The format mismatch with neuroglancer-chat

neuroglancer-chat's `llm.py` uses the OpenAI Python client with `tools=TOOLS, tool_choice="auto"`. OpenAI returns tool calls as structured JSON dicts. OLMo-3-Instruct returns Python-style text like `ng_set_view(center={"x":100}, zoom=2)` wrapped in XML tags.

These are incompatible without an adapter layer. **A `base_url` swap alone will not work** — the response format differs fundamentally.

### What an adapter requires

A custom LLM adapter for neuroglancer-chat (~150 lines of Python) needs to:

1. **Format tools**: convert the `TOOLS` list (OpenAI JSON schema format) into a `<functions>` XML block for the OLMo prompt
2. **Serve**: call OLMo via HuggingFace Transformers or a local vLLM instance (vLLM needs a custom `--tool-call-parser` since OLMo's format isn't hermes/mistral)
3. **Parse responses**: extract `function_name(arg=value, ...)` strings from `<function_calls>` blocks, eval/parse the Python-style call syntax into `{"name": ..., "arguments": {...}}` dicts
4. **Return OpenAI-shaped dicts**: the existing tool dispatch code in `main.py` expects `choice["message"]["tool_calls"][i]["function"]["name"]` and `["arguments"]` — the adapter must produce exactly that structure

This is a contained, mechanical translation layer — not architecturally complex, but it does require writing and testing.

---

## What the cloned `/code/olmo` repo actually is

`/code/olmo` is the original AllenAI OLMo repository, which the repo itself notes is **"out of date and no longer active"** (README line 25). The current development lives at `allenai/OLMo-core`. What it still provides:

- HuggingFace-compatible model definitions (`hf_olmo/`)
- Reference inference scripts, including a vLLM + Modal deployment (`scripts/olmo2_modal_openai.py`)
- Documentation of the OLMo-2 model family on HuggingFace

The key insight: **OLMo-2 is already in this system**. MolmoWeb-4B (`allenai/MolmoWeb-4B`) is built on OLMo-2 with a vision encoder added. The `ai2-molmo2` package in MolmoWeb's deps IS OLMo-2. The cloned repo is the text-only ancestor.

---

## Does OLMo complete the system?

**Yes — with a custom adapter layer of ~150 lines.**

### What OLMo-3-7B-Instruct gives us

| Capability | Available? | Notes |
|---|---|---|
| Local text LLM inference | Yes | HuggingFace Transformers or vLLM |
| Native function calling | **Yes** | Via SFT on `olmo-toolu-*` datasets |
| OpenAI-compatible format | **No** | Custom XML + Python-syntax format |
| Plug-in to neuroglancer-chat with no code change | No | Format adapter required in `llm.py` |
| Streaming | Yes | vLLM handles it |

### The format gap (not a capability gap)

OLMo-3-7B-Instruct can call tools reliably. The issue is purely the wire format: the existing neuroglancer-chat code expects OpenAI JSON tool calls; OLMo produces XML-wrapped Python-syntax calls. An adapter layer bridges this. The model's actual reasoning and tool-selection quality is what matters, and that was explicitly trained on tool-use trajectories.

---

## VRAM reality check on the T4 (15 GB)

OLMo-3-7B-Instruct is the right model but it's large relative to the T4.

The two LLMs in this system compete for the same 15 GB:

| Model | Size (bfloat16) | Size (INT4) | Use |
|---|---|---|---|
| MolmoWeb-4B | ~8 GB | ~2 GB | Visual agent (required for MolmoWeb steps) |
| OLMo-3-7B-Instruct | ~14 GB | ~3.5 GB | Text LLM + tool calling for neuroglancer-chat |
| OLMo-3-7B-Think | ~14 GB | ~3.5 GB | Reasoning + tool calling variant |

**MolmoWeb-4B + OLMo-3-7B-Instruct (bfloat16) = 22 GB — does not fit simultaneously.**

However, in the intended workflow they run in different phases:
- neuroglancer-chat (OLMo-3-7B) does state analysis and URL generation — no vision needed
- MolmoWeb does visual verification — no text LLM needed

Sequential phasing: load OLMo-3-7B, run neuroglancer-chat to generate the annotated URL, unload it, load MolmoWeb-4B, run visual verification. Or use INT4 quantization (~3.5 GB) alongside MolmoWeb-4B (~8 GB) for ~11.5 GB total, both on GPU simultaneously.

---

## What actually completes the system

### Option A: OLMo-3-7B-Instruct with custom adapter (fully AllenAI stack)

Run sequentially. The custom adapter converts between OLMo's XML format and the OpenAI format neuroglancer-chat expects.

```
Phase 1: neuroglancer-chat analysis
  → OLMo-3-7B-Instruct loaded (~14 GB on T4)
  → Custom adapter in llm.py handles format translation
  → Generates annotated NG URL

Phase 2: MolmoWeb verification
  → Unload OLMo, load MolmoWeb-4B (~8 GB)
  → Navigate annotated URL, screenshot
```

**What makes this interesting**: OLMo-3-7B-Instruct was explicitly trained on the `olmo-toolu` datasets — tool-use trajectories — so it should generalise well to neuroglancer-chat's tool schema, even though that specific schema is novel.

**Risk**: The Python-syntax `function_name(arg=val)` parsing is not perfectly reliable if the model outputs malformed syntax. The adapter needs error recovery.

### Option B: OLMo-3-7B-Instruct (INT4) + MolmoWeb-4B simultaneously

With INT4 quantization (~3.5 GB for OLMo-3-7B), both models fit on the T4 at the same time (~11.5 GB total). vLLM supports GPTQ/AWQ. Requires a pre-quantized checkpoint or quantization at download time.

### Option C: Different open model (OpenAI-format native, no adapter needed)

Use a model with native OpenAI-compatible tool calling. These work with vLLM's `--tool-call-parser` and need only a `base_url` change in `llm.py` — no adapter:

| Model | VRAM (bfloat16) | VRAM (INT4) | Tool calling |
|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | ~3.5 GB | Native, excellent, hermes parser |
| `Qwen/Qwen2.5-3B-Instruct` | ~6 GB | ~1.5 GB | Native, fits with MolmoWeb-4B simultaneously |

**Qwen2.5-3B-Instruct** is the zero-adapter path: 6 GB + 8 GB (MolmoWeb-4B) = 14 GB, both on GPU together, 3-line change to `llm.py`.

### Option D: Keep OpenAI for neuroglancer-chat (fallback)

The existing architecture already works if `OPENAI_API_KEY` is available. neuroglancer-chat's `llm.py` gracefully degrades to `"(LLM disabled)"` without it.

---

## Code changes required per option

### Option A: OLMo-3-7B-Instruct adapter

Replace `llm.py`'s `run_chat()` with a new `OLMoAdapter` class. The structural changes:

```python
# llm.py — OLMo-3 adapter (conceptual sketch)
import re, ast
from transformers import AutoModelForCausalLM, AutoTokenizer

def _tools_to_functions_xml(tools: list) -> str:
    """Convert OpenAI-format TOOLS list → <functions> XML block."""
    schemas = [t["function"] for t in tools]
    return f"<functions>\n{json.dumps(schemas, indent=2)}\n</functions>"

def _parse_function_calls(text: str) -> list:
    """Extract Python-style calls from <function_calls> block → OpenAI tool_calls dicts."""
    match = re.search(r"<function_calls>(.*?)</function_calls>", text, re.DOTALL)
    if not match:
        return []
    calls = []
    for call_str in match.group(1).strip().splitlines():
        name, args_str = call_str.split("(", 1)
        args_str = args_str.rstrip(")")
        # Parse Python kwargs → dict (use ast.parse or json if model outputs JSON-style)
        # ... error handling needed here
        calls.append({"id": uuid4().hex, "type": "function",
                      "function": {"name": name.strip(), "arguments": args_str}})
    return calls
```

Also remove `reasoning_effort="minimal"` (OpenAI-only parameter).

### Option B/C: vLLM OpenAI-compatible endpoint (3-line change)

For any OpenAI-format–native model (Qwen2.5, Llama-3.1, etc.) served via vLLM:

```python
# llm.py — 3-line change
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8002/v1")
client = OpenAI(api_key="local", base_url=LOCAL_LLM_URL)
MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")
# Also remove: reasoning_effort="minimal" from the completions call
```

Start the server:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8002 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

No other changes. neuroglancer-chat's agent loop, tool schemas, streaming, and state management stay identical.

---

## Updated system diagram (fully local, Option A)

```
┌──────────────────────────────────────────────────────────────────────┐
│  MolmoWeb-4B  (OLMo-2 + vision encoder, local, T4 GPU)               │
│  Role: visual verification — screenshots, navigation, action loop    │
│  Model server: FastAPI /predict  (port 8001)                         │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ goto(annotated_url) → screenshot
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Playwright / Chromium (headless)                                    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  neuroglancer-chat  (FastAPI backend + Panel UI)                     │
│  Role: structured NG state manipulation, data queries, URL gen       │
│  LLM: OLMo-3-7B-Instruct  (local HF Transformers or vLLM)          │
│    └── custom adapter in llm.py: OLMo XML format ↔ OpenAI format   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ NeuroglancerState (no LLM)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  NeuroglancerState  (pure Python, zero dependencies on LLM)          │
│  Role: parse NG URLs, mutate state, generate new URLs               │
└──────────────────────────────────────────────────────────────────────┘
```

**External API dependencies remaining: zero** (if using Option B or running phases sequentially with Option A).

---

## What the cloned `/code/olmo` is actually useful for

The older `/code/olmo` repo's value is mainly reference:

1. **vLLM deployment pattern** — `scripts/olmo2_modal_openai.py` shows how to serve OLMo via OpenAI-compatible API. Same pattern works locally with `vllm serve`.
2. **Model family reference** — confirms HuggingFace IDs and sizes.
3. **HF model integration** — `hf_olmo/` shows how OLMo registers with HuggingFace AutoModel, which is how `ai2-molmo2` (MolmoWeb's backbone) works under the hood.

For everything else, use `/code/OLMo-core` — it's the active repo and contains the actual SFT training configs, tool-use dataset references, and model release documentation.

---

## Recommended next steps

**Fastest path to fully local system (Option C — no adapter needed):**
1. Install vLLM: `pip install vllm`
2. Serve Qwen2.5-7B-Instruct: `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8002 --enable-auto-tool-choice --tool-call-parser hermes`
3. Apply 3-line patch to `llm.py` (base_url + remove `reasoning_effort`)
4. Test neuroglancer-chat tool calls end-to-end

**Best path for a fully AllenAI/open stack (Option A — with adapter):**
1. Write the OLMo-3-7B-Instruct adapter in `llm.py` (~150 lines)
2. Download `allenai/OLMo-3-7B-Instruct` to `/scratch/checkpoints/`
3. Load via HF Transformers (sequential with MolmoWeb-4B) or vLLM with custom tool-call-parser
4. Test on neuroglancer-chat's most demanding case: parallel batch tool calls (5–8 per iteration)

**Key thing to validate**: neuroglancer-chat's parallel tool call pattern (the LLM issuing 5–8 `data_ng_annotations_from_data` calls in one response) is the hardest test for any local model. Test this early.
