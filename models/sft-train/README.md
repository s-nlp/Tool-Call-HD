# SFT-Train — Generative hallucination detector

Fine-tune a **chat LLM** (Qwen3.5 or Gemma) with LoRA so that, given a
tool-calling trace, it emits the error verdict as a single JSON object:

```json
{"type": "answer_mismatch", "spans": [{"start": 42, "end": 71, "text": "..."}]}
```

This is the *generative* alternative to the ModernBERT token-classifier in
[`../lettucedetect/`](../lettucedetect/): instead of tagging tokens, the model
is trained to write out the type and character spans directly.

The pipeline is three scripts, run in order:

```text
prepare_data_qwen_gemma.py   unified ToolHACE rows  →  SFT messages (save_to_disk)
train_sloth.py               SFT dataset            →  LoRA adapter (Unsloth)
vllm_eval_lora.py            SFT dataset + adapters  →  predictions + metrics
```

## Install

`train_sloth.py` requires **Unsloth** (which pulls a matching `transformers`).
Install it on its own, then the rest:

```bash
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
pip install trl datasets vllm transformers
```

> Unsloth must be importable *before* `transformers` so its model patches
> apply — `train_sloth.py` already imports it first; don't reorder those imports.

VRAM (bf16 LoRA, no 4-bit — Unsloth advises against 4-bit for Qwen3.5):

| Model | Registry key | ~VRAM |
| --- | --- | --- |
| Qwen3.5-2B  | `qwen_2b`  | small |
| Qwen3.5-0.8B | `qwen_08b` | smallest |
| Gemma 4 E2B | `gemma`    | ~10 GB |

## 1. Prepare the SFT dataset

Converts unified ToolHACE rows (`system`, `conversations`, `span_labels`,
`type`, `label`, `dialogue_id`, …) into chat-format examples. Each row becomes
`[system, user, assistant]` messages; the assistant message is the JSON target,
and validation checks that every span's `text` equals `answer[start:end]`.

```bash
python prepare_data_qwen_gemma.py \
  --input  s-nlp/toolace-unified-hallucinations_upd_v2 \
  --output ./sft_data_qwen \
  --disable-thinking          # Qwen3.5: pins enable_thinking=False per row
```

| Argument | Default | Description |
| --- | --- | --- |
| `--input` | *(required)* | Source dataset (see note below on what's accepted) |
| `--output` | *(required)* | Output dir, written with `save_to_disk` |
| `--model` | *(optional)* | Tokenizer id for a chat-template token-count preview only |
| `--strict` | off | Abort if any row fails validation (default: drop bad rows and continue) |
| `--disable-thinking` | off | Add `chat_template_kwargs={'enable_thinking': False}` to every row. Needed for Qwen3.5; **do not** use for Gemma (it has no thinking mode) |

The output preserves `messages`, `type`, `label`, `n_spans`, `dialogue_id`, and
(if `--disable-thinking`) `chat_template_kwargs`. Splits (`train`/`test`/…) pass
through unchanged.

## 2. Train (LoRA via Unsloth)

Reads the `save_to_disk` dataset from step 1 (must contain `train` and `test`
splits), renders the chat template, filters over-length rows, then LoRA-SFTs
with response-only loss masking (`train_on_responses_only`).

```bash
python train_sloth.py \
  --model  qwen_2b \
  --data   ./sft_data_qwen \
  --output ./out/qwen2b
```

`--model` must be one of the registry keys: **`qwen_2b`**, **`qwen_08b`**,
**`gemma`**. Each key fixes the HF id plus the chat-template role markers used
for loss masking.

Key arguments (see `--help` for the full list):

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | *(required)* | Registry key: `qwen_2b`, `qwen_08b`, or `gemma` |
| `--data` | *(required)* | SFT `DatasetDict` from step 1 (needs `train` + `test`) |
| `--output` | *(required)* | Checkpoint dir; final adapter saved under `<output>/final` |
| `--max-seq-length` | `4096` | Rows longer than this are dropped after templating |
| `--lora-r` / `--lora-alpha` | `16` / `16` | Unsloth recommends `alpha == r` |
| `--lora-dropout` | `0.0` | Unsloth's fast kernels require `0.0` |
| `--lr` | `2e-4` | — |
| `--epochs` | `1.0` | — |
| `--batch-size` / `--grad-accum` | `1` / `16` | Effective batch = product |
| `--max-train-samples` | `None` | Cap for smoke tests |
| `--max-eval-samples` | `500` | Cap eval-split size |

The script sanity-checks that the assistant `response_part` marker appears in a
rendered example before training — if it's missing, loss masking would zero
everything, so it exits early with an explanatory error instead of crashing
mid-run. The final LoRA adapter + tokenizer land in `<output>/final`.

## 3. Evaluate with vLLM

Loads the base model once and swaps LoRA adapters in-engine, so several
adapters can be compared cheaply on the same prompts. Reads the **step-1**
dataset (it needs the `messages` / `chat_template_kwargs` columns, which the
trainer strips in-memory but never re-saves).

```bash
python vllm_eval_lora.py \
  --base-model Qwen/Qwen3.5-2B \
  --adapters v1=./out/qwen2b/final \
             v2=./out/qwen2b-run2/final \
  --data ./sft_data_qwen --split test \
  --include-base
```

Outputs per-adapter JSONL (`raw_output`, `parsed`, `gold_type`, `gold_spans`,
…) plus a side-by-side table of parse rate / overall accuracy / throughput. The
gold spans are embedded in each row so span-level scoring can be recomputed
offline with `../verbalized/check_eval.py`.

| Argument | Default | Description |
| --- | --- | --- |
| `--base-model` | *(required)* | Base checkpoint the adapters were trained on |
| `--adapters` | *(required)* | One or more `name=path` (or bare path) adapters |
| `--include-base` | off | Also eval the base model with no adapter (reference row) |
| `--data` / `--split` | — / `test` | Step-1 dataset and split |
| `--max-lora-rank` | `64` | Must be `>=` the largest adapter rank |
| `--max-samples` | `None` | Cap number of eval rows |

> **vLLM + LoRA gotcha:** vLLM matches adapters against Qwen3.5's *fused*
> module names. Adapters trained on the unfused gated-delta projections can be
> **silently ignored** at load (warning, not error), making a trained adapter
> look base-like. The script preflights each adapter's `target_modules` and
> flags risky ones, but the only authoritative check is empirical — run one
> adapter here and via a merged path on ~200 rows and confirm predictions
> match before trusting a comparison.
