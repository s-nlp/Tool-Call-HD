
## Method

This method is based on the simple generation of erroneous sentences using a large language model. The generation pipeline consists of generation  -> cleaning -> verification

## How to run

### Generation

Synthetic error generation with SPAN-LEVEL annotations.
Supports multiple model providers (OpenAI-compatible, OpenRouter).

Pipeline order:
  1. "correct" is ALWAYS generated first — it produces data-span-annotated
     answers that other classes (especially hallucination) depend on.
  2. Other classes run after, using correct results where needed.

Classes:
  - correct: faithfully generated answer with <DATA> span annotations
  - hallucination: reuses correct answer + corrupted tool_response;
                   spans derived by cross-referencing DATA tags with JSON diff
  - overgeneration: spans mark unsupported additions
  - missing_tool: spans mark references to non-existent capabilities
  - undergeneration: lists omitted fields/items (no text spans)

Input formats supported:
  Datasets are HuggingFace Dataset objects (from disk, Hub, or files).
  Two layouts are accepted:

  Format A (flat, single tool call per sample):
    user_prompt, tool_call, tool_response, original_answer

  Format B (conversation, multi-turn with possibly multiple tool calls):
    conversations: [{"from": "user"|"assistant"|"tool", "value": str}, ...]
    -- The script targets the LAST tool turn for generation/modification.
    -- All prior messages become conversation history context.

Output:
  HuggingFace dataset saved to disk (load with datasets.load_from_disk)
  or pushed to the Hub with --push-to-hub.

Usage:
```
  python generate_errors.py --input ./toolace_filtered --output ./generated
  python generate_errors.py --input team-ace/ToolACE --output ./generated
  python generate_errors.py --input ./multi_tool_data --output ./generated --format conversation
  python generate_errors.py --input ./data --output my-org/errors --push-to-hub
```

### Cleaning

Repair span-text unicode / whitespace drift in tool-hallucination datasets.

Invariant enforced per span:   answer[start:end] == text
where `answer` is the last `final_answer` turn of `conversations`
(fallback: last assistant turn).

Auto-repair policy (conservative):
  * Only rewrites `text` when the offset substring is a *position-aligned
    punctuation/whitespace variant* of the stored text -- i.e. same length,
    every alphanumeric char identical in place, and only non-alphanumeric
    chars differ (U+2011 -> '-', U+2019 -> "'", U+202F/U+00A0 -> ' ', curly
    quotes, etc.). In that case the offsets are provably stable, so we can
    safely overwrite `text` with the exact substring.
  * Anything else (length mismatch, any alnum difference, out-of-range
    offsets, missing offsets, no answer turn) is FLAGGED, never rewritten,
    so genuine offset bugs are surfaced rather than masked.

Conversations / span_labels may be native Python objects (HF datasets) or
stringified Python literals (CSV) -- both are handled.

Usage
```
  # CSV
  python fix_unicode_drift.py data.csv --out data.fixed.csv --report drift.jsonl
  # HuggingFace dataset saved with save_to_disk (Dataset or DatasetDict)
  python fix_unicode_drift.py ./hf_dataset_dir --out ./hf_fixed --report drift.jsonl
  # detect only, write nothing
  python fix_unicode_drift.py data.csv --check
```

### Verification

LLM-judge for verifying synthetic error annotations.

Takes rows from the synthetic error dataset (HuggingFace dataset OR the small
CSV) and asks a judge model whether each row's annotation is correct:
  - is the `type` field right?
  - are the `span_labels` right (correct text spans, no missing spans, no
    spurious spans)?
  - is the binary `label` (hallucination present yes/no) right?

The judge writes one JSON object per row to a JSONL file. The schema is
fixed (see JUDGE_SCHEMA below) so downstream analysis can aggregate cleanly.

Designed to be provider-agnostic via OpenAI-compatible endpoints — runs
against OpenRouter for hosted models or a local vLLM server for self-hosted
ones. Concurrency is bounded by asyncio.Semaphore.

```
python judge_verify_annotations.py \
    --hf-dataset "./datasets/singlehop/singlehop-gpt_oss/singlehop_synthetic_toolace-correct_turn1" --hf-split "train" \
    --model gpt/gpt-oss-120 \
    --base-url base-ip-for-openrouter \
    --api-key your-api-key \
    --output gpt-oss-120b_train.jsonl \
    --concurrency 8
```

