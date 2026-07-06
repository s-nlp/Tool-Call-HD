# LLM-Based Hallucination Generation Pipeline

Generates synthetic tool-calling hallucinations with an LLM (span-annotated
at generation time), audits every row with an LLM judge, and exports
LettuceDetect-ready training data in the **same format as the editing
pipeline** (`../hallucination_editing_pipeline/`), so the outputs of both
can be concatenated for training.

This folder ships **code only** — supply your own input files (see
[Input data](#3-input-data)); everything the pipeline produces is written
under `output/` (created automatically).

## Error classes

Generation (`generate_errors.py`) works in raw class names; the `unify`
stage maps them onto the unified ToolHACE 5-class taxonomy used by
`evaluate/`, the judge, and the released datasets:

| Raw class (generate) | Unified type | label | spans | LLM needed |
|---|---|---|---|---|
| `correct` | `clean` | 0 | `[]` | Yes |
| `hallucination` | `answer_mismatch` | 1 | required — from corrupted-tool-response ⨯ DATA-tag diff | Yes (2-step) |
| `overgeneration` | `overgeneration` | 1 | required — unsupported additions | Yes |
| `missing_tool` | `missing_tool` | 1 | required — non-existent capabilities | Yes |
| `undergeneration` | `undergeneration` | 1 | `[]` **by design** (omissions kept in `omitted_items`) | Yes |

`correct` is ALWAYS generated first — `hallucination` reuses the correct
answer's `<DATA>` span annotations, corrupting the **tool response** (not
the answer text) and deriving spans by cross-referencing DATA tags with
the JSON diff.

Span invariant enforced end-to-end: `answer[start:end] == span.text`,
where `answer` is the last `final_answer` turn of `conversations`.

## 1. Install deps

```bash
python3 -m pip install -r requirements.txt
```

Stages `generate` and `judge` need an OpenAI-compatible endpoint (local
vLLM or OpenRouter) — see [Server config](#server-config). All other
stages never call an LLM.

## 2. Create env

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 3. Input data

Two input layouts are accepted by `generate_errors.py` (HF Hub ID,
`save_to_disk` dir, or `.json/.jsonl/.csv/.parquet` file):

| Format | Expected schema |
| --- | --- |
| flat | `user_prompt, tool_call, tool_response, original_answer` (single tool call per row) |
| conversation | `conversations: [{"from": "user"\|"assistant"\|"tool", "value": str}, ...]` — the LAST tool turn is targeted; prior turns become history |

Point `DATASET` in `run.sh` at your file, e.g.
`data/singlehop_synthetic_toolace_converted.json`.

## 4. Run

```bash
./run.sh                          # all stages: generate → … → validate
./run.sh unify fix judge filter   # re-run from an existing generation
./run.sh generate                 # generation only (needs gen server)
FRESH=1 ./run.sh judge filter     # wipe judge output and re-judge
```

Edit the `CONFIG` block at the top of `run.sh` (run name, dataset,
generator/judge endpoints, filter policy, split ratios).

Pipeline at a glance:

```text
data/*.json ──► generate_errors.py ──► output/generated/<RUN>/     (raw HF dataset)
                        │
                  to_unified.py    ──► output/unified/<RUN>/       (unified schema)
                        │                       + reports/<RUN>_unify.jsonl
                fix_unicode_drift.py ─► output/fixed/<RUN>/
                        │                       + reports/<RUN>_drift.jsonl
             judge_verify_annotations.py ► output/judge/<RUN>.jsonl
                        │
                 filter_by_judge.py ──► output/final_dataset/<RUN>/  (+ .jsonl)
                        │
        ┌───────────────┴────────────────┐
        ▼                                ▼
export_lettucedetect.py         validate_output.py  (CI gate, exit 1 on failure)
        │
output/lettucedetect_data/toolhace_generation_<RUN>_{train,dev,test}.json
```

## Stage details

| Stage | Script | LLM | What it does |
| --- | --- | --- | --- |
| generate | `generate_errors.py` | ✅ | Span-annotated error generation, `correct` first; resumable per-class |
| unify | `to_unified.py` | — | Raw → unified schema; class→type mapping; safe span re-anchoring; drops `UNIFY_SPAN_BROKEN` rows; position-aligned punct drift is passed through for the fix stage |
| fix | `fix_unicode_drift.py` | — | Repairs punct/unicode span-text drift (`SPAN_FIXED_PUNCT_DRIFT`); flags genuine offset errors instead of masking them |
| judge | `judge_verify_annotations.py` | ✅ | Per-row verdict (`pass/fail/uncertain`), type/span checks, suggested corrected spans; `--resume` skips already-judged rows |
| filter | `filter_by_judge.py` | — | Keeps `pass`; drops `fail`/`uncertain` (configurable); `--apply-suggested-spans` rescues span-only failures when the suggested text anchors uniquely in the answer |
| export | `export_lettucedetect.py` | — | Unified rows → LettuceDetect format identical to the editing pipeline's `make_lettucedetect_data.py`; **splits assigned by `dialogue_id`** (one dialogue yields up to 5 class-rows — per-row splits would leak) |
| validate | `validate_output.py` | — | Invariant gate with stable issue codes; exit 1 on any failure |

## Unified row schema

```text
dialogue_id     : source dialogue id (NOT unique alone — one per class!)
generation_id   : "<class>_<idx>" from generation
subset, type    : run tag / unified 5-class type
label           : 0 (clean) | 1
system          : original system prompt (tool inventory)
conversations   : [{from, value, turn_role}] — last assistant turn has
                  turn_role="final_answer"; span offsets index into it
span_labels     : [{start, end, text}]
tool_call, tool_response, original_tool_response, omitted_items
```

Join key everywhere downstream (judge verdicts, filtering) is
**`(dialogue_id, type)`**.

## Validate before calling it done

```bash
python3 validate_output.py output/final_dataset/<RUN>       # CI gate
python3 validate_output.py output/final_dataset/<RUN> -v    # per-row failures
```

Stable issue codes: `MISSING_FIELD`, `BAD_TYPE`, `LABEL_TYPE_MISMATCH`,
`NO_FINAL_ANSWER`, `SPAN_OOR`, `SPAN_INVARIANT`, `EMPTY_SPANS_REQUIRED`,
`SPANS_REQUIRED`.

## Server config

The `CONFIG` block in `run.sh` holds two independent endpoints:

```bash
GEN_BASE_URL="http://172.17.0.1:8000/v1"      # generation (vLLM)
GEN_MODEL="openai/gpt-oss-120b"
JUDGE_BASE_URL="https://openrouter.ai/api/v1" # judge (OpenRouter)
JUDGE_MODEL="openai/gpt-oss-120b"
```

Using a different judge model than the generator is recommended — a model
judging its own generations inherits its own blind spots.

## Files

| File | Role |
| --- | --- |
| `run.sh` | Entry point — staged orchestration |
| `generate_errors.py` | LLM error generation with span annotations |
| `to_unified.py` | Raw → unified schema converter (the glue stage) |
| `fix_unicode_drift.py` | Conservative span-text drift repair |
| `judge_verify_annotations.py` | LLM-judge annotation audit (JSONL verdicts) |
| `filter_by_judge.py` | Verdict-driven row filtering / span rescue |
| `export_lettucedetect.py` | Unified → LettuceDetect train/dev/test |
| `validate_output.py` | CI gate on schema + span invariants |