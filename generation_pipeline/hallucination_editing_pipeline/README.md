# Hallucination Generation Pipeline

Generates synthetic tool-calling hallucinations from ToolACE-derived dialogues and
converts them into RAGTruth/LettuceDetect-ready training data.

This folder ships **code only** — no data is bundled. Supply your own input
files under `data/` (see [Load ToolACE](#3-load-toolace) below); everything the
pipeline produces is written under `output/` (created automatically).

## Hallucination types

Type names follow the paper's taxonomy: **Answer Mismatch** (answer contradicts
the tool response), **Undergeneration** (answer omits required tool info),
**Overgeneration** (answer adds unsupported content). ("Missing Tool" and
"Correct" are the other two paper classes; this pipeline doesn't generate
"Missing Tool" data.)

Singlehop generation (`generate.py`, `run.sh`) takes these as `--types` values.
**Use the preferred variants below** — the `_legacy` variants exist for
backward compatibility / historical runs but should not be used for new
generation (audits removed malformed rows from their output; the preferred
variants supersede them).

| Type | LLM needed | Status |
|------|-----------|--------|
| `undergeneration` | No | ✅ preferred, current default |
| `answer_mismatch_gated` | Yes (vLLM) | ✅ works, needs `undergeneration` output first |
| `answer_mismatch_targeted` | Yes (vLLM) | ✅ preferred over `answer_mismatch_gated` (higher yield), needs `undergeneration` first |
| `overgeneration` | Yes (vLLM) | ✅ preferred, current default |
| `answer_mismatch_legacy` / `undergeneration_legacy` / `overgeneration_legacy` | Yes/No | ⚠️ superseded, do not use for new data |

Multihop/pruning generation (`generate_multihop.py`, `run_multihop.sh`) only
implements one (legacy) variant per class — `answer_mismatch`, `undergeneration`,
`overgeneration` — there's no gated/targeted refinement there yet. Included
here for completeness, but treat it as the not-yet-redacted part of the
pipeline.

## 1. Install deps

```bash
python3 -m pip install -r requirements.txt
```

For the LLM-driven types (`answer_mismatch_*`, `overgeneration*`) you also need
a running **vLLM** server (OpenAI-compatible API) — see
[Server config](#server-config) below. `undergeneration*` never calls an LLM.

## 2. Create env

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 3. Load ToolACE

Place your own input files under `data/` (create the folder). Expected files:

| File | Expected schema |
| --- | --- |
| `data/singlehop_synthetic_toolace_converted.json` | Singlehop ToolACE rows: `user_prompt, tool_call, tool_response, original_answer, system` |
| `data/toolace_multistep_clean.json` | Multihop/multistep ToolACE dialogues (`conversations` list per dialogue) |

## 4. ToolACE → RAGTruth

```bash
python3 build_toolace_clean_ragtruth.py
```

Converts `data/singlehop_synthetic_toolace_converted.json` into
`output/singlehop_toolace_clean_ragtruth.jsonl` (clean/non-hallucinated baseline
rows in RAGTruth schema: `id, query, context, output, hallucination_labels, ...`).

## 5. Generate Hallucinations

**Singlehop** (preferred types, in order — `undergeneration` must run before
`answer_mismatch_gated`/`answer_mismatch_targeted`):

```bash
./run.sh                                                          # default: undergeneration, answer_mismatch_gated, overgeneration
./run.sh undergeneration answer_mismatch_targeted overgeneration  # use targeted (recommended — higher yield)
./run.sh undergeneration                                          # just cascade deletion, no vLLM server needed
FRESH=1 ./run.sh overgeneration                                   # wipe existing overgeneration output and regenerate
```

Edit the `CONFIG` block at the top of `run.sh` to point `BASE_URL`/`MODEL` at your
vLLM server. Writes to `output/singlehop_new/`.

**Multihop** (legacy types only):

```bash
./run_multihop.sh                                    # all 3 types
./run_multihop.sh undergeneration                    # undergeneration only, no server needed
```

Writes to `output/pruneddataset/`.

## 6. Dataset (post-processing)

```bash
# Collect singlehop + multihop output into canonical final_dataset/
python3 collect_final.py

# Merge all splits into one deduplicated JSONL
python3 merge_dataset.py

# Convert final_dataset/ into LettuceDetect train/dev/test split format
python3 make_lettucedetect_data.py
```

Pipeline at a glance:

```text
data/*.json ──► build_toolace_clean_ragtruth.py ──► output/*_ragtruth.jsonl
                                │
data/*.json ──► run.sh / run_multihop.sh ──► output/singlehop_new/, output/pruneddataset/
                                │
                       collect_final.py
                                │
                   output/final_dataset/
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
        merge_dataset.py            make_lettucedetect_data.py
                │                                │
   output/merged_dataset/merged.jsonl   output/lettucedetect_data/*.json
```

## Validate before calling it done

```bash
python3 test_output.py            # checks label offsets, overgeneration span position, no leftover <hall> tags
python3 test_output.py -v         # verbose per-row failures
```

## Server config

Both `run.sh` and `run_multihop.sh` have a `CONFIG` block near the top:

```bash
BASE_URL="http://172.17.0.1:8000/v1"   # vLLM endpoint
MODEL="Qwen/Qwen2.5-14B-Instruct"
BATCH_SIZE=50                           # concurrent async requests
```

`undergeneration`/`undergeneration_legacy` never call the LLM and ignore these settings.

## Files

| File | Role |
| --- | --- |
| `run.sh` | Entry point — singlehop generation |
| `run_multihop.sh` | Entry point — multihop/pruning generation (legacy types only) |
| `generate.py` | CLI driving singlehop generation |
| `generate_multihop.py` | CLI driving multihop generation |
| `hallucination_auto.py` | Assembles all mixins into `HallucinationAuto` |
| `schema.py` | JSON schema tools, locked schema, cascade/focused masking (Answer Mismatch backbone) |
| `singlehop.py` | Answer Mismatch, Undergeneration, Overgeneration logic for single-turn QA |
| `multihop.py` | Multistep injection + pruning-based generation (legacy types only) |
| `export.py` | Convert generated data → RAGTruth JSONL format |
| `build_toolace_clean_ragtruth.py` | ToolACE → RAGTruth clean-row conversion |
| `collect_final.py` | Collect generation output into `final_dataset/` |
| `merge_dataset.py` | Merge `final_dataset/` splits into one JSONL |
| `make_lettucedetect_data.py` | Convert `final_dataset/` into LettuceDetect format |
| `test_output.py` | Validate generated labels/spans |

Only files actually exercised by this documented flow are included — Jupyter
showcase notebooks, the terminal sample-viewer, the vLLM ping utility, and the
Russian/Glaive-RU branch were dropped as out of scope for the main pipeline.

## Known-fixed bugs (do not reintroduce)

- `collect_final.py`, `merge_dataset.py`, `make_lettucedetect_data.py`, and
  `test_output.py` previously pointed at a `last dub/` directory that no longer
  exists and/or legacy `type1/type2/type3` filenames that don't match what
  `generate.py` actually produces (`type2_1_output.jsonl`, `type1_2_output.jsonl`,
  `type3_1_output.jsonl`). Fixed here to read from `output/singlehop_new/` and
  `output/pruneddataset/` with the correct filenames.
- Word-boundary-only span matching in `undergeneration` — never match substrings
  (e.g. `18` must not match inside `1840`).
- `answer_mismatch_gated`/`answer_mismatch_targeted` require `undergeneration`
  output to exist first (`generate.py` enforces this).

## CLI reference note

`--types` accepts the word names above as the canonical interface
(`generate.py --types undergeneration answer_mismatch_targeted overgeneration`).
The old numeric keys (`1`, `1.1`, `1.2`, `2`, `2.1`, `3`, `3.1`) still work too,
for anything that scripted against the previous interface — see
`WORD_ALIASES` in `generate.py` / `generate_multihop.py` for the full mapping.
Internally, and in the actual output filenames on disk (e.g. `type2_1_output.json`),
the numeric keys are still used — only the human-facing CLI/docs surface changed.
