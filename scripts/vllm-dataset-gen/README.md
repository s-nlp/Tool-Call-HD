# scripts/ — tool-calling hallucination dataset pipeline

This directory contains every runnable entry point for generating, collecting, converting, and evaluating the dataset.

## Hallucination types

| Type | Name | Mechanism |
| --- | --- | --- |
| 1 | Incorrect Info | LLM-guided JSON corruption — field values changed, answer still references originals |
| 2 | Undergeneration | Fields deleted from tool response — answer references absent data |
| 3 | missing_tool | One sentence appended referencing an unused/fabricated tool result |

---

## Input data files

| File | Description |
| --- | --- |
| `dataset_v3_tagged_cleaned_sys.json` | Singlehop input — annotated tool-calling QA pairs with system prompts |
| `toolace_multistep_clean.json` | Multihop input — 186 multi-turn tool-calling dialogues |
| `singlehop_synthetic_toolace_converted.json` | Synthetic singlehop data from ToolACE |
| `glaive_converted_toolcall_dataset.json` | Converted Glaive tool-calling dataset |

---

## Generation

### Step 1 — Singlehop

```bash
./scripts/run.sh              # all 3 types, async, batch_size=50
./scripts/run.sh 2            # type 2 only (no LLM, instant)
./scripts/run.sh 1 3          # types 1 and 3 only
BATCH_SIZE=20 ./scripts/run.sh
FRESH=1 ./scripts/run.sh      # wipe existing output first
```

Reads `dataset_v3_tagged_cleaned_sys.json`, writes to `singlehop_new/`:

```text
singlehop_new/
  type1_output.json   + type1_output.jsonl
  type2_output.json   + type2_output.jsonl
  type3_output.json   + type3_output.jsonl
```

### Step 2 — Multihop (pruning-based)

```bash
./scripts/run_multihop.sh         # all 3 types
./scripts/run_multihop.sh 2       # type 2 only
FRESH=1 ./scripts/run_multihop.sh
```

Reads `toolace_multistep_clean.json`, writes to `pruneddataset/`:

```text
pruneddataset/
  pruned_type1.json   + pruned_type1.jsonl
  pruned_type2.json   + pruned_type2.jsonl
  pruned_type3.json   + pruned_type3.jsonl
```

Each output row is a truncated multi-turn dialogue where the last tool-call turn carries the hallucination. Dialogues with N completed tool turns produce N−1 rows each (depths 2…N; depth 1 is single-hop).

---

## Post-processing

### Step 3 — Collect into `final_dataset/`

```bash
python3 scripts/collect_final.py
python3 scripts/collect_final.py --no-drop-bad-type3  # keep malformed type3 rows
```

Reads from `singlehop_new/` and `pruneddataset/`, canonically renames, drops type3 rows whose missing_tool span is not at the end of the output:

```text
final_dataset/
  singlehop_incorrect_info_type1.jsonl
  singlehop_undergeneration_type2.jsonl
  singlehop_missing_tool_type3.jsonl
  multistep_incorrect_info_type1.jsonl
  multistep_undergeneration_type2.jsonl
  multistep_missing_tool_type3.jsonl
```

### Step 4 — Merge into one JSONL

```bash
python3 scripts/merge_dataset.py
python3 scripts/merge_dataset.py --input-dir final_dataset --out merged_dataset/merged.jsonl
```

Combines all 6 splits. Hallucinated rows (`hall=1`) kept as-is; clean rows (`hall=0`) deduplicated by `(query, output)`.

Output: `merged_dataset/merged.jsonl`

### Step 5 — Convert to LettuceDetect format

```bash
python3 scripts/make_lettucedetect_data.py
python3 scripts/make_lettucedetect_data.py --dev-ratio 0.1 --test-ratio 0.1
```

Reads `final_dataset/`, writes `lettucedetect_data/tool_calling_hallucination.json` with train/dev/test splits.

---

## Evaluation

### Validate label offsets

```bash
python3 scripts/test_output.py        # check offsets, type3 span positions
python3 scripts/test_output.py -v     # verbose — details on each bad row
```

Run this after generation or after any change to export logic to confirm labels are clean.

### Evaluate a trained LettuceDetect model

```bash
python3 scripts/evaluate_model.py
python3 scripts/evaluate_model.py --model ./output_lettuce \
    --data lettucedetect_data/tool_calling_hallucination.json --split test
python3 scripts/evaluate_model.py --data singlehop_new/type1_output.jsonl --by-type
```

---

## Utilities

```bash
python3 scripts/show_samples.py --type type1 -n 3   # view generated rows
python3 scripts/show_samples.py --type type2 --idx 42
python3 scripts/test_api.py                          # ping the vLLM server
```

---

## Module files (not run directly)

These are imported by the generation scripts — do not run them directly.

| File | Role |
| --- | --- |
| `hallucination_auto.py` | Thin entry point — assembles all mixins into `HallucinationAuto` |
| `schema.py` | JSON schema tools, locked schema, cascade/focused masking (Type 1 backbone) |
| `singlehop.py` | Type 1/2/3 logic for single-turn QA — API calls, async generation loops |
| `multihop.py` | Multistep injection + pruning-based generation |
| `export.py` | Convert generated data → RAGTruth JSONL format |
| `showcase.py` | Rich HTML visualisation for Jupyter notebooks |

---

## Server config

Both `run.sh` and `run_multihop.sh` have a `CONFIG` block at the top. Edit there to change:

```bash
BASE_URL="http://172.17.0.1:8000/v1"   # vLLM endpoint
MODEL="Qwen/Qwen2.5-14B-Instruct"
BATCH_SIZE=50                           # concurrent async requests
```

Type 2 generation never calls the LLM and ignores `BASE_URL`/`MODEL`.

---

## Full pipeline at a glance

```text
dataset_v3_tagged_cleaned_sys.json  ──┐
toolace_multistep_clean.json        ──┤
                                      ▼
                          run.sh / run_multihop.sh
                                      │
               ┌──────────────────────┼──────────────────────┐
               ▼                      ▼                      ▼
        singlehop_new/          pruneddataset/         (type 2: instant)
               └──────────────────────┴──────────────────────┘
                                      │
                              collect_final.py
                                      │
                               final_dataset/
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
              merge_dataset.py           make_lettucedetect_data.py
                     │                                 │
          merged_dataset/merged.jsonl    lettucedetect_data/*.json
                                                       │
                                              evaluate_model.py
```
