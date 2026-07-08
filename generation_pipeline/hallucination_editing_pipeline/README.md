# Hallucination Generation Pipeline

Generates synthetic tool-calling hallucinations from ToolACE-derived dialogues and
converts them into RAGTruth/LettuceDetect-ready training data.

This folder ships **code only** — no data is bundled. Supply your own input
files under `data/` (see [Load ToolACE](#3-load-toolace) below); everything the
pipeline produces is written under `output/` (created automatically).

## Hallucination types

Singlehop generation (`generate.py`) supports two generations of types. **Use the
`.1`/`.2` variants below — they are the current, bug-fixed set** (`error/fixed/`
audits removed malformed rows from the legacy `1`/`2` outputs; `.1`/`.2` supersede
them). The legacy flat types `1`/`2`/`3` still exist in the code (kept for
compatibility / historical runs) but should not be used for new generation.

| Type | Name | LLM needed | Status |
|------|------|-----------|--------|
| **2.1** | Undergeneration — cascade field deletion | No | ✅ current default |
| **1.2** | Incorrect Info — span-targeted schema hallucination | Yes (vLLM) | ✅ preferred over 1.1 (higher yield), needs 2.1 first |
| **3.1** | Overgeneration — filler-filtered spurious sentence | Yes (vLLM) | ✅ current default |
Multihop/pruning generation (`generate_multihop.py`, `run_multihop.sh`) has **not**
been migrated to the new type scheme yet — it only produces the legacy flat types
`1`/`2`/`3` (schema corruption / field deletion / overgeneration on multi-turn
pruned dialogues). Included here for completeness, but treat it as the
not-yet-redacted part of the pipeline.

## 1. Install deps

```bash
python3 -m pip install -r requirements.txt
```

For the LLM-driven types (1.1, 1.2, 3.1, and legacy 1/3) you also need a running
**vLLM** server (OpenAI-compatible API) — see [Server config](#server-config) below.
Type 2.1 (and legacy 2) never call an LLM.

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

**Singlehop** (preferred types, in order — 2.1 must run before 1.1/1.2):

```bash
./run.sh                      # default: 2.1, 1.1, 3.1
./run.sh 2.1 1.2 3.1          # use 1.2 instead of 1.1 (recommended — higher yield)
./run.sh 2.1                  # just cascade deletion, no vLLM server needed
FRESH=1 ./run.sh 3.1          # wipe existing type 3.1 output and regenerate
```

Edit the `CONFIG` block at the top of `run.sh` to point `BASE_URL`/`MODEL` at your
vLLM server. Writes to `output/singlehop_new/`.

**Multihop** (legacy types only):

```bash
./run_multihop.sh          # types 1, 2, 3
./run_multihop.sh 2        # type 2 only, no server needed
```

Writes to `output/pruneddataset/`.

## 6. Dataset (post-processing)

```bash
# Collect singlehop (2.1/1.2/3.1) + multihop (1/2/3) into canonical final_dataset/
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
python3 test_output.py            # checks label offsets, type3 span position, no leftover <hall> tags
python3 test_output.py -v         # verbose per-row failures
```

## Server config

Both `run.sh` and `run_multihop.sh` have a `CONFIG` block near the top:

```bash
BASE_URL="http://172.17.0.1:8000/v1"   # vLLM endpoint
MODEL="Qwen/Qwen2.5-14B-Instruct"
BATCH_SIZE=50                           # concurrent async requests
```

Type 2.1 (and legacy type 2) never call the LLM and ignore these settings.

## Files

| File | Role |
| --- | --- |
| `run.sh` | Entry point — singlehop generation |
| `run_multihop.sh` | Entry point — multihop/pruning generation (legacy types only) |
| `generate.py` | CLI driving singlehop generation |
| `generate_multihop.py` | CLI driving multihop generation |
| `hallucination_auto.py` | Assembles all mixins into `HallucinationAuto` |
| `schema.py` | JSON schema tools, locked schema, cascade/focused masking (Type 1.x backbone) |
| `singlehop.py` | Type 1/1.1/1.2/2/2.1/3/3.1 logic for single-turn QA |
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


- Word-boundary-only span matching in Type 2.1 — never match substrings (e.g. `18`
  must not match inside `1840`).
- Type 1.1/1.2 require Type 2.1 output to exist first (`generate.py` enforces this).
