# Hallucination Generation

Two pipelines: **singlehop** (flat QA) and **multihop** (multistep dialogues with pruning).

---

## Singlehop — `run.sh`

Generates hallucinations from a flat QA dataset (one user question → one tool call → one answer).

**Input:** `dataset_v3_tagged_cleaned_sys.json`  
**Output:** `type1_output.json`, `type2_output.json`, `type3_output.json`

```bash
./run.sh          # all three types
./run.sh 2        # type 2 only  (no LLM, instant)
./run.sh 1 3      # types 1 and 3
```

To change the dataset or server, edit the top of `generate.py`:

```python
--dataset   dataset_v3_tagged_cleaned_sys.json
--base-url  http://172.17.0.1:8000/v1
--model     Qwen/Qwen2.5-14B-Instruct
```

---

## Multihop — `run_multihop.sh`

Generates hallucinations from multistep dialogues using **pruning**: for a dialogue
with N tool turns, produces samples at depths N, N-1, ..., 2 (skips depth 1 = singlehop).

**Input:** `toolace_multistep_clean (1).json` (186 dialogues)  
**Output:** `pruneddataset/pruned_type1.json`, `pruned_type2.json`, `pruned_type3.json`

```bash
./run_multihop.sh          # all three types
./run_multihop.sh 2        # type 2 only  (no LLM, instant)
./run_multihop.sh 1 3      # types 1 and 3
```

To change the server or batch size, edit the top of `run_multihop.sh`:

```bash
BASE_URL="http://172.17.0.1:8000/v1"
MODEL="Qwen/Qwen2.5-14B-Instruct"
BATCH_SIZE=5
```

---

## Hallucination types

| Type | LLM needed | What happens |
|------|-----------|--------------|
| **1** | Yes (vLLM + guided JSON) | Tool response values replaced with plausible fakes |
| **2** | No | Fields deleted from tool response; answer now over-claims |
| **3** | Yes | One extra sentence added referencing an unused tool |

Type 2 is always instant. Types 1 and 3 require the vLLM server to be running.

**Check the server is up before running type 1 or 3:**

```bash
curl -s http://172.17.0.1:8000/v1/models | python3 -m json.tool
```

---

## Notes

- All generation is **resumable** — if interrupted, re-run the same command and it picks up where it left off.
- Type 3 will **auto-add a system column** if the dataset is missing one (e.g. Glaive).
- Results can be inspected with `python3 scripts/show_samples.py --type type2`.
