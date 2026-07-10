# LettuceDetect — Inference & Evaluation

`evaluate_save.py` is the actual evaluation/inference script used in practice
(taken as-is, aside from a path fix for this folder layout), plus
`predict_spans.py` for running the trained model over arbitrary new data
(no gold labels required).

No data is bundled here — supply your own eval/inference data.

## 1. Install deps

```bash
python3 -m pip install lettucedetect torch transformers pandas tqdm
```

## 2. Environment

| Var | Used by | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | `predict_spans.py` | HF token if the model repo is private |
| `LETTUCE_PATH` | `evaluate_save.py` | Path to a local `lettucedetect` repo checkout, if not pip-installed (or use `--lettuce-path`) |

## 3. Evaluate a trained model

```bash
python3 evaluate_save.py \
  --lettuce-path ./LettuceDetect \
  --model        results_hallucination_detector \
  --data         test.json \
  --save-preds   test_predictions.jsonl \
  --by-type
```

Accepts both LettuceDetect JSON format and raw RAGTruth JSONL (auto-converted).
Reports response-/character-/span-level P/R/F1, plus a 5-class type
classification report (Rule A: detection-based collapse; Rule B: span-overlap
based) — built in, no separate script needed. `--save-preds` writes rich
per-sample predictions (`query, context, prompt, answer, split, task_type,
gold, pred`) so metrics can be recomputed offline with `--load-preds` instead
of re-running inference.

## 4. Run inference on new data (no gold labels needed)

```bash
python3 predict_spans.py --input my_data.parquet --output my_preds.jsonl
python3 predict_spans.py --batch-size 8   # reduce if OOM
```

Uses `etomoscow/tool-calling-hallucination-modernbert-large-crf-best` by default
(`--model` to override). Input parquet needs `query`/`context`/`output` columns.
