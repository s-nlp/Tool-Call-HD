# How to RUN

## Evaluate the best released ToolHACE model

```bash
python evaluate/evaluate_save.py \
  --model s-nlp/tool-calling-hallucination-modernbert-base-unified-final \
  --hf-dataset s-nlp/toolace-unified-hallucinations \
  --hf-split test \
  --save-preds evaluate/results/toolhace_modernbert_base_test_predictions.jsonl \
  --by-type
```

This script computes:

- response-level binary hallucination metrics
- character-level span metrics
- span-level overlap metrics
- 5-class type reports

It also saves per-row predictions so new metrics can be recomputed without rerunning inference.

To run:
```bash
python3 evaluate/evaluate_lettuce_metrics.py \
  --input dataset_v3_corrupted.json \
  --output dataset_v3_evaluated.json \
  --summary-output dataset_v3_metrics.json
```

## Comparable tables across detectors

`baselines/` holds the unified scorers used for the result tables: `compute_metrics.py` for
span-only detectors, `compute_metrics_decoder.py` for class-aware ones, both with the same
columns (per-class response F1, Avg. without undergeneration, span F1 at IoU > 0.75). See
[`baselines/README.md`](baselines/README.md). Outputs of `zero_shot.py` / `few_shot.py` written
before the span-anchoring fix should be re-parsed with `baselines/reparse_llm_predictions.py`.

## Evaluation using VLLM\OpenRouter

Training-free few-shot baseline: prompt a big model (via OpenRouter, OpenAI lib)
to do BOTH 5-class hallucination classification AND span labeling on the TEST
split, then score it against gold. A check-up of how well large models find these
tool-augmented hallucinations out of the box.

Example:
```bash
# generation
python fewshot_llm_baseline.py ./test_dir \
      --models openai/gpt-4o anthropic/claude-3.5-sonnet \
      --out-dir runs/ --sample-per-class 40 --concurrency 8
  python fewshot_llm_baseline.py test.csv --models qwen/qwen-2.5-72b-instruct --out-dir runs/
  
# classification
  python check_eval.py --preds runs/gpt-4o_preds.jsonl
  python check_eval.py --preds runs/claude-3.5-sonnet_preds.jsonl
```

## How to check answers
```bash
python evaluate/export_lettuce_binary_labels.py \
  --input dataset_v3_tagged_cleaned.json \
  --answer-field original_answer \
  --output-prefix evaluate/outputs/clean_labeled \
  --batch-size 8
```

```bash
python evaluate/export_lettuce_binary_labels.py \
  --input generations/generations_gpt52.json \
  --answer-field generated_answer \
  --output-prefix evaluate/outputs/gpt52_labeled \
  --batch-size 8
```
