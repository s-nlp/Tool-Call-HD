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

Чтобы запустить:
```bash
python3 evaluate/evaluate_lettuce_metrics.py \
  --input dataset_v3_corrupted.json \
  --output dataset_v3_evaluated.json \
  --summary-output dataset_v3_metrics.json
```

## как проверить на ответе
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
