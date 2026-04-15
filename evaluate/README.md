# How to RUN

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