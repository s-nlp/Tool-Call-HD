# LettuceDetect-like baselines on ToolHACE

Install the inference dependencies once:

```bash
python3 -m pip install lettucedetect datasets torch transformers tqdm
```

Run a Hugging Face or local checkpoint on the `test` split:

```bash
python evaluate/baselines/infer_lettucedetect.py \
  --checkpoint_path KRLabsOrg/lettucedect-large-modernbert-en-v1 \
  --output evaluate/baselines/results/lettucedect_large_test.jsonl
```

For a private model or dataset, set `HF_TOKEN` or pass `--hf-token`. The
script downloads `s-nlp/toolHACE` itself; the output JSONL stores gold labels
and spans along with every prediction.

Compute metrics without loading ToolHACE again:

```bash
python evaluate/baselines/compute_metrics.py \
  evaluate/baselines/results/lettucedect_large_test.jsonl \
  --output-json evaluate/baselines/results/lettucedect_large_metrics.json \
  --output-csv evaluate/baselines/results/lettucedect_large_table.csv
```

The CSV contains one row with `Setting`, `Data`, `Model`, five response-level
class columns plus their average, and three span-level class columns plus
their average. LettuceDetect returns generic hallucination spans rather than
the hallucination type, so response-level class values are per-gold-class
detection rates: specificity for `Correct` and recall for the four error
classes. Span-level values are overlap-based span F1.
