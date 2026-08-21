# LettuceDetect-like baselines on ToolHACE

Install the inference dependencies once:

```bash
python3 -m pip install lettucedetect datasets huggingface_hub pyarrow torch transformers tqdm
```

Run a Hugging Face or local checkpoint on the `test` split:

```bash
python evaluate/baselines/infer_lettucedetect.py \
  --checkpoint_path KRLabsOrg/lettucedect-large-modernbert-en-v1 \
  --output evaluate/baselines/results/lettucedect_large_test.jsonl
```

For a private model or dataset, set `HF_TOKEN` or pass `--hf-token`. The
script downloads the split parquet from `s-nlp/toolHACE` itself and reads it
directly with PyArrow, which avoids a compatibility issue in some older
`datasets` versions; the output JSONL stores gold labels and spans along with
every prediction.

Compute metrics without loading ToolHACE again:

```bash
python evaluate/baselines/compute_metrics.py \
  evaluate/baselines/results/lettucedect_large_test.jsonl \
  --output-json evaluate/baselines/results/lettucedect_large_metrics.json \
  --output-csv evaluate/baselines/results/lettucedect_large_table.csv
```

The CSV contains one row with `Setting`, `Data`, `Model`, five response-level
class columns plus their average, and three span-level class columns plus
their average. Response-level columns are one-vs-rest F1 values. Span-level
class values are one-to-one span F1 with IoU matching (`IoU > 0.75` by
default); clean-row and undergeneration false-positive spans are included in
the pooled span-level `Avg.`. Use `--iou-threshold` to change the threshold.
Scores in the console summary and CSV table are rendered with two decimal
places (`XX.XX`). The full JSON metrics retain numeric values for downstream
processing.

## How to interpret the metrics

The evaluation measures two different capabilities:

1. whether the detector recognizes that a response is problematic;
2. whether it localizes the problematic text correctly.

### Response-level metrics

Response-level evaluation is binary:

- `clean` is a negative example (`gold = 0`);
- `answer_mismatch`, `overgeneration`, `missing_tool`, and `undergeneration`
  are positive examples (`gold = 1`);
- a prediction is positive if the detector returns at least one span.

Response-level evaluation does not check whether the predicted span has the
correct boundaries. For example, a completely misplaced span still counts as
a response-level true positive if it is emitted for a hallucinated response.

The metrics have the usual interpretation:

- `Precision`: how often a response flagged by the detector is actually
  hallucinated;
- `Recall`: how many hallucinated responses are flagged at all;
- `F1`: the harmonic mean of response-level precision and recall;
- `Accuracy`: the fraction of all responses classified correctly, including
  true negatives on clean examples.

Clean and undergeneration examples are handled explicitly:

- any predicted span on a `clean` example is a response-level false positive;
- an `undergeneration` example is response-level positive, so any predicted
  span counts as a response-level true positive, even though a
  LettuceDetect-like model cannot predict the undergeneration class itself;
- no prediction on a hallucinated response is a response-level false
  negative.

### Per-class response-level values

`Response-level Correct`, `Mismatch`, `Overgen.`, `Missing Tool`, and
`Undergen.` are one-vs-rest F1 values. However, LettuceDetect-like detectors
only return spans and do not return a semantic hallucination class. Therefore,
for each of the four defect classes, the same rule is used:

```text
predicted_class = predicted_spans is not empty
```

These columns must therefore not be interpreted as evidence that the model
distinguishes answer mismatch from overgeneration or missing-tool errors.
They measure how well the detector's decision to emit any span aligns with
examples of each gold type. `Response-level Avg.` is the macro-average of the
five class F1 values, not a semantic classification score.

### Span-level metrics

Span-level evaluation checks localization against the gold character spans.
For every predicted/gold pair, IoU is computed as:

```text
IoU = intersection_length / union_length
```

Matching is greedy and one-to-one:

1. all predicted/gold pairs are sorted by decreasing IoU;
2. a pair is matched only when `IoU > 0.75` by default;
3. each prediction and each gold span can participate in at most one match;
4. unmatched predictions are false positives and unmatched gold spans are
   false negatives.

The threshold comparison is strict: `IoU == 0.75` is not a match. Use
`--iou-threshold` to change the threshold.

Consequently:

- span precision measures how many predicted spans are sufficiently well
  aligned with a gold span;
- span recall measures how many gold spans are sufficiently well recovered;
- span F1 balances span precision and recall.

Clean and undergeneration examples normally have no gold spans:

- a predicted span on `clean` is a span-level false positive;
- a predicted span on `undergeneration` is also a span-level false positive;
- not predicting a span on either of these classes is not a span-level false
  negative, because there is no gold span to recover.

The class-specific span scores are meaningful for `answer_mismatch`,
`overgeneration`, and `missing_tool`, which contain gold hallucination spans.
Clean and undergeneration are retained as false-positive sources in the
overall calculation. The table's `Span-level Avg.` is the pooled span F1 over
all scored rows, including false-positive spans on clean and undergeneration
examples; it is not the arithmetic mean of the three span-bearing class F1s.

### Reading common metric patterns

- High response-level F1 but low span-level F1 means that the detector often
  recognizes a problematic response but localizes the error poorly.
- Low response recall means that the detector frequently remains silent on
  hallucinated responses.
- Low response precision, especially with many clean-row false positives,
  indicates overprediction.
- High span precision with low span recall means that the detector emits few
  spans, but those spans are usually well localized.
- High span recall with low span precision means that the detector covers many
  gold spans but also emits too many extra or overly broad spans.
- A high undergeneration response score does not prove that the model
  identifies undergeneration semantically: it may simply emit some span on
  those responses. Such emissions are still penalized as span-level false
  positives because undergeneration has no target hallucination span.
