# ToolHACE scoring: span-only and class-aware detectors in one table

Two scorers share one set of table columns, so encoders, decoders and prompted LLMs can be
reported side by side:

| script | detector | a response is *flagged* when | undergeneration |
|---|---|---|---|
| `compute_metrics.py` | span-only (LettuceDetect / ModernBERT taggers, uncertainty heads, Lookback Lens, span-output LLMs) | at least one span is emitted | cannot be expressed → undergeneration rows are left out of the response columns |
| `compute_metrics_decoder.py` | class-aware (SFT decoders, prompted LLMs) | a non-`clean` class is predicted | scored like the other classes, reported in its own column |

Shared code: `toolhace_metrics_common.py`. Other tools:

| file | purpose |
|---|---|
| `infer_lettucedetect.py` | run a LettuceDetect-compatible checkpoint on the toolHACE test split and write scorer input |
| `reparse_llm_predictions.py` | re-derive type and spans from `raw_response` of older `zero_shot.py` / `few_shot.py` outputs (see *Known issues*) |
| `merge_tables.py` | concatenate per-model CSV rows into one table |
| `chance_baselines.py` | chance level of every response column, and each model's lift over it |

## Run

```bash
pip install datasets huggingface_hub pyarrow   # + lettucedetect torch transformers for inference

# span-only model: inference, then scoring
python evaluate/baselines/infer_lettucedetect.py \
  --checkpoint_path KRLabsOrg/lettucedect-large-modernbert-en-v1 \
  --output results/lettucedetect_large_test.jsonl
python evaluate/baselines/compute_metrics.py results/lettucedetect_large_test.jsonl \
  --output-json results/lettucedetect_large_metrics.json --output-csv results/lettucedetect_large_table.csv

# class-aware model (verbalized-SFT verdicts.jsonl, gold read from the test parquet)
python evaluate/baselines/compute_metrics_decoder.py runs/qwen3.5-2b-sft/verdicts.jsonl \
  --gold-parquet test.parquet --model Qwen3.5-2B-SFT --output-csv results/qwen2b_table.csv

# prompted LLM (output of evaluate/zero_shot.py or few_shot.py)
python evaluate/baselines/compute_metrics_decoder.py results/gpt4o_fewshot_test.jsonl \
  --gold-parquet test.parquet --output-csv results/gpt4o_fewshot_table.csv

# one table, and the chance level
python evaluate/baselines/merge_tables.py results/*_table.csv --output-csv results/all.csv --markdown
python evaluate/baselines/chance_baselines.py --gold test.parquet --metrics results/*_metrics.json
```

`test.parquet` is the test split of `s-nlp/toolHACE` (`data/test-00000-of-00001.parquet`).
Table and console scores are F1 × 100 with two decimals (`52.73`); the JSON keeps fractions and
raw tp/fp/fn. Span matching uses IoU > 0.75 (strict); `--iou-threshold` changes it.

## Input formats

Span-only (`compute_metrics.py`) — one JSON object per line:
`answer`, `gold_type`, `gold_spans` `[{start, end, text}]`, `pred_spans` `[{start, end, text}]`,
`status` (`ok` unless inference failed). Every test row must be present: a row the detector did
not cover is written with `pred_spans: []` (no detection), never dropped.

Class-aware (`compute_metrics_decoder.py`) — field aliases in brackets:

```text
gold_type        [type, gold_class]                gold class
gold_spans       [span_labels, gold]               [{start, end, text}]
answer           [final_answer, output]            text the offsets index
pred_type        [pred_class, predicted_type,
                  pred_types, pred_classes]        one class, or a list
pred_span_labels [pred_spans, pred]                [{start, end, text}] or ["text", ...]
status           "ok" unless inference failed
```

The verbalized-SFT layout (`idx`, `parsed`, `errors = [{class, span}]`) is also accepted. Missing
gold fields are filled from `--gold-parquet` (joined on `uid`, then `row_index` / `idx`, then a
unique `dialogue_id`).

## Columns

**Response level.**

* **Correct** — F1 of "not flagged" against gold clean.
* **Mismatch / Overgen. / Missing Tool** — detection F1 on {that class} ∪ {clean}: a flagged row of
  the class is a TP, a flagged clean row a FP, a silent row of the class a FN. Correctly flagged rows
  of *other* classes never count against a column.
* **Undergen.** — same definition; class-aware detectors only (an omission has no span to mark).
* **Avg. (w/o Undergen.)** — mean of Correct, Mismatch, Overgen., Missing Tool.

Undergeneration rows are excluded from every response column except *Undergen.*, for every
detector, so *Avg.* is computed the same way for all models.

**Span level.** Greedy one-to-one matching inside each answer, IoU > 0.75.

* **Mismatch / Overgen. / Missing Tool** — span F1 on the rows of that gold class.
* **Avg. (macro)** — mean of the three.
* **ALL (pooled)** — every row pooled, so spans emitted on clean and undergeneration rows count
  as false positives.

**Type Acc.** (class-aware) — share of hallucinated rows whose gold class is among the predicted
classes. Per-class predicted-class F1 and a confusion matrix are in the JSON (`class_level`).

**Chance level** (`chance_baselines.py`). A detector that ignores its input has, on
{class} ∪ {clean}, precision equal to the class share and recall equal to its flag rate. On the
toolHACE test the best such detector reaches **Avg. 36.70** (flag rate ≈ 54 %); "always flag"
gives 28.71, "never flag" 16.90. A model needs to clear ~36.7 to show it uses the input at all.

## Conventions

* **Spans are anchored on the quoted text.** If the given offsets do not reproduce the span's
  text, the span is moved to the occurrence of the text nearest the given start; text-only spans
  take the first occurrence; if the text is not in the answer the offsets are used, clipped to the
  answer. A predicted span that cannot be placed at all is a false positive.
* **Unparseable output** (`status != ok`, `parsed: false`, no `pred_type`) counts as predicted
  clean (`--parse-fail-policy clean`, default) — a model is charged for unusable output.
* **A parsed verdict naming a class outside the taxonomy** (e.g. free text in the class field) is
  flagged with class `other`: it counts for detection, never for type accuracy.
* **Missing rows are no detection**, not dropped — dropping them shrinks the denominator and
  flatters recall.

## Known issues these scripts fix

* **One-vs-rest on "any span emitted".** The earlier scorer derived every class column from one
  bit and scored it one-vs-rest over all rows: each correctly flagged row of another class became
  a false positive, capping every column near the class prior (a *perfect* span tagger scored
  Mismatch 36, Overgen. 60, Missing Tool 44), and a decoder that correctly answered
  `{"type": "undergeneration", "span_labels": []}` was scored as "predicted clean" (Undergen. ≈ 0).
  Span-level matching was correct and is unchanged.
* **Span repair in `evaluate/zero_shot.py` / `few_shot.py`** (fixed there): when a model's offsets
  did not reproduce its quote, the quote was overwritten with whatever sat at those offsets, and
  spans with out-of-range offsets were dropped. LLM offsets are almost always off, so stored spans
  were mostly misplaced (few-shot span F1 ~11 instead of ~58). Outputs written before the fix keep
  `raw_response`; run `reparse_llm_predictions.py` on them instead of re-querying the API.
* **`infer_lettucedetect.py` rewrote JSON text.** It decoded every JSON-looking string in a row,
  so an answer that is itself JSON was re-printed as a Python dict and tool outputs bypassed
  `toolhace_lettuce_utils.flatten_json_to_text`. Free-text fields are now left untouched.

`evaluate/compute_metrics.py` is a different, earlier report for zero/few-shot files (5-class type
classification, binary detection, character/token overlap); it has no IoU span F1 and no
per-class detection columns. Use the scripts here for comparable tables.
