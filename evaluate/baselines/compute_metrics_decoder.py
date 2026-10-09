#!/usr/bin/env python3
"""Score a class-aware detector (decoder / LLM-judge style output) on ToolHACE.

The model predicts a hallucination class for every response and, for the
three span-bearing classes, the offending text.  Undergeneration is a real
class here: the correct prediction is ``pred_type = "undergeneration"`` with
an empty span list, and it is scored exactly like the other classes.

Accepted input (one JSON object per line; field aliases in brackets):

    gold_type      [type, gold_class]           gold class
    gold_spans     [span_labels, gold]          list of {start, end, text}
    answer         [final_answer, output]       answer text the offsets index
    pred_type      [pred_class, predicted_type, pred_types, pred_classes]
                                                 one class or a list of classes
    pred_span_labels [pred_spans, pred]         list of {start, end, text} or
                                                 of strings; text-only spans are
                                                 located in the answer by first
                                                 occurrence, unlocatable = FP
    status         "ok" unless inference failed
    pred_parse_issues / parsed                  parse diagnostics

The verbalized-SFT ``verdicts.jsonl`` layout (``idx``, ``parsed``, ``errors =
[{class, span}]``, a non-empty ``errors`` list = flagged) is also understood.
Rows missing gold fields or
the answer can be completed from the test parquet with ``--gold-parquet``
(joined on ``uid``, then ``row_index``/``idx``, then a unique
``dialogue_id``).

Rows whose class could not be parsed count as "predicted clean" by default
(``--parse-fail-policy clean``) so a model is charged for unusable output;
``--parse-fail-policy skip`` drops them instead.

The CSV columns are identical to ``compute_metrics.py``; concatenate rows with
``merge_tables.py``.

Examples
--------
    python evaluate/baselines/compute_metrics_decoder.py \
        results/kimi_k26_fewshot_test.jsonl \
        --output-json results/kimi_k26_metrics.json \
        --output-csv results/kimi_k26_table.csv

    python evaluate/baselines/compute_metrics_decoder.py \
        runs/qwen3.5-2b-sft/verdicts.jsonl \
        --gold-parquet test.parquet --model Qwen3.5-2B-SFT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from toolhace_metrics_common import (  # noqa: E402
    DEFAULT_IOU_THRESHOLD,
    VALID_TYPES,
    Record,
    build_metrics,
    coerce_json,
    finalize_spans,
    first_present,
    load_jsonl,
    normalize_type,
    parse_span_list,
    print_summary,
    table_row,
    write_csv,
    write_json,
)


# --------------------------------------------------------------------------- #
# Gold completion from the test parquet
# --------------------------------------------------------------------------- #
class GoldIndex:
    def __init__(self, parquet_path: Path) -> None:
        import pyarrow.parquet as parquet

        table = parquet.read_table(parquet_path)
        wanted = [c for c in ("uid", "dialogue_id", "type", "label", "answer", "span_labels", "conversations") if c in table.schema.names]
        self.rows = table.select(wanted).to_pylist()
        self.by_uid = {r["uid"]: r for r in self.rows if r.get("uid") is not None}
        by_dialogue: dict[Any, list[dict[str, Any]]] = {}
        for r in self.rows:
            by_dialogue.setdefault(r.get("dialogue_id"), []).append(r)
        self.by_dialogue = by_dialogue

    def lookup(self, row: dict[str, Any]) -> dict[str, Any] | None:
        uid = row.get("uid")
        if uid is not None and uid in self.by_uid:
            return self.by_uid[uid]
        index = first_present(row, "row_index", "idx", "index")
        if isinstance(index, int) and 0 <= index < len(self.rows):
            candidate = self.rows[index]
            dialogue = row.get("dialogue_id")
            if dialogue is None or candidate.get("dialogue_id") == dialogue:
                return candidate
        dialogue = row.get("dialogue_id")
        candidates = self.by_dialogue.get(dialogue, [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            gold = normalize_type(first_present(row, "gold_type", "type"))
            narrowed = [c for c in candidates if normalize_type(c.get("type")) == gold]
            if len(narrowed) == 1:
                return narrowed[0]
        return None


def gold_answer(gold_row: dict[str, Any]) -> str:
    if gold_row.get("answer"):
        return str(gold_row["answer"])
    for turn in reversed(gold_row.get("conversations") or []):
        if isinstance(turn, dict) and turn.get("turn_role") == "final_answer":
            return str(turn.get("value", ""))
    return ""


# --------------------------------------------------------------------------- #
# Row conversion
# --------------------------------------------------------------------------- #
OTHER_CLASS = "other"  # a reported error whose class name is outside the taxonomy


def _class_list(value: Any) -> list[str]:
    value = coerce_json(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    elif isinstance(value, dict):
        items = [value.get("type", value.get("class"))]
    else:
        items = [value]
    classes: list[str] = []
    for item in items:
        if isinstance(item, dict):
            item = item.get("class", item.get("type"))
        label = normalize_type(item)
        if label in VALID_TYPES and label not in classes:
            classes.append(label)
    return classes


def predicted_classes_and_spans(row: dict[str, Any]) -> tuple[list[str] | None, list[dict[str, Any]], bool]:
    """Return (classes, raw pred spans, parse_failed).

    A parsed verdict that reports an error under a class name outside the
    taxonomy is flagged with class ``other`` (counts for detection, never for
    type accuracy) -- the verbalized-SFT convention (non-empty ``errors`` =
    flagged). Only unparseable output is a parse failure.
    """
    status = str(row.get("status", "ok"))
    if status != "ok":
        return None, [], True

    if "errors" in row and not any(k in row for k in ("pred_type", "pred_class", "predicted_type", "pred_types", "pred_classes")):
        # verbalized-SFT verdicts.jsonl layout
        parsed = row.get("parsed", True)
        errors = coerce_json(row.get("errors"))
        if not parsed or errors is None:
            return None, [], True
        errors = errors if isinstance(errors, list) else []
        classes = _class_list([e for e in errors if isinstance(e, dict)])
        spans = parse_span_list([e for e in errors if isinstance(e, dict) and e.get("span")])
        if not errors:
            classes = ["clean"]
        elif not classes:
            classes = [OTHER_CLASS]
        return classes, spans, False

    raw_type = first_present(row, "pred_type", "pred_class", "predicted_type", "pred_types", "pred_classes")
    classes = _class_list(raw_type)
    spans = parse_span_list(first_present(row, "pred_span_labels", "pred_spans", "pred", default=[]))
    if not classes:
        if normalize_type(raw_type) in (None, "", "unknown", "null"):
            return None, spans, True
        classes = [OTHER_CLASS]
    return classes, spans, False


def row_to_record(
    row: dict[str, Any], gold_index: GoldIndex | None, parse_fail_policy: str
) -> Record | None:
    gold_row = gold_index.lookup(row) if gold_index else None
    gold = normalize_type(first_present(row, "gold_type", "type", "gold_class"))
    if gold not in VALID_TYPES and gold_row is not None:
        gold = normalize_type(gold_row.get("type"))
    if gold not in VALID_TYPES:
        return None
    answer = str(first_present(row, "answer", "final_answer", "output", default="") or "")
    if not answer and gold_row is not None:
        answer = gold_answer(gold_row)
    raw_gold = first_present(row, "gold_spans", "span_labels", "gold")
    if raw_gold is None and gold_row is not None:
        raw_gold = gold_row.get("span_labels")
    gold_spans, _ = finalize_spans(parse_span_list(raw_gold or []), answer, locate_by_text=False)

    classes, raw_pred, parse_failed = predicted_classes_and_spans(row)
    if classes is None:
        if parse_fail_policy == "skip":
            return None
        classes = ["clean"]
    pred_spans, unlocatable = finalize_spans(raw_pred, answer, locate_by_text=True)
    flagged = any(label != "clean" for label in classes)
    return Record(
        gold=gold,
        answer=answer,
        gold_spans=gold_spans,
        pred_spans=pred_spans,
        flagged=flagged,
        pred_classes=tuple(classes),
        status=str(row.get("status", "ok")),
        unlocatable=unlocatable,
        parse_failed=parse_failed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ToolHACE response-/class-/span-level metrics for a class-aware detector."
    )
    parser.add_argument("predictions", type=Path, help="JSONL with pred_type and pred_span_labels per row")
    parser.add_argument("--gold-parquet", type=Path, default=None, help="Test parquet used to fill missing gold fields / answers.")
    parser.add_argument("--output-json", type=Path, default=None, help="Write the full metrics summary as JSON.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Write one table row as CSV.")
    parser.add_argument("--setting", default=None, help="Override the Setting table field.")
    parser.add_argument("--data", dest="data_name", default=None, help="Override the Data table field.")
    parser.add_argument("--model", dest="model_name", default=None, help="Override the Model table field.")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU threshold for span matching, strict (default: 0.75).",
    )
    parser.add_argument(
        "--parse-fail-policy",
        choices=("clean", "skip"),
        default="clean",
        help="Rows without a parseable class: count as predicted clean (default) or drop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(args.predictions)
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be between 0 and 1")
    rows = load_jsonl(args.predictions)
    if not rows:
        raise ValueError(f"Prediction file is empty: {args.predictions}")
    gold_index = GoldIndex(args.gold_parquet) if args.gold_parquet else None

    records = [
        record
        for record in (row_to_record(row, gold_index, args.parse_fail_policy) for row in rows)
        if record is not None
    ]
    if not records:
        raise ValueError("No scorable rows: check gold_type/type fields or pass --gold-parquet")
    first = rows[0]
    mode = first.get("mode") or first.get("setting")
    meta = {
        "model": args.model_name or first.get("model") or first.get("checkpoint_path"),
        "checkpoint_path": first.get("checkpoint_path"),
        "dataset": first.get("dataset"),
        "split": first.get("split"),
        "setting": first.get("setting") or (str(mode).replace("_", " ") if mode else "Class-aware"),
        "data_name": first.get("data_name", first.get("dataset", "ToolHACE")),
        "parse_fail_policy": args.parse_fail_policy,
    }
    metrics = build_metrics(
        records,
        iou_threshold=args.iou_threshold,
        undergen_policy="score",
        detector="class-aware",
        meta=meta,
        total_rows=len(rows),
    )
    row = table_row(metrics, setting=args.setting, data_name=args.data_name)
    print_summary(metrics, row)
    if args.output_json:
        write_json(args.output_json, {"metrics": metrics, "table_row": row})
        print(f"Metrics written to {args.output_json}")
    if args.output_csv:
        write_csv(args.output_csv, row)
        print(f"Table row written to {args.output_csv}")


if __name__ == "__main__":
    main()
