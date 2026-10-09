#!/usr/bin/env python3
"""Score a span-only detector (LettuceDetect / encoder token taggers) on ToolHACE.

Input: the JSONL produced by ``infer_lettucedetect.py``.  Gold labels are read
from that file; the benchmark is never downloaded here.

A span-only detector has no class output, so "flagged" means "emitted at least
one span".  Undergeneration has no gold spans by construction and therefore
cannot be expressed by such a detector: under the default ``--undergen-policy
exclude`` its rows are dropped from every response-level number and the
``Response-level Undergen.`` column is left empty.  They remain a span-level
false-positive source, and the share of undergeneration rows that received
spurious spans is reported as ``flag_rate``.  ``--undergen-policy positive``
restores the old behaviour for the *overall* binary numbers (undergeneration
rows count as positives, any emitted span is a hit) but still never turns
that accident into a class column.

Column definitions are documented in ``toolhace_metrics_common.py``.  Both
this script and ``compute_metrics_decoder.py`` emit the same CSV columns, so
``merge_tables.py`` can put encoder and decoder models in one table.

Examples
--------
    python evaluate/baselines/compute_metrics.py \
        evaluate/baselines/results/lettucedect_large_test.jsonl \
        --output-json evaluate/baselines/results/lettucedect_large_metrics.json \
        --output-csv evaluate/baselines/results/lettucedect_large_table.csv
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


def answer_of(row: dict[str, Any]) -> str:
    return str(first_present(row, "answer", "final_answer", "output", default="") or "")


def row_to_record(row: dict[str, Any]) -> Record | None:
    gold = normalize_type(first_present(row, "gold_type", "type"))
    if gold not in VALID_TYPES:
        return None
    answer = answer_of(row)
    status = str(row.get("status", "ok"))
    gold_spans, _ = finalize_spans(
        parse_span_list(first_present(row, "gold_spans", "gold", "span_labels", default=[])),
        answer,
        locate_by_text=False,
    )
    pred_spans: list[dict[str, Any]] = []
    unlocatable = 0
    if status == "ok":
        # spans without usable offsets (e.g. generated text not found in the answer)
        # still mean the detector flagged the row; they are span-level false positives
        pred_spans, unlocatable = finalize_spans(
            parse_span_list(first_present(row, "pred_spans", "pred", "pred_span_labels", default=[])),
            answer,
            locate_by_text=False,
        )
    return Record(
        gold=gold,
        answer=answer,
        gold_spans=gold_spans,
        pred_spans=pred_spans,
        flagged=bool(pred_spans) or unlocatable > 0,
        pred_classes=None,
        status=status,
        unlocatable=unlocatable,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ToolHACE response-/span-level metrics for a span-only detector."
    )
    parser.add_argument("predictions", type=Path, help="JSONL produced by infer_lettucedetect.py")
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
        "--undergen-policy",
        choices=("exclude", "positive"),
        default="exclude",
        help=(
            "exclude (default): drop undergeneration rows from all response-level numbers; "
            "positive: keep them as positives in the overall binary numbers (legacy). "
            "The Undergen. class column is never scored for a span-only detector."
        ),
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

    records = [record for record in (row_to_record(row) for row in rows) if record is not None]
    first = rows[0]
    meta = {
        "model": args.model_name or first.get("model") or first.get("checkpoint_path"),
        "checkpoint_path": first.get("checkpoint_path"),
        "dataset": first.get("dataset"),
        "split": first.get("split"),
        "setting": first.get("setting", "Transfer SFT"),
        "data_name": first.get("data_name", first.get("dataset", "ToolHACE")),
    }
    metrics = build_metrics(
        records,
        iou_threshold=args.iou_threshold,
        undergen_policy=args.undergen_policy,
        detector="span-only",
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
