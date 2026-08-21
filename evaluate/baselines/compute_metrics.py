#!/usr/bin/env python3
"""Compute response-level and span-level ToolHACE metrics from JSONL output.

The input is the JSONL produced by ``infer_lettucedetect.py``.  Gold labels
are read only from that file; the benchmark is never downloaded here.

Response-level scoring is binary because LettuceDetect returns hallucinated
spans, not one of the five hallucination types.  For the table, each class is
scored one-vs-rest: ``clean`` means no predicted span and every hallucination
class means at least one predicted span.  The reported class value is the
resulting F1, so false positives are included.

Per-class span metrics are reported for the three classes with gold spans:
``answer_mismatch``, ``overgeneration`` and ``missing_tool``.  Primary
``span_f1`` is one-to-one span matching performed independently inside each
answer, with IoU > 0.75 by default.  The summary additionally contains
character-overlap F1, token F1, and exact-match rate.  The pooled ``ALL``
span-F1 includes every row: spans predicted on clean and undergeneration rows
are false positives.  The CSV span-level average uses this pooled value.

Examples
--------
    python evaluate/baselines/compute_metrics.py \
        evaluate/baselines/results/lettucedect_large_test.jsonl \
        --output-json evaluate/baselines/results/lettucedect_large_metrics.json \
        --output-csv evaluate/baselines/results/lettucedect_large_table.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


VALID_TYPES = ["clean", "answer_mismatch", "overgeneration", "missing_tool", "undergeneration"]
SPAN_TYPES = ["answer_mismatch", "overgeneration", "missing_tool"]
DEFAULT_IOU_THRESHOLD = 0.75
TYPE_ALIASES = {
    "correct": "clean",
    "clean": "clean",
    "answer_mismatch": "answer_mismatch",
    "answer_missmatch": "answer_mismatch",
    "mismatch": "answer_mismatch",
    "contradiction": "answer_mismatch",
    "overgeneration": "overgeneration",
    "overgen": "overgeneration",
    "missing_tool": "missing_tool",
    "missing tool": "missing_tool",
    "undergeneration": "undergeneration",
    "undergen": "undergeneration",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object.")
            rows.append(value)
    return rows


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    if hasattr(value, "tolist"):
        return _coerce_json(value.tolist())
    return value


def normalize_type(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower().replace("-", "_")
    for prefix in ("singlehop_", "multihop_"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return TYPE_ALIASES.get(value, value)


def row_type(row: dict[str, Any]) -> str | None:
    return normalize_type(row.get("gold_type", row.get("type")))


def row_spans(row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    aliases = {
        "gold_spans": ("gold_spans", "gold", "span_labels"),
        "pred_spans": ("pred_spans", "pred", "pred_span_labels"),
    }
    value: Any = []
    for name in aliases[key]:
        if name in row and row[name] is not None:
            value = _coerce_json(row[name])
            break
    if not isinstance(value, list):
        return []

    result = []
    for span in value:
        if not isinstance(span, dict):
            continue
        try:
            start, end = int(span["start"]), int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        result.append({"start": start, "end": end, "text": str(span.get("text", ""))})
    return result


def answer_of(row: dict[str, Any]) -> str:
    return str(row.get("answer", row.get("final_answer", row.get("output", ""))) or "")


def covered_chars(spans: list[dict[str, Any]], answer: str) -> set[int]:
    result: set[int] = set()
    for span in spans:
        start, end = span["start"], span["end"]
        if 0 <= start < end <= len(answer):
            result.update(range(start, end))
    return result


def span_tokens(spans: list[dict[str, Any]], answer: str) -> set[str]:
    result: set[str] = set()
    for span in spans:
        text = span.get("text") or answer[span["start"] : span["end"]]
        result.update(token.lower() for token in text.split() if token)
    return result


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def round_numbers(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: round_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_numbers(item) for item in value]
    return value


def response_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        gold = row_type(row)
        if gold not in VALID_TYPES:
            continue
        gold_positive = gold != "clean"
        pred_positive = bool(row_spans(row, "pred_spans")) and row.get("status", "ok") == "ok"
        if gold_positive and pred_positive:
            counts["tp"] += 1
        elif not gold_positive and pred_positive:
            counts["fp"] += 1
        elif gold_positive and not pred_positive:
            counts["fn"] += 1
        else:
            counts["tn"] += 1

    binary = prf(counts["tp"], counts["fp"], counts["fn"])
    total = sum(counts[key] for key in ("tp", "fp", "fn", "tn"))
    binary["accuracy"] = (counts["tp"] + counts["tn"]) / total if total else 0.0
    binary.update({key: counts[key] for key in ("tp", "fp", "fn", "tn")})

    by_class: dict[str, Any] = {}
    class_scores: list[float] = []
    for label in VALID_TYPES:
        tp = fp = fn = 0
        for row in rows:
            gold_is_label = row_type(row) == label
            predicted_hallucination = (
                bool(row_spans(row, "pred_spans")) and row.get("status", "ok") == "ok"
            )
            predicted_is_label = (
                not predicted_hallucination if label == "clean" else predicted_hallucination
            )
            if gold_is_label and predicted_is_label:
                tp += 1
            elif not gold_is_label and predicted_is_label:
                fp += 1
            elif gold_is_label and not predicted_is_label:
                fn += 1
        report = prf(tp, fp, fn)
        report["support"] = sum(1 for row in rows if row_type(row) == label)
        report["metric"] = "f1"
        if report["support"]:
            class_scores.append(float(report["f1"]))
        by_class[label] = report

    return {
        "overall": binary,
        "by_class": by_class,
        "macro_class_score": sum(class_scores) / len(class_scores) if class_scores else None,
    }


def overlap_span_prf(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    """One-to-one span P/R/F1 for a single answer.

    A prediction can match at most one gold span.  Matching is intentionally
    done per row, since offsets are relative to each answer and cannot be
    compared across examples.
    """
    pairs = sorted(
        (
            _span_iou(predicted, target),
            pred_index,
            gold_index,
        )
        for pred_index, predicted in enumerate(pred)
        for gold_index, target in enumerate(gold)
    )[::-1]
    used_pred: set[int] = set()
    matched_gold: set[int] = set()
    tp = fp = 0
    for iou, pred_index, gold_index in pairs:
        if iou <= iou_threshold:
            break
        if pred_index in used_pred or gold_index in matched_gold:
            continue
        used_pred.add(pred_index)
        matched_gold.add(gold_index)
        tp += 1
    fp = len(pred) - tp
    fn = len(gold) - len(matched_gold)
    return prf(tp, fp, fn)


def _span_iou(predicted: dict[str, Any], target: dict[str, Any]) -> float:
    overlap = max(
        0,
        min(predicted["end"], target["end"])
        - max(predicted["start"], target["start"]),
    )
    union = (predicted["end"] - predicted["start"]) + (
        target["end"] - target["start"]
    ) - overlap
    return overlap / union if union else 0.0


def char_f1(gold: list[dict[str, Any]], pred: list[dict[str, Any]], answer: str) -> float:
    gold_chars = covered_chars(gold, answer)
    pred_chars = covered_chars(pred, answer)
    if not gold_chars and not pred_chars:
        return 1.0
    if not gold_chars or not pred_chars:
        return 0.0
    tp = len(gold_chars & pred_chars)
    precision = tp / len(pred_chars)
    recall = tp / len(gold_chars)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def token_f1(gold: list[dict[str, Any]], pred: list[dict[str, Any]], answer: str) -> float:
    gold_tokens = span_tokens(gold, answer)
    pred_tokens = span_tokens(pred, answer)
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    tp = len(gold_tokens & pred_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def exact_match(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> bool:
    return {(s["start"], s["end"]) for s in gold} == {(s["start"], s["end"]) for s in pred}


def span_metrics(rows: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    by_class: dict[str, Any] = {}
    class_f1s: list[float] = []
    scopes = [*SPAN_TYPES, "undergeneration", "clean(FP source)"]
    scope_rows: dict[str, list[dict[str, Any]]] = {scope: [] for scope in scopes}
    scope_rows["ALL"] = rows
    for row in rows:
        gold_type = row_type(row)
        if gold_type in scopes:
            scope_rows[gold_type].append(row)
        elif gold_type == "clean":
            scope_rows["clean(FP source)"].append(row)

    def compute_scope(scope: str, scope_data: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter()
        gold_span_count = pred_span_count = 0
        char_scores: list[float] = []
        token_scores: list[float] = []
        exact_scores: list[int] = []
        for row in scope_data:
            gold = row_spans(row, "gold_spans")
            pred = row_spans(row, "pred_spans") if row.get("status", "ok") == "ok" else []
            answer = answer_of(row)
            row_span_metrics = overlap_span_prf(gold, pred, iou_threshold)
            for key in ("tp", "fp", "fn"):
                counts[key] += row_span_metrics[key]
            gold_span_count += len(gold)
            pred_span_count += len(pred)
            if scope in SPAN_TYPES:
                char_scores.append(char_f1(gold, pred, answer))
                token_scores.append(token_f1(gold, pred, answer))
                exact_scores.append(int(exact_match(gold, pred)))

        report = prf(counts["tp"], counts["fp"], counts["fn"])
        report["span_f1"] = report["f1"]
        report["iou_threshold"] = iou_threshold
        report["support_rows"] = len(scope_data)
        report["gold_spans"] = gold_span_count
        report["pred_spans"] = pred_span_count
        report["char_overlap_f1"] = (
            sum(char_scores) / len(char_scores) if char_scores else None
        )
        report["token_f1"] = sum(token_scores) / len(token_scores) if token_scores else None
        report["exact_match_rate"] = (
            sum(exact_scores) / len(exact_scores) if exact_scores else None
        )
        return report

    overall = compute_scope("ALL", scope_rows["ALL"])
    for scope in scopes:
        report = compute_scope(scope, scope_rows[scope])
        by_class[scope] = report
        if scope in SPAN_TYPES and scope_rows[scope]:
            class_f1s.append(float(report["span_f1"]))

    return {
        "by_class": by_class,
        "macro_span_f1": sum(class_f1s) / len(class_f1s) if class_f1s else None,
        "overall": overall,
        "pooled": overall,
    }


def build_metrics(rows: list[dict[str, Any]], iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> dict[str, Any]:
    usable_rows = [row for row in rows if row_type(row) in VALID_TYPES]
    statuses = Counter(row.get("status", "ok") for row in usable_rows)
    response = response_metrics(usable_rows)
    spans = span_metrics(usable_rows, iou_threshold)
    first = usable_rows[0] if usable_rows else (rows[0] if rows else {})
    return round_numbers(
        {
            "total_rows": len(rows),
            "scored_rows": len(usable_rows),
            "skipped_rows": len(rows) - len(usable_rows),
            "status_counts": dict(statuses),
            "model": first.get("model", first.get("checkpoint_path")),
            "checkpoint_path": first.get("checkpoint_path"),
            "dataset": first.get("dataset"),
            "split": first.get("split"),
            "setting": first.get("setting", "Transfer SFT"),
            "data_name": first.get("data_name", first.get("dataset", "ToolHACE")),
            "response_level": response,
            "span_level": spans,
        }
    )


def table_row(metrics: dict[str, Any], setting: str | None = None, data_name: str | None = None) -> dict[str, Any]:
    response = metrics["response_level"]["by_class"]
    spans = metrics["span_level"]["by_class"]

    def response_score(label: str) -> float | None:
        return response.get(label, {}).get("f1")

    def span_score(label: str) -> float | None:
        return spans.get(label, {}).get("span_f1")

    return {
        "Setting": setting or metrics.get("setting") or "Transfer SFT",
        "Data": data_name or metrics.get("data_name") or metrics.get("dataset") or "ToolHACE",
        "Model": metrics.get("model") or metrics.get("checkpoint_path"),
        "Response-level Correct": response_score("clean"),
        "Response-level Mismatch": response_score("answer_mismatch"),
        "Response-level Overgen.": response_score("overgeneration"),
        "Response-level Missing Tool": response_score("missing_tool"),
        "Response-level Undergen.": response_score("undergeneration"),
        "Response-level Avg.": metrics["response_level"].get("macro_class_score"),
        "Span-level Mismatch": span_score("answer_mismatch"),
        "Span-level Overgen.": span_score("overgeneration"),
        "Span-level Missing Tool": span_score("missing_tool"),
        "Span-level Avg.": metrics["span_level"].get("overall", {}).get("span_f1"),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow({key: format_table_value(value) for key, value in row.items()})


def format_table_value(value: Any) -> str:
    """Render table scores with exactly two digits after the decimal point."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_summary(metrics: dict[str, Any], row: dict[str, Any]) -> None:
    response = metrics["response_level"]["overall"]
    print(f"Rows: {metrics['scored_rows']} scored / {metrics['total_rows']} total")
    print(
        "Response-level: "
        f"P={response['precision']:.2f} R={response['recall']:.2f} "
        f"F1={response['f1']:.2f} Acc={response['accuracy']:.2f}"
    )
    print(f"Response-level macro class score: {format_table_value(row['Response-level Avg.'])}")
    print(f"Span-level pooled F1: {format_table_value(row['Span-level Avg.'])}")
    print("Table row:")
    print(json.dumps({key: format_table_value(value) for key, value in row.items()}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ToolHACE response-/span-level metrics from inference JSONL."
    )
    parser.add_argument("predictions", type=Path, help="JSONL produced by infer_lettucedetect.py")
    parser.add_argument("--output-json", type=Path, default=None, help="Write the full metrics summary as JSON.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Write one paper-table row as CSV.")
    parser.add_argument("--setting", default=None, help="Override the Setting table field.")
    parser.add_argument("--data", dest="data_name", default=None, help="Override the Data table field.")
    parser.add_argument("--model", dest="model_name", default=None, help="Override the Model table field.")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU threshold for span matching (default: 0.75).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(args.predictions)
    rows = load_jsonl(args.predictions)
    if not rows:
        raise ValueError(f"Prediction file is empty: {args.predictions}")
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be between 0 and 1")
    metrics = build_metrics(rows, iou_threshold=args.iou_threshold)
    if args.model_name:
        metrics["model"] = args.model_name
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
