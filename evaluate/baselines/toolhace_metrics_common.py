#!/usr/bin/env python3
"""Shared ToolHACE metric code for the span-only (encoder) and class-aware
(decoder) scorers.

Both ``compute_metrics.py`` (span-only detectors such as LettuceDetect) and
``compute_metrics_decoder.py`` (models that verbalize a hallucination class)
convert their input JSONL into a list of :class:`Record` objects and then call
the same functions here, so their CSV rows have identical columns and can be
concatenated into one table with ``merge_tables.py``.

Column semantics (identical for both scorers)
---------------------------------------------
Response level, positive = "the detector flagged the response":

* span-only detectors flag a response by emitting at least one span;
* class-aware detectors flag a response by predicting a non-``clean`` class.

``Response-level Correct``
    F1 of the *not flagged* decision against gold clean, undergeneration rows
    excluded (for every detector, so the column is comparable across them).
``Response-level Mismatch / Overgen. / Missing Tool / Undergen.``
    Detection F1 on the subset ``{gold == class} ∪ {gold == clean}``: a flagged
    row of that class is a TP, a flagged clean row is a FP, a silent row of
    that class is a FN.  Correctly flagged rows of *other* defect classes are
    never counted against a class column (the old one-vs-rest-on-any-span rule
    did that and capped every column near the class prior).
``Response-level Undergen.``
    Undergeneration carries no gold spans by construction, so a span-only
    detector cannot express it.  Under the ``exclude`` policy (default for
    span-only) undergeneration rows are removed from *all* response-level
    numbers and the column is empty (``—``); their span-level role as a
    false-positive source is unchanged.  Class-aware detectors use the
    ``score`` policy and get a real number here.
``Response-level Avg. (w/o Undergen.)``
    Mean of Correct, Mismatch, Overgen., Missing Tool — the same four columns
    for every detector (undergeneration is reported, not averaged).
``Type Acc.``
    Class-aware detectors only: fraction of gold-hallucinated rows whose gold
    class is among the predicted classes.
Span level
    Greedy one-to-one matching inside each answer, IoU > threshold (default
    0.75, strict).  Unmatched predictions are FP (including spans on clean and
    undergeneration rows and spans that cannot be located in the answer),
    unmatched gold spans are FN.  ``Span-level Avg. (macro)`` is the mean of the
    three span-class F1s; ``Span-level ALL (pooled)`` pools every scored row, so
    spans emitted on clean and undergeneration rows count against it.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VALID_TYPES = ["clean", "answer_mismatch", "overgeneration", "missing_tool", "undergeneration"]
DEFECT_TYPES = VALID_TYPES[1:]
SPAN_TYPES = ["answer_mismatch", "overgeneration", "missing_tool"]
DEFAULT_IOU_THRESHOLD = 0.75
UNDERGEN_POLICIES = ("exclude", "positive", "score")

TYPE_ALIASES = {
    "correct": "clean",
    "clean": "clean",
    "ok": "clean",
    "none": "clean",
    "no_hallucination": "clean",
    "answer_mismatch": "answer_mismatch",
    "answer_missmatch": "answer_mismatch",
    "mismatch": "answer_mismatch",
    "contradiction": "answer_mismatch",
    "am": "answer_mismatch",
    "overgeneration": "overgeneration",
    "overgen": "overgeneration",
    "og": "overgeneration",
    "missing_tool": "missing_tool",
    "missing tool": "missing_tool",
    "mt": "missing_tool",
    "undergeneration": "undergeneration",
    "undergen": "undergeneration",
    "ug": "undergeneration",
}

TABLE_COLUMNS = [
    "Setting",
    "Data",
    "Model",
    "Detector",
    "Response-level Correct",
    "Response-level Mismatch",
    "Response-level Overgen.",
    "Response-level Missing Tool",
    "Response-level Undergen.",
    "Response-level Avg. (w/o Undergen.)",
    "Type Acc.",
    "Span-level Mismatch",
    "Span-level Overgen.",
    "Span-level Missing Tool",
    "Span-level Avg. (macro)",
    "Span-level ALL (pooled)",
    "Undergen. rows",
    "Scored rows",
]


@dataclass
class Record:
    """One scored test row, independent of the detector family."""

    gold: str
    answer: str
    gold_spans: list[dict[str, Any]]
    pred_spans: list[dict[str, Any]]
    flagged: bool
    pred_classes: tuple[str, ...] | None = None  # None for span-only detectors
    status: str = "ok"
    unlocatable: int = 0
    parse_failed: bool = False


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
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


def coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    if hasattr(value, "tolist"):
        return coerce_json(value.tolist())
    return value


def normalize_type(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return normalize_type(value[0]) if value else None
    value = str(value).strip().lower().replace("-", "_")
    for prefix in ("singlehop_", "multihop_"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return TYPE_ALIASES.get(value, value)


def first_present(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def parse_span_list(value: Any) -> list[dict[str, Any]]:
    """Return raw span dicts (``start``/``end`` ints when present, ``text``)."""
    value = coerce_json(value)
    if isinstance(value, dict):
        value = value.get("spans", value.get("predictions", []))
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for span in value:
        span = coerce_json(span)
        if isinstance(span, str):
            if span.strip():
                result.append({"text": span})
            continue
        if isinstance(span, (list, tuple)) and len(span) >= 2:
            span = {"start": span[0], "end": span[1], **({"text": span[2]} if len(span) > 2 else {})}
        if not isinstance(span, dict):
            continue
        item: dict[str, Any] = {}
        try:
            if span.get("start") is not None and span.get("end") is not None:
                item["start"], item["end"] = int(span["start"]), int(span["end"])
        except (TypeError, ValueError):
            pass
        text = span.get("text", span.get("span"))
        if isinstance(text, str) and text:
            item["text"] = text
        label = span.get("label_type", span.get("type", span.get("category", span.get("class"))))
        if label is not None:
            item["label_type"] = str(label)
        if item:
            result.append(item)
    return result


def nearest_occurrence(answer: str, text: str, anchor: int) -> int:
    """Start of the occurrence of ``text`` in ``answer`` closest to ``anchor`` (-1 if absent)."""
    best, position = -1, answer.find(text)
    while position >= 0:
        if best < 0 or abs(position - anchor) < abs(best - anchor):
            best = position
        position = answer.find(text, position + 1)
    return best


def finalize_spans(
    raw_spans: list[dict[str, Any]], answer: str, *, locate_by_text: bool
) -> tuple[list[dict[str, Any]], int]:
    """Turn raw spans into ``{start, end, text}`` offsets into ``answer``.

    * Offsets past the end of the answer are clipped, not dropped.
    * When ``locate_by_text`` is set and the offsets do not reproduce ``text``,
      the occurrence of ``text`` nearest to the given start is used; without
      offsets, the first occurrence is used.
    * Predicted spans that cannot be placed inside the answer (no usable
      offsets and text not found, or offsets entirely outside the answer) are
      dropped and counted in the returned ``unlocatable`` figure (the caller
      scores them as false positives).
    """
    spans: list[dict[str, Any]] = []
    unlocatable = 0
    for raw in raw_spans:
        start, end, text = raw.get("start"), raw.get("end"), raw.get("text", "")
        has_offsets = isinstance(start, int) and isinstance(end, int)
        if has_offsets and answer and text and answer[start:end] != text and locate_by_text:
            found = nearest_occurrence(answer, text, start)
            if found >= 0:
                start, end = found, found + len(text)
        if not has_offsets:
            if locate_by_text and text:
                found = answer.find(text)
                if found < 0:
                    unlocatable += 1
                    continue
                start, end = found, found + len(text)
            else:
                unlocatable += 1
                continue
        if answer:
            end = min(end, len(answer))
        if start < 0 or end <= start:
            if locate_by_text:
                unlocatable += 1
            continue
        span = {"start": start, "end": end, "text": answer[start:end] if answer else text}
        if "label_type" in raw:
            span["label_type"] = raw["label_type"]
        spans.append(span)
    return spans, unlocatable


# --------------------------------------------------------------------------- #
# Basic scoring helpers
# --------------------------------------------------------------------------- #
def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
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


def span_iou(predicted: dict[str, Any], target: dict[str, Any]) -> float:
    overlap = max(0, min(predicted["end"], target["end"]) - max(predicted["start"], target["start"]))
    union = (predicted["end"] - predicted["start"]) + (target["end"] - target["start"]) - overlap
    return overlap / union if union else 0.0


def match_spans(pred: list[dict[str, Any]], gold: list[dict[str, Any]], iou_threshold: float) -> int:
    """Greedy one-to-one matching by descending IoU; strict ``IoU > threshold``."""
    pairs = sorted(
        ((span_iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold)),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    tp = 0
    for iou, i, j in pairs:
        if iou <= iou_threshold:
            break
        if i in used_pred or j in used_gold:
            continue
        used_pred.add(i)
        used_gold.add(j)
        tp += 1
    return tp


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


def _set_f1(gold: set, pred: set) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    precision, recall = tp / len(pred), tp / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def char_f1(gold: list[dict[str, Any]], pred: list[dict[str, Any]], answer: str) -> float:
    return _set_f1(covered_chars(gold, answer), covered_chars(pred, answer))


def token_f1(gold: list[dict[str, Any]], pred: list[dict[str, Any]], answer: str) -> float:
    return _set_f1(span_tokens(gold, answer), span_tokens(pred, answer))


def exact_match(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> bool:
    return {(s["start"], s["end"]) for s in gold} == {(s["start"], s["end"]) for s in pred}


# --------------------------------------------------------------------------- #
# Response level
# --------------------------------------------------------------------------- #
def _binary(records: Iterable[Record]) -> dict[str, Any]:
    counts: Counter = Counter()
    for record in records:
        gold_positive = record.gold != "clean"
        if gold_positive and record.flagged:
            counts["tp"] += 1
        elif not gold_positive and record.flagged:
            counts["fp"] += 1
        elif gold_positive and not record.flagged:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    report = prf(counts["tp"], counts["fp"], counts["fn"])
    total = sum(counts.values())
    report["tn"] = counts["tn"]
    report["accuracy"] = (counts["tp"] + counts["tn"]) / total if total else 0.0
    report["rows"] = total
    return report


def response_metrics(records: list[Record], undergen_policy: str) -> dict[str, Any]:
    if undergen_policy not in UNDERGEN_POLICIES:
        raise ValueError(f"undergen_policy must be one of {UNDERGEN_POLICIES}")
    full = _binary(records)
    excluding = _binary(r for r in records if r.gold != "undergeneration")
    scored = [r for r in records if not (undergen_policy == "exclude" and r.gold == "undergeneration")]
    overall = excluding if undergen_policy == "exclude" else full

    by_class: dict[str, Any] = {}
    # clean: F1 of the "not flagged" decision against gold clean. Undergeneration
    # rows are always left out so the column means the same for every detector.
    pool = [r for r in records if r.gold != "undergeneration"]
    tp = sum(1 for r in pool if r.gold == "clean" and not r.flagged)
    fp = sum(1 for r in pool if r.gold != "clean" and not r.flagged)
    fn = sum(1 for r in pool if r.gold == "clean" and r.flagged)
    clean = prf(tp, fp, fn)
    clean["support"] = sum(1 for r in pool if r.gold == "clean")
    clean["definition"] = "F1 of the not-flagged decision vs gold clean, undergeneration rows excluded"
    by_class["clean"] = clean

    undergen_rows = [r for r in records if r.gold == "undergeneration"]
    for label in DEFECT_TYPES:
        support = sum(1 for r in records if r.gold == label)
        if label == "undergeneration" and undergen_policy != "score":
            by_class[label] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "support": support,
                "flag_rate": (
                    sum(1 for r in undergen_rows if r.flagged) / len(undergen_rows)
                    if undergen_rows
                    else None
                ),
                "definition": (
                    "not scored: undergeneration has no gold spans, a span-only detector "
                    "cannot express it; flag_rate = share of undergeneration rows on which "
                    "spans were (spuriously) emitted"
                ),
            }
            continue
        subset = [r for r in scored if r.gold in (label, "clean")]
        tp = sum(1 for r in subset if r.gold == label and r.flagged)
        fp = sum(1 for r in subset if r.gold == "clean" and r.flagged)
        fn = sum(1 for r in subset if r.gold == label and not r.flagged)
        report = prf(tp, fp, fn)
        report["support"] = support
        report["definition"] = f"detection F1 on gold in {{{label}, clean}}"
        by_class[label] = report

    # Avg. = mean of Correct, Mismatch, Overgen., Missing Tool for every detector:
    # undergeneration cannot be scored for span-only detectors, so it stays out of
    # the average for all of them (its F1 is still reported for class-aware ones).
    scores = [
        float(by_class[label]["f1"])
        for label in ("clean", *SPAN_TYPES)
        if by_class[label].get("f1") is not None and by_class[label].get("support")
    ]
    return {
        "undergen_policy": undergen_policy,
        "overall": overall,
        "overall_full": full,
        "overall_excluding_undergen": excluding,
        "by_class": by_class,
        "macro_class_score": sum(scores) / len(scores) if scores else None,
        "macro_definition": "mean F1 of clean, answer_mismatch, overgeneration, missing_tool (undergeneration excluded)",
        "undergeneration": {
            "rows": len(undergen_rows),
            "in_response_metrics": undergen_policy != "exclude",
            "flag_rate": (
                sum(1 for r in undergen_rows if r.flagged) / len(undergen_rows)
                if undergen_rows
                else None
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Class-aware metrics (decoder models)
# --------------------------------------------------------------------------- #
def class_metrics(records: list[Record]) -> dict[str, Any] | None:
    typed = [r for r in records if r.pred_classes is not None]
    if not typed:
        return None
    by_class: dict[str, Any] = {}
    scores: list[float] = []
    for label in VALID_TYPES:
        tp = sum(1 for r in typed if r.gold == label and label in r.pred_classes)
        fp = sum(1 for r in typed if r.gold != label and label in r.pred_classes)
        fn = sum(1 for r in typed if r.gold == label and label not in r.pred_classes)
        report = prf(tp, fp, fn)
        report["support"] = sum(1 for r in typed if r.gold == label)
        by_class[label] = report
        if report["support"]:
            scores.append(float(report["f1"]))

    type_accuracy: dict[str, Any] = {}
    hallucinated = [r for r in typed if r.gold != "clean"]
    for label in DEFECT_TYPES:
        rows = [r for r in typed if r.gold == label]
        type_accuracy[label] = (
            sum(1 for r in rows if label in r.pred_classes) / len(rows) if rows else None
        )
    type_accuracy["overall"] = (
        sum(1 for r in hallucinated if r.gold in r.pred_classes) / len(hallucinated)
        if hallucinated
        else None
    )

    confusion: dict[str, dict[str, int]] = {label: Counter() for label in VALID_TYPES}
    for r in typed:
        primary = r.pred_classes[0] if r.pred_classes else "clean"
        confusion[r.gold][primary] += 1
    return {
        "definition": "one-vs-rest on the predicted class; a row may predict several classes",
        "by_class": by_class,
        "macro_f1": sum(scores) / len(scores) if scores else None,
        "type_accuracy": type_accuracy,
        "confusion": {gold: dict(counts) for gold, counts in confusion.items()},
        "multi_class_rows": sum(1 for r in typed if len(r.pred_classes) > 1),
        "parse_failures": sum(1 for r in typed if r.parse_failed),
    }


# --------------------------------------------------------------------------- #
# Span level
# --------------------------------------------------------------------------- #
def span_metrics(records: list[Record], iou_threshold: float) -> dict[str, Any]:
    scopes = [*SPAN_TYPES, "undergeneration", "clean(FP source)"]
    scope_rows: dict[str, list[Record]] = {scope: [] for scope in scopes}
    scope_rows["ALL"] = records
    for record in records:
        if record.gold in scope_rows:
            scope_rows[record.gold].append(record)
        elif record.gold == "clean":
            scope_rows["clean(FP source)"].append(record)

    def compute_scope(scope: str, scope_data: list[Record]) -> dict[str, Any]:
        counts: Counter = Counter()
        gold_span_count = pred_span_count = 0
        char_scores: list[float] = []
        token_scores: list[float] = []
        exact_scores: list[int] = []
        for record in scope_data:
            gold, pred = record.gold_spans, record.pred_spans
            tp = match_spans(pred, gold, iou_threshold)
            counts["tp"] += tp
            counts["fp"] += len(pred) - tp + record.unlocatable
            counts["fn"] += len(gold) - tp
            gold_span_count += len(gold)
            pred_span_count += len(pred) + record.unlocatable
            if scope in SPAN_TYPES:
                char_scores.append(char_f1(gold, pred, record.answer))
                token_scores.append(token_f1(gold, pred, record.answer))
                exact_scores.append(int(exact_match(gold, pred)))
        report = prf(counts["tp"], counts["fp"], counts["fn"])
        report["span_f1"] = report["f1"]
        report["iou_threshold"] = iou_threshold
        report["support_rows"] = len(scope_data)
        report["gold_spans"] = gold_span_count
        report["pred_spans"] = pred_span_count
        report["unlocatable_pred_spans"] = sum(r.unlocatable for r in scope_data)
        report["char_overlap_f1"] = sum(char_scores) / len(char_scores) if char_scores else None
        report["token_f1"] = sum(token_scores) / len(token_scores) if token_scores else None
        report["exact_match_rate"] = sum(exact_scores) / len(exact_scores) if exact_scores else None
        return report

    overall = compute_scope("ALL", scope_rows["ALL"])
    by_class: dict[str, Any] = {}
    class_f1s: list[float] = []
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


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_metrics(
    records: list[Record],
    *,
    iou_threshold: float,
    undergen_policy: str,
    detector: str,
    meta: dict[str, Any],
    total_rows: int,
) -> dict[str, Any]:
    response = response_metrics(records, undergen_policy)
    classes = class_metrics(records) if detector == "class-aware" else None
    spans = span_metrics(records, iou_threshold)
    statuses = Counter(record.status for record in records)
    return round_numbers(
        {
            "total_rows": total_rows,
            "scored_rows": len(records),
            "skipped_rows": total_rows - len(records),
            "response_level_rows": response["overall"]["rows"],
            "status_counts": dict(statuses),
            "detector": detector,
            "undergen_policy": undergen_policy,
            "iou_threshold": iou_threshold,
            **meta,
            "response_level": response,
            "class_level": classes,
            "span_level": spans,
        }
    )


def table_row(metrics: dict[str, Any], setting: str | None = None, data_name: str | None = None) -> dict[str, Any]:
    response = metrics["response_level"]["by_class"]
    spans = metrics["span_level"]["by_class"]
    classes = metrics.get("class_level") or {}
    undergen = metrics["response_level"]["undergeneration"]
    undergen_note = (
        f"scored ({undergen['rows']})"
        if undergen["in_response_metrics"] and metrics["undergen_policy"] == "score"
        else f"positive, not in columns ({undergen['rows']})"
        if undergen["in_response_metrics"]
        else f"excluded ({undergen['rows']})"
    )
    return {
        "Setting": setting or metrics.get("setting") or "Transfer SFT",
        "Data": data_name or metrics.get("data_name") or metrics.get("dataset") or "ToolHACE",
        "Model": metrics.get("model") or metrics.get("checkpoint_path"),
        "Detector": metrics["detector"],
        "Response-level Correct": response["clean"]["f1"],
        "Response-level Mismatch": response["answer_mismatch"]["f1"],
        "Response-level Overgen.": response["overgeneration"]["f1"],
        "Response-level Missing Tool": response["missing_tool"]["f1"],
        "Response-level Undergen.": response["undergeneration"]["f1"],
        "Response-level Avg. (w/o Undergen.)": metrics["response_level"]["macro_class_score"],
        "Type Acc.": (classes.get("type_accuracy") or {}).get("overall"),
        "Span-level Mismatch": spans["answer_mismatch"]["span_f1"],
        "Span-level Overgen.": spans["overgeneration"]["span_f1"],
        "Span-level Missing Tool": spans["missing_tool"]["span_f1"],
        "Span-level Avg. (macro)": metrics["span_level"]["macro_span_f1"],
        "Span-level ALL (pooled)": metrics["span_level"]["overall"]["span_f1"],
        "Undergen. rows": undergen_note,
        "Scored rows": f"{metrics['response_level_rows']} resp / {metrics['scored_rows']} span",
    }


def format_table_value(value: Any) -> str:
    """Render scores as percentages with two decimals (0.5273 -> 52.73); ``None`` becomes an em dash."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{100 * value:.2f}"
    return str(value)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerow({key: format_table_value(row.get(key)) for key in TABLE_COLUMNS})


def print_summary(metrics: dict[str, Any], row: dict[str, Any]) -> None:
    response = metrics["response_level"]
    overall = response["overall"]
    print(
        f"Rows: {metrics['scored_rows']} scored / {metrics['total_rows']} total; "
        f"response-level rows: {metrics['response_level_rows']} "
        f"(undergen policy: {metrics['undergen_policy']}, detector: {metrics['detector']})"
    )
    if metrics["status_counts"].get("error"):
        print(f"WARNING: {metrics['status_counts']['error']} rows have status=error and count as not flagged")
    print(
        "Response-level: "
        f"P={100 * overall['precision']:.2f} R={100 * overall['recall']:.2f} "
        f"F1={100 * overall['f1']:.2f} Acc={100 * overall['accuracy']:.2f}"
    )
    if metrics["undergen_policy"] == "exclude":
        full = response["overall_full"]
        print(
            "  (with undergeneration rows counted as positives: "
            f"P={100 * full['precision']:.2f} R={100 * full['recall']:.2f} F1={100 * full['f1']:.2f} Acc={100 * full['accuracy']:.2f})"
        )
    else:
        excluding = response["overall_excluding_undergen"]
        print(
            "  (excluding undergeneration rows: "
            f"P={100 * excluding['precision']:.2f} R={100 * excluding['recall']:.2f} "
            f"F1={100 * excluding['f1']:.2f} Acc={100 * excluding['accuracy']:.2f})"
        )
    undergen = response["undergeneration"]
    if undergen["flag_rate"] is not None:
        print(f"Undergeneration rows flagged: {undergen['flag_rate']:.2%} of {undergen['rows']}")
    print("Per-class response-level F1:")
    for label in VALID_TYPES:
        report = response["by_class"][label]
        print(f"  {label:16s} F1={format_table_value(report['f1']):>5s}  (n={report['support']})")
    classes = metrics.get("class_level")
    if classes:
        print(
            f"Class-aware: macro F1={format_table_value(classes['macro_f1'])}, "
            f"type accuracy={format_table_value(classes['type_accuracy']['overall'])}, "
            f"parse failures={classes['parse_failures']}, multi-class rows={classes['multi_class_rows']}"
        )
    print(f"Response-level Avg. (w/o Undergen.): {format_table_value(row['Response-level Avg. (w/o Undergen.)'])}")
    print(f"Span-level @ IoU>{metrics['iou_threshold']}: Avg. (macro) {format_table_value(row['Span-level Avg. (macro)'])}, "
          f"ALL (pooled) {format_table_value(row['Span-level ALL (pooled)'])}")
    print("Table row:")
    print(json.dumps({key: format_table_value(row.get(key)) for key in TABLE_COLUMNS}, ensure_ascii=False))
