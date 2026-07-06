"""
Compute evaluation metrics for zero-shot / few-shot prediction JSONL files.

Metrics computed
----------------
1. Type classification (5-class):
   - Per-class Precision / Recall / F1
   - Macro / Weighted F1
   - Confusion matrix

2. Binary hallucination detection (clean vs. any hallucination):
   - Precision / Recall / F1 / Accuracy

3. Span detection (for rows where both gold and pred are non-clean, non-undergeneration):
   - Token-level F1  (split on whitespace)
   - Exact-match rate (predicted span set == gold span set)
   - Span-overlap F1 (intersection-over-union of character ranges)

Usage
-----
# Single file
python evaluate/compute_metrics.py \
    evaluate/results/zero_shot_predictions.jsonl

# Compare two files
python evaluate/compute_metrics.py \
    evaluate/results/zero_shot_predictions.jsonl \
    evaluate/results/few_shot_predictions.jsonl \
    --output evaluate/results/metrics_comparison.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
VALID_TYPES = ["clean", "answer_mismatch", "missing_tool", "overgeneration", "undergeneration"]
SPAN_TYPES = {"answer_mismatch", "missing_tool", "overgeneration"}  # types that carry spans


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------

def spans_to_char_set(spans: list[dict], answer: str) -> set[int]:
    """Return the set of character indices covered by the spans."""
    covered: set[int] = set()
    for sp in (spans or []):
        s, e = sp.get("start"), sp.get("end")
        if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(answer):
            covered.update(range(s, e))
    return covered


def span_token_set(spans: list[dict], answer: str) -> set[str]:
    """Return whitespace-tokenised words from the covered span text."""
    tokens: set[str] = set()
    for sp in (spans or []):
        text = sp.get("text") or answer[sp.get("start", 0):sp.get("end", 0)]
        tokens.update(t.lower() for t in text.split() if t)
    return tokens


def char_overlap_f1(gold_spans, pred_spans, answer: str) -> float:
    """Character-level overlap F1 between gold and predicted spans."""
    gold_chars = spans_to_char_set(gold_spans, answer)
    pred_chars = spans_to_char_set(pred_spans, answer)
    if not gold_chars and not pred_chars:
        return 1.0
    if not gold_chars or not pred_chars:
        return 0.0
    tp = len(gold_chars & pred_chars)
    precision = tp / len(pred_chars)
    recall = tp / len(gold_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_f1(gold_spans, pred_spans, answer: str) -> float:
    """Token-level F1 over whitespace-tokenised span text."""
    gold_tokens = span_token_set(gold_spans, answer)
    pred_tokens = span_token_set(pred_spans, answer)
    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0
    tp = len(gold_tokens & pred_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_span_match(gold_spans, pred_spans) -> bool:
    """True if the span sets are identical.

    Compares (start, end) offset pairs when available — offsets are this
    benchmark's ground truth, and text-set comparison silently collapses
    duplicate-text spans and drops spans with missing text (an empty gold
    set would 'exactly match' an empty prediction).
    """
    def span_set(spans):
        out = set()
        for sp in (spans or []):
            s, e = sp.get("start"), sp.get("end")
            if isinstance(s, int) and isinstance(e, int):
                out.add((s, e))
            else:
                t = str(sp.get("text", "")).strip()
                if t:
                    out.add(("text", t))
        return out

    gold, pred = span_set(gold_spans), span_set(pred_spans)
    return gold == pred


# ---------------------------------------------------------------------------
# Classification metrics (precision / recall / F1)
# ---------------------------------------------------------------------------

def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def classification_report(
    golds: list[str],
    preds: list[str],
    labels: list[str],
) -> dict[str, Any]:
    """Per-class P/R/F1 + macro/weighted aggregates."""
    counts: dict[str, dict[str, int]] = {
        lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in labels
    }
    total = len(golds)
    correct = 0
    for g, p in zip(golds, preds):
        if g == p:
            correct += 1
        for lbl in labels:
            if g == lbl and p == lbl:
                counts[lbl]["tp"] += 1
            elif g != lbl and p == lbl:
                counts[lbl]["fp"] += 1
            elif g == lbl and p != lbl:
                counts[lbl]["fn"] += 1

    report: dict[str, Any] = {}
    support: dict[str, int] = {lbl: golds.count(lbl) for lbl in labels}
    total_support = sum(support.values())

    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_f1 = 0.0

    for lbl in labels:
        p, r, f1 = prf(counts[lbl]["tp"], counts[lbl]["fp"], counts[lbl]["fn"])
        report[lbl] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": support[lbl]}
        macro_p += p
        macro_r += r
        macro_f1 += f1
        weighted_f1 += f1 * support[lbl]

    n = len(labels)
    report["macro"] = {
        "precision": round(macro_p / n, 4),
        "recall": round(macro_r / n, 4),
        "f1": round(macro_f1 / n, 4),
    }
    report["weighted"] = {
        "f1": round(weighted_f1 / total_support, 4) if total_support else 0.0,
    }
    report["accuracy"] = round(correct / total, 4) if total else 0.0
    return report


def confusion_matrix(golds: list[str], preds: list[str], labels: list[str]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {g: {p: 0 for p in labels + ["other"]} for g in labels}
    for g, p in zip(golds, preds):
        if g in matrix:
            col = p if p in labels else "other"
            matrix[g][col] += 1
    return matrix


# ---------------------------------------------------------------------------
# Main metric computation
# ---------------------------------------------------------------------------

UNPARSED = "<unparseable>"


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # PRINCIPLE (matches evaluate/verbalized/check_eval.py): unparseable or
    # errored predictions stay IN the denominator and count as wrong — they
    # are a model failure, not missing data. Dropping them silently inflates
    # every metric and makes numbers incomparable with check_eval outputs.
    scored_rows = [r for r in rows if r.get("gold_type")]
    error_count = sum(1 for r in scored_rows if r.get("status") != "ok")

    def pred_of(r: dict[str, Any]) -> str:
        if r.get("status") != "ok" or not r.get("pred_type"):
            return UNPARSED
        return r["pred_type"] if r["pred_type"] in VALID_TYPES else "unknown"

    golds_type = [r["gold_type"] for r in scored_rows]
    preds_type = [pred_of(r) for r in scored_rows]

    # --- 5-class type classification (UNPARSED is never a target class:
    #     it contributes FN to its gold class and FP to nothing) ---
    type_report = classification_report(golds_type, preds_type, VALID_TYPES)
    cm = confusion_matrix(golds_type, preds_type, VALID_TYPES + [UNPARSED])

    # --- Binary: clean (0) vs. hallucinated (1). Same convention as
    #     check_eval: an unparseable prediction never counts as a positive
    #     "hallucinated" call (FN on hallucinated gold, TN on clean gold).
    gold_binary = [0 if g == "clean" else 1 for g in golds_type]
    pred_binary = [1 if (p != UNPARSED and p != "clean") else 0 for p in preds_type]
    tp_b = sum(1 for g, p in zip(gold_binary, pred_binary) if g == 1 and p == 1)
    fp_b = sum(1 for g, p in zip(gold_binary, pred_binary) if g == 0 and p == 1)
    fn_b = sum(1 for g, p in zip(gold_binary, pred_binary) if g == 1 and p == 0)
    tn_b = sum(1 for g, p in zip(gold_binary, pred_binary) if g == 0 and p == 0)
    bp, br, bf1 = prf(tp_b, fp_b, fn_b)
    binary_accuracy = (tp_b + tn_b) / len(gold_binary) if gold_binary else 0.0

    binary_report = {
        "precision": round(bp, 4),
        "recall": round(br, 4),
        "f1": round(bf1, 4),
        "accuracy": round(binary_accuracy, 4),
        "tp": tp_b, "fp": fp_b, "fn": fn_b, "tn": tn_b,
    }

    ok_rows = [r for r in scored_rows if pred_of(r) != UNPARSED]

    # --- Span detection metrics: gold-span rows stay in the denominator
    #     even when the prediction failed to parse (pred spans = empty). ---
    span_rows = [
        r for r in ok_rows
        if r.get("gold_type") in SPAN_TYPES
        and r.get("pred_type") in SPAN_TYPES
    ]
    # All rows where gold has spans — including unparseable predictions
    span_rows_gold_only = [
        r for r in scored_rows
        if r.get("gold_type") in SPAN_TYPES
    ]

    char_f1_scores = []
    tok_f1_scores = []
    exact_matches = []

    def pred_spans_of(r: dict[str, Any]) -> list:
        if pred_of(r) not in SPAN_TYPES:
            return []
        return r.get("pred_span_labels", [])

    for r in span_rows_gold_only:
        answer = r.get("final_answer", "")
        gold_spans = r.get("gold_spans", [])
        pred_spans = pred_spans_of(r)

        char_f1_scores.append(char_overlap_f1(gold_spans, pred_spans, answer))
        tok_f1_scores.append(token_f1(gold_spans, pred_spans, answer))
        exact_matches.append(int(exact_span_match(gold_spans, pred_spans)))

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    span_report = {
        "n_gold_span_rows": len(span_rows_gold_only),
        "n_both_span_rows": len(span_rows),
        "char_overlap_f1_mean": safe_mean(char_f1_scores),
        "token_f1_mean": safe_mean(tok_f1_scores),
        "exact_match_rate": safe_mean(exact_matches),
    }

    # --- Per-type span metrics ---
    per_type_span: dict[str, Any] = {}
    for t in SPAN_TYPES:
        t_rows = [r for r in span_rows_gold_only if r.get("gold_type") == t]
        t_char, t_tok, t_exact = [], [], []
        for r in t_rows:
            answer = r.get("final_answer", "")
            gold_spans = r.get("gold_spans", [])
            pred_spans = pred_spans_of(r)
            t_char.append(char_overlap_f1(gold_spans, pred_spans, answer))
            t_tok.append(token_f1(gold_spans, pred_spans, answer))
            t_exact.append(int(exact_span_match(gold_spans, pred_spans)))
        per_type_span[t] = {
            "n": len(t_rows),
            "char_overlap_f1": safe_mean(t_char),
            "token_f1": safe_mean(t_tok),
            "exact_match_rate": safe_mean(t_exact),
        }

    return {
        "total_rows": len(rows),
        "ok_rows": len(ok_rows),
        "error_rows": error_count,
        "model": rows[0].get("model") if rows else None,
        "mode": rows[0].get("mode") if rows else None,
        "type_classification": type_report,
        "confusion_matrix": cm,
        "binary_hallucination": binary_report,
        "span_detection": span_report,
        "span_detection_per_type": per_type_span,
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_metrics(name: str, m: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name}  (model={m['model']}, mode={m['mode']})")
    print(f"{'=' * 60}")
    print(f"  Rows: total={m['total_rows']}  ok={m['ok_rows']}  errors={m['error_rows']}")

    print("\n--- Type Classification (5-class) ---")
    tc = m["type_classification"]
    header = f"  {'Type':<20} {'P':>7} {'R':>7} {'F1':>7} {'Support':>9}"
    print(header)
    print("  " + "-" * 55)
    for t in VALID_TYPES:
        r = tc.get(t, {})
        print(f"  {t:<20} {r.get('precision', 0):>7.4f} {r.get('recall', 0):>7.4f} {r.get('f1', 0):>7.4f} {r.get('support', 0):>9}")
    print("  " + "-" * 55)
    print(f"  {'macro':<20} {tc['macro']['precision']:>7.4f} {tc['macro']['recall']:>7.4f} {tc['macro']['f1']:>7.4f}")
    print(f"  {'weighted F1':<20} {'':>7} {'':>7} {tc['weighted']['f1']:>7.4f}")
    print(f"  Accuracy: {tc['accuracy']:.4f}")

    print("\n--- Binary (clean vs. hallucinated) ---")
    b = m["binary_hallucination"]
    print(f"  Precision={b['precision']:.4f}  Recall={b['recall']:.4f}  F1={b['f1']:.4f}  Accuracy={b['accuracy']:.4f}")
    print(f"  TP={b['tp']}  FP={b['fp']}  FN={b['fn']}  TN={b['tn']}")

    print("\n--- Span Detection ---")
    s = m["span_detection"]
    print(f"  Gold-span rows: {s['n_gold_span_rows']}  (pred also predicted span type: {s['n_both_span_rows']})")
    print(f"  Char-overlap F1 (mean): {s['char_overlap_f1_mean']}")
    print(f"  Token F1        (mean): {s['token_f1_mean']}")
    print(f"  Exact match     (rate): {s['exact_match_rate']}")

    print("\n--- Per-type Span ---")
    for t, ps in m["span_detection_per_type"].items():
        print(f"  {t:<20}  n={ps['n']}  char_f1={ps['char_overlap_f1']}  tok_f1={ps['token_f1']}  exact={ps['exact_match_rate']}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute evaluation metrics from prediction JSONL files.")
    p.add_argument("inputs", nargs="+", help="One or more prediction JSONL files.")
    p.add_argument("--output", default=None, help="Optional path to write JSON summary.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_metrics: dict[str, Any] = {}
    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"[WARNING] File not found: {input_path}")
            continue
        rows = load_predictions(input_path)
        if not rows:
            print(f"[WARNING] Empty file: {input_path}")
            continue
        m = compute_metrics(rows)
        name = input_path.stem
        all_metrics[name] = m
        print_metrics(name, m)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Metrics written to {out}")


if __name__ == "__main__":
    main()
