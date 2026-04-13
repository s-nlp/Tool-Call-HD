"""
Evaluate hallucination detector predictions against gold spans.

Input CSV must have columns:
  pred  – JSON list of predicted spans: [{"start": int, "end": int, ...}, ...]
  gold  – JSON list of gold spans (same format)

Metrics
-------
1. Binary hallucination F1
   label = 1 if len(spans) > 0 else 0
   sklearn classification_report (includes precision, recall, F1 per class + macro)

2. Character-level F1 (charF1)
   For each row: union all gold spans → char set G, union all pred spans → char set P.
   char_p = |P ∩ G| / |P|,  char_r = |P ∩ G| / |G|,  char_f1 = harmonic mean.
   Handles many-to-one and boundary misalignment gracefully; always defined.
   Global metric: intersection/pred_chars and intersection/gold_chars summed across rows.

3. Span-level F1 (--iou-threshold, default 0.75)
   For each row, greedily match pred spans to gold spans by pairwise IoU.
   A pair is a TP if IoU >= threshold. Unmatched pred = FP, unmatched gold = FN.
   TP/FP/FN are summed across all rows, then P/R/F1 are computed globally.

Usage
-----
python scripts/evaluate.py --predictions predictions/lettuce_large.csv
"""

import argparse
import ast
import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def spans_to_char_set(spans: list[dict]) -> set[int]:
    chars: set[int] = set()
    for s in spans:
        chars.update(range(s["start"], s["end"]))
    return chars


def char_prf(pred: list[dict], gold: list[dict]) -> tuple[float | None, float | None, float]:
    """Character-level precision, recall, F1 for one row.

    Returns (char_p, char_r, char_f1).
    char_p is None when pred is empty but gold is non-empty (FN).
    char_r is None when gold is empty but pred is non-empty (FP).
    char_f1 is always defined: 1.0 for TN, 0.0 for pure FP/FN.
    """
    if not pred and not gold:
        return 1.0, 1.0, 1.0

    pred_chars = spans_to_char_set(pred)
    gold_chars = spans_to_char_set(gold)
    inter = len(pred_chars & gold_chars)

    p = inter / len(pred_chars) if pred_chars else None
    r = inter / len(gold_chars) if gold_chars else None

    if p is None or r is None:
        f = 0.0
    elif p + r == 0:
        f = 0.0
    else:
        f = 2 * p * r / (p + r)

    return p, r, f


def span_pair_iou(a: dict, b: dict) -> float:
    inter_start = max(a["start"], b["start"])
    inter_end = min(a["end"], b["end"])
    inter = max(0, inter_end - inter_start)
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def span_f1_counts(pred: list[dict], gold: list[dict], threshold: float) -> tuple[int, int, int]:
    """Return (TP, FP, FN) for one row using greedy IoU matching."""
    if not pred and not gold:
        return 0, 0, 0

    # Build all pairs sorted by IoU descending
    pairs = sorted(
        ((span_pair_iou(p, g), pi, gi) for pi, p in enumerate(pred) for gi, g in enumerate(gold)),
        reverse=True,
    )

    matched_pred, matched_gold = set(), set()
    tp = 0
    for iou, pi, gi in pairs:
        if iou < threshold:
            break
        if pi not in matched_pred and gi not in matched_gold:
            tp += 1
            matched_pred.add(pi)
            matched_gold.add(gi)

    fp = len(pred) - len(matched_pred)
    fn = len(gold) - len(matched_gold)
    return tp, fp, fn


def parse_spans(cell) -> list[dict]:
    if isinstance(cell, list):
        return cell
    if pd.isna(cell) or cell == "":
        return []
    try:
        return json.loads(cell)
    except json.JSONDecodeError:
        return ast.literal_eval(cell)


def prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    p = tp / (tp + fp) if (tp + fp) > 0 else None
    r = tp / (tp + fn) if (tp + fn) > 0 else None
    f = 2 * p * r / (p + r) if p is not None and r is not None and (p + r) > 0 else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser(description="Evaluate hallucination detector predictions.")
    ap.add_argument("--predictions", required=True, help="Path to predictions CSV file.")
    ap.add_argument("--pred-col", default="pred", help="Column with predicted spans (default: pred).")
    ap.add_argument("--gold-col", default="gold", help="Column with gold spans (default: gold).")
    ap.add_argument("--iou-threshold", type=float, default=0.75,
                    help="IoU threshold for span matching in span-F1 (default: 0.75).")
    ap.add_argument("--output", default=None,
                    help="Optional path to save per-row metrics CSV.")
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)

    for col in (args.pred_col, args.gold_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {args.predictions}")

    pred_spans = df[args.pred_col].map(parse_spans)
    gold_spans = df[args.gold_col].map(parse_spans)

    total_inter = total_pred_chars = total_gold_chars = 0

    rows = []
    for ps, gs in zip(pred_spans, gold_spans):
        # charF1
        c_p, c_r, c_f1 = char_prf(ps, gs)
        pred_chars = spans_to_char_set(ps)
        gold_chars = spans_to_char_set(gs)
        inter = len(pred_chars & gold_chars)
        total_inter += inter
        total_pred_chars += len(pred_chars)
        total_gold_chars += len(gold_chars)

        # span F1
        s_tp, s_fp, s_fn = span_f1_counts(ps, gs, args.iou_threshold)
        if not ps and not gs:
            s_p, s_r, s_f1 = 1.0, 1.0, 1.0
        else:
            s_p, s_r, s_f1 = prf(s_tp, s_fp, s_fn)

        r_pred = int(len(ps) > 0)
        r_gold = int(len(gs) > 0)

        rows.append({
            "char_p":        c_p,   "char_r":  c_r,   "char_f1": c_f1,
            "span_tp":       s_tp,  "span_fp": s_fp,  "span_fn": s_fn,
            "span_p":        s_p,   "span_r":  s_r,   "span_f1": s_f1,
            "response_pred": r_pred, "response_gold": r_gold,
            "response_correct": int(r_pred == r_gold),
        })

    metrics = pd.DataFrame(rows)

    # --- Binary hallucination F1 ---
    print("=== Binary hallucination F1 ===")
    print(classification_report(
        metrics["response_gold"], metrics["response_pred"],
        labels=[0, 1],
        target_names=["no hallucination", "hallucination"],
        zero_division=0,
    ))

    # --- CharF1 (global micro) ---
    g_char_p = total_inter / total_pred_chars if total_pred_chars > 0 else None
    g_char_r = total_inter / total_gold_chars if total_gold_chars > 0 else None
    g_char_f1 = (2 * g_char_p * g_char_r / (g_char_p + g_char_r)
                 if g_char_p and g_char_r else 0.0)

    print("=== CharF1 (character-level) ===")
    print(f"Precision : {g_char_p:.4f}  ({total_inter} / {total_pred_chars} pred chars)")
    print(f"Recall    : {g_char_r:.4f}  ({total_inter} / {total_gold_chars} gold chars)")
    print(f"F1        : {g_char_f1:.4f}")

    # --- Span-level F1 (global, summing TP/FP/FN) ---
    total_tp = metrics["span_tp"].sum()
    total_fp = metrics["span_fp"].sum()
    total_fn = metrics["span_fn"].sum()
    g_p, g_r, g_f1 = prf(total_tp, total_fp, total_fn)

    print(f"\n=== Span F1 (IoU threshold={args.iou_threshold}) ===")
    print(f"Precision : {g_p:.4f}  ({total_tp} TP / {total_tp + total_fp} pred)")
    print(f"Recall    : {g_r:.4f}  ({total_tp} TP / {total_tp + total_fn} gold)")
    print(f"F1        : {g_f1:.4f}")

    # --- Optional per-row output ---
    if args.output:
        out = pd.concat([df.reset_index(drop=True), metrics], axis=1)
        out.to_csv(args.output, index=False)
        print(f"\nPer-row metrics saved to {args.output}")


if __name__ == "__main__":
    main()
