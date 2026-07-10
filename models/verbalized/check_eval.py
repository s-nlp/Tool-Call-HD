#!/usr/bin/env python
"""Score a predictions JSONL from vllm_eval.py / eval.py.

Reports:
  CLASSIFICATION (from the preds file alone)
    - parse rate, overall accuracy
    - per-class precision / recall / F1 / support
    - macro-F1, weighted-F1, balanced accuracy
    - confusion matrix (gold x pred, incl. <unparseable>)
    - binary collapse: clean vs hallucinated (P/R/F1/acc)
    - parse-error breakdown

  SPAN-LEVEL (needs gold spans)
    Your spans are CHARACTER offsets into the final answer, so the primary
    metric is character-level micro P/R/F1 (it needs only the offset ranges).
    Gold spans aren't in the preds file, so we recover them by position from
    the SAME dataset the eval ran on (the gold answer is the last assistant
    turn of each row's `messages`). Alignment is verified via gold_type.
      - char-level micro P/R/F1 over hallucinated examples
      - the same restricted to rows where the TYPE was also predicted correctly
        (the actionable number — a span only helps if the class is right)
      - per-type span F1
      - exact-span-match rate
      - false-positive spans emitted on clean rows

  Usage:
    # classification only
    python check_eval.py --preds v1_preds.jsonl

    # + span eval (recover gold spans from the eval dataset, by position)
    python check_eval.py --preds v1_preds.jsonl \
        --data ./sft_data_full_V3_nothinking_qwen --split test

    # if your preds already carry "gold_spans" per row, --data isn't needed
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

UNPARSED = "<unparseable>"


# ----------------------------- JSON helpers --------------------------------
def _first_json_obj(s: str) -> str | None:
    start = s.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            esc = (c == "\\" and not esc)
            if c == '"' and not esc:
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return None


def parse_json(text: str):
    if not text:
        return None
    raw = _first_json_obj(text)
    if raw is None:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


# ----------------------------- span helpers --------------------------------
def extract_spans(obj) -> tuple[list[tuple[int, int]], int]:
    """Return ([(start,end), ...], n_unplaceable). Accepts start/end ints;
    counts text-only spans as unplaceable (we score on offsets)."""
    if not obj:
        return [], 0
    raw = obj.get("spans")
    if not isinstance(raw, list):
        return [], 0
    out, unplaceable = [], 0
    for s in raw:
        if isinstance(s, dict):
            st, en = s.get("start"), s.get("end")
            if isinstance(st, int) and isinstance(en, int) and en > st:
                out.append((st, en))
            else:
                unplaceable += 1
        else:
            unplaceable += 1
    return out, unplaceable


def merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    iv = sorted((s, e) for s, e in intervals if e > s)
    if not iv:
        return []
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def covered_len(merged: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in merged)


def overlap_len(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ----------------------------- loading -------------------------------------
def load_preds(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pred_type(row: dict) -> str:
    p = row.get("parsed")
    if isinstance(p, dict) and "type" in p:
        return p["type"]
    return UNPARSED


def gold_from_dataset(data: str, split: str, n_expected: int):
    """Return per-row (gold_type, [(s,e),...]) aligned to dataset order, by
    parsing the last assistant turn (the gold target) of each row's messages."""
    from datasets import load_from_disk
    ds = load_from_disk(data)
    if split not in ds:
        raise SystemExit(f"split {split!r} not in dataset {data}")
    sp = ds[split]
    if len(sp) != n_expected:
        raise SystemExit(
            f"dataset {split} has {len(sp)} rows but preds has {n_expected}. "
            f"They must match for position alignment — make sure --data/--split "
            f"are the exact eval inputs and the run wasn't subsampled.")
    out = []
    msgs_col = sp["messages"]
    type_col = sp["type"] if "type" in sp.column_names else [None] * len(sp)
    for i in range(len(sp)):
        gold_obj = parse_json(msgs_col[i][-1]["content"])
        spans, _ = extract_spans(gold_obj)
        gtype = type_col[i] if type_col[i] is not None else \
            (gold_obj.get("type") if gold_obj else None)
        out.append((gtype, spans))
    return out


# ----------------------------- reports -------------------------------------
def classification_report(preds: list[dict]) -> dict:
    n = len(preds)
    golds = [r.get("gold_type") for r in preds]
    classes = sorted({g for g in golds if g is not None})
    pred_labels = [pred_type(r) for r in preds]

    n_parsed = sum(1 for p in pred_labels if p != UNPARSED)
    n_correct = sum(1 for g, p in zip(golds, pred_labels) if g == p)

    # per-class P/R/F1 (unparseable counts as a wrong prediction, never a class)
    tp = Counter(); fp = Counter(); fn = Counter(); support = Counter()
    for g, p in zip(golds, pred_labels):
        support[g] += 1
        if p == g:
            tp[g] += 1
        else:
            fn[g] += 1
            if p in classes:
                fp[p] += 1
    per_class = {}
    recalls = []
    for c in classes:
        P, R, F = prf(tp[c], fp[c], fn[c])
        per_class[c] = {"precision": P, "recall": R, "f1": F,
                        "support": support[c]}
        recalls.append(R)
    macro_f1 = sum(per_class[c]["f1"] for c in classes) / len(classes)
    weighted_f1 = sum(per_class[c]["f1"] * support[c] for c in classes) / n
    balanced_acc = sum(recalls) / len(recalls) if recalls else 0.0

    # confusion matrix
    cm = defaultdict(Counter)
    for g, p in zip(golds, pred_labels):
        cm[g][p] += 1

    # binary collapse: hallucinated (any non-clean) is positive
    def is_h(t):
        return t is not None and t != "clean"
    bt = Counter()
    for g, p in zip(golds, pred_labels):
        gh, ph = is_h(g), (p != UNPARSED and p != "clean")
        bt[("TP" if gh and ph else "FN" if gh and not ph
            else "FP" if (not gh) and ph else "TN")] += 1
    bP, bR, bF = prf(bt["TP"], bt["FP"], bt["FN"])
    bin_acc = (bt["TP"] + bt["TN"]) / n

    parse_errors = Counter(r.get("parse_error") for r in preds
                           if r.get("parse_error"))

    return {
        "n": n, "parse_rate": n_parsed / n, "accuracy": n_correct / n,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc, "per_class": per_class,
        "classes": classes, "confusion": {g: dict(cm[g]) for g in cm},
        "binary": {"precision": bP, "recall": bR, "f1": bF, "accuracy": bin_acc,
                   **{k: bt[k] for k in ("TP", "FP", "FN", "TN")}},
        "parse_errors": dict(parse_errors.most_common()),
    }


def span_report(preds: list[dict], gold_spans: list[list[tuple[int, int]]]) -> dict:
    """Char-level micro P/R/F1. gold_spans[i] aligns with preds[i]."""
    overall = Counter()          # tp/fp/fn over hallucinated rows
    type_correct = Counter()     # same, restricted to correct-type rows
    per_type = defaultdict(Counter)
    exact_match_h = exact_total_h = 0
    clean_fp_rows = clean_total = 0
    unplaceable_pred = 0

    for r, gs in zip(preds, gold_spans):
        g_type = r.get("gold_type")
        p_type = pred_type(r)
        pspans, unpl = extract_spans(r.get("parsed"))
        unplaceable_pred += unpl
        gmerged = merge(gs)
        pmerged = merge(pspans)
        gl, pl = covered_len(gmerged), covered_len(pmerged)
        ov = overlap_len(gmerged, pmerged)
        tp, fp, fn = ov, pl - ov, gl - ov

        if g_type == "clean" or not gs:
            clean_total += 1
            if pl > 0:
                clean_fp_rows += 1
            continue

        # hallucinated row
        overall["tp"] += tp; overall["fp"] += fp; overall["fn"] += fn
        per_type[g_type]["tp"] += tp
        per_type[g_type]["fp"] += fp
        per_type[g_type]["fn"] += fn
        exact_total_h += 1
        if gmerged == pmerged:
            exact_match_h += 1
        if p_type == g_type:
            type_correct["tp"] += tp
            type_correct["fp"] += fp
            type_correct["fn"] += fn

    oP, oR, oF = prf(overall["tp"], overall["fp"], overall["fn"])
    tP, tR, tF = prf(type_correct["tp"], type_correct["fp"], type_correct["fn"])
    pt = {}
    for t, c in per_type.items():
        P, R, F = prf(c["tp"], c["fp"], c["fn"])
        pt[t] = {"precision": P, "recall": R, "f1": F}

    return {
        "char_micro": {"precision": oP, "recall": oR, "f1": oF,
                       **dict(overall)},
        "char_micro_type_correct": {"precision": tP, "recall": tR, "f1": tF,
                                    **dict(type_correct)},
        "per_type": pt,
        "exact_match_rate": exact_match_h / exact_total_h if exact_total_h else 0.0,
        "n_hallucinated": exact_total_h,
        "clean_false_positive_rows": clean_fp_rows,
        "n_clean": clean_total,
        "unplaceable_pred_spans": unplaceable_pred,
    }


# ----------------------------- printing ------------------------------------
def pct(x):
    return f"{x:6.2%}"


def print_classification(c: dict) -> None:
    print("\n================  CLASSIFICATION  ================")
    print(f"examples            {c['n']}")
    print(f"parse rate          {pct(c['parse_rate'])}")
    print(f"accuracy            {pct(c['accuracy'])}")
    print(f"macro-F1            {pct(c['macro_f1'])}")
    print(f"weighted-F1         {pct(c['weighted_f1'])}")
    print(f"balanced accuracy   {pct(c['balanced_accuracy'])}")
    print("\nper-class:")
    print(f"  {'class':22s} {'P':>7s} {'R':>7s} {'F1':>7s} {'support':>8s}")
    for cls in c["classes"]:
        m = c["per_class"][cls]
        print(f"  {cls:22s} {pct(m['precision'])} {pct(m['recall'])} "
              f"{pct(m['f1'])} {m['support']:>8d}")
    print("\nbinary (clean vs hallucinated):")
    b = c["binary"]
    print(f"  acc {pct(b['accuracy'])}  P {pct(b['precision'])}  "
          f"R {pct(b['recall'])}  F1 {pct(b['f1'])}  "
          f"(TP={b['TP']} FP={b['FP']} FN={b['FN']} TN={b['TN']})")
    print("\nconfusion matrix (rows=gold, cols=pred):")
    cols = c["classes"] + [UNPARSED]
    head = "  {:18s}".format("gold\\pred") + "".join(f"{x[:10]:>11s}" for x in cols)
    print(head)
    for g in c["classes"]:
        row = c["confusion"].get(g, {})
        print("  {:18s}".format(g) +
              "".join(f"{row.get(x,0):>11d}" for x in cols))
    if c["parse_errors"]:
        print("\nparse errors:", c["parse_errors"])


def print_span(s: dict) -> None:
    print("\n================  SPAN-LEVEL (char offsets)  ================")
    m = s["char_micro"]
    print(f"hallucinated rows scored: {s['n_hallucinated']}")
    print(f"char micro   P {pct(m['precision'])}  R {pct(m['recall'])}  "
          f"F1 {pct(m['f1'])}   (tp={m['tp']} fp={m['fp']} fn={m['fn']})")
    t = s["char_micro_type_correct"]
    print(f"  type-correct subset only:  "
          f"P {pct(t['precision'])}  R {pct(t['recall'])}  F1 {pct(t['f1'])}")
    print(f"exact-span-match rate     {pct(s['exact_match_rate'])}")
    print(f"clean rows with false-positive spans: "
          f"{s['clean_false_positive_rows']}/{s['n_clean']}")
    if s["unplaceable_pred_spans"]:
        print(f"⚠ {s['unplaceable_pred_spans']} predicted spans had no "
              f"usable start/end (text-only or malformed); not scored.")
    print("\nper-type span F1:")
    print(f"  {'type':22s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    for t_, mm in sorted(s["per_type"].items()):
        print(f"  {t_:22s} {pct(mm['precision'])} {pct(mm['recall'])} "
              f"{pct(mm['f1'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="predictions JSONL")
    ap.add_argument("--data", default=None,
                    help="DatasetDict the eval ran on (for gold spans, by "
                         "position). Omit to skip span eval, or if preds rows "
                         "already contain a 'gold_spans' field.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    args = ap.parse_args()

    preds = load_preds(args.preds)
    print(f"Loaded {len(preds)} predictions from {args.preds}")

    cls = classification_report(preds)
    print_classification(cls)

    metrics = {"classification": cls}

    # Resolve gold spans: embedded in preds, or recovered from --data.
    gold_spans = None
    if all("gold_spans" in r for r in preds):
        gold_spans = [[tuple(x) for x in r["gold_spans"]] for r in preds]
        print("\n(using gold_spans embedded in the preds file)")
    elif args.data:
        gold = gold_from_dataset(args.data, args.split, len(preds))
        # verify alignment via gold_type agreement
        agree = sum(1 for r, (gt, _) in zip(preds, gold)
                    if r.get("gold_type") == gt)
        rate = agree / len(preds)
        print(f"\nalignment check: gold_type agreement {rate:.1%} "
              f"({agree}/{len(preds)})")
        if rate < 0.99:
            print("⚠ low agreement — preds and --data may be misaligned or a "
                  "different split; span numbers may be wrong.", file=sys.stderr)
        gold_spans = [g for _, g in gold]

    if gold_spans is not None:
        span = span_report(preds, gold_spans)
        print_span(span)
        metrics["span"] = span
    else:
        print("\n(span eval skipped — pass --data to enable it)")

    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        print(f"\nWrote metrics to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())