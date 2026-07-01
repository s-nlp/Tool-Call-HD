"""
Evaluate a trained LettuceDetect model on our dataset, and save rich
per-sample predictions for further offline analysis (no GPU needed).

This is evaluate.py plus:
  - a 5-class "type classification" report (Rule A: detection-based collapse,
    Rule B: span-overlap based) — see type_classification_report.py for the
    rule definitions
  - a richer --save-preds format (query, context, prompt, answer, split,
    task_type, gold, pred) so any future metric can be recomputed locally
    from the saved JSONL without re-running inference on the server

Accepts both LettuceDetect JSON format and raw RAGTruth JSONL.

Usage:
    python3 evaluate_save.py
    python3 evaluate_save.py --model ./output_lettuce --data lettucedetect_data/tool_calling_hallucination.json --split test
    python3 evaluate_save.py --data output_testing/type1_output.jsonl --by-type
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

# Some environments (e.g. older torch builds in unsloth images) ship a torch
# version that predates the float8_e8m0fnu dtype, which newer `transformers`
# releases reference unconditionally while importing their FP8 quantization
# integration (transformers/integrations/finegrained_fp8.py). That dtype is
# only used for FP8 microscaling, which ModernBERT token classification never
# exercises, so stub it out to unblock the import chain.
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.uint8

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent

# LettuceDetect path is resolved in this order:
#   1. --lettuce-path CLI arg
#   2. LETTUCE_PATH env variable
#   3. <project_root>/LettuceDetect (default)
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--lettuce-path", default=None)
_pre_args, _ = _pre.parse_known_args()
LETTUCE_PATH = (_pre_args.lettuce_path
                or os.environ.get("LETTUCE_PATH")
                or str(PROJECT_ROOT / "LettuceDetect"))
sys.path.insert(0, LETTUCE_PATH)

from lettucedetect.models.inference import HallucinationDetector


# ── Batch detector (from run_detector.py) ────────────────────────────────────

def _spans_from_token_preds(token_preds, probabilities, labels, offsets, answer_start_token, answer):
    if answer_start_token < offsets.size(0):
        answer_char_offset = offsets[answer_start_token][0].item()
    else:
        answer_char_offset = 0
    spans, current_span = [], None
    for i in range(answer_start_token, token_preds.size(0)):
        if labels[i].item() == -100:
            continue
        token_start, token_end = offsets[i].tolist()
        if token_start == token_end:
            continue
        rel_start = token_start - answer_char_offset
        rel_end   = token_end   - answer_char_offset
        is_hal    = token_preds[i].item() == 1
        if is_hal:
            if current_span is None:
                current_span = {"start": rel_start, "end": rel_end}
            else:
                current_span["end"] = rel_end
        else:
            if current_span is not None:
                current_span["text"] = answer[current_span["start"]:current_span["end"]]
                spans.append(current_span)
                current_span = None
    if current_span is not None:
        current_span["text"] = answer[current_span["start"]:current_span["end"]]
        spans.append(current_span)
    return spans


def predict_batch(detector, questions, contexts, answers, batch_size):
    from lettucedetect.datasets.hallucination_dataset import HallucinationDataset
    from lettucedetect.detectors.prompt_utils import PromptUtils

    inner     = detector.detector
    tokenizer = inner.tokenizer
    model     = inner.model
    device    = inner.device
    max_len   = inner.max_length
    pad_id    = tokenizer.pad_token_id or 0

    all_spans = []
    for i in tqdm(range(0, len(answers), batch_size), desc="Inference"):
        bq  = questions[i:i + batch_size]
        bc  = contexts [i:i + batch_size]
        ba  = answers  [i:i + batch_size]

        encodings, offsets_list, ans_starts = [], [], []
        for q, ctx, ans in zip(bq, bc, ba):
            prompt = PromptUtils.format_context(ctx, q, inner.lang)
            enc, _, offsets, ans_start = HallucinationDataset.prepare_tokenized_input(
                tokenizer, prompt, ans, max_len
            )
            encodings.append(enc)
            offsets_list.append(offsets)
            ans_starts.append(ans_start)

        max_seq = max(e["input_ids"].shape[1] for e in encodings)

        def pad(t, val):
            sz = max_seq - t.shape[1]
            return torch.cat([t, torch.full((1, sz), val, dtype=t.dtype)], dim=1)

        input_ids      = torch.cat([pad(e["input_ids"],      pad_id) for e in encodings], dim=0).to(device)
        attention_mask = torch.cat([pad(e["attention_mask"], 0)      for e in encodings], dim=0).to(device)

        labels_batch = []
        for j, enc in enumerate(encodings):
            seq_len = enc["input_ids"].shape[1]
            lbl = torch.full((max_seq,), -100, dtype=torch.long)
            lbl[ans_starts[j]:seq_len] = 0
            labels_batch.append(lbl)
        labels_tensor = torch.stack(labels_batch).to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        token_preds   = torch.where(labels_tensor == -100, labels_tensor, torch.argmax(logits, dim=-1))
        probabilities = torch.softmax(logits, dim=-1)

        for j, ans in enumerate(ba):
            spans = _spans_from_token_preds(
                token_preds[j], probabilities[j], labels_tensor[j],
                offsets_list[j], ans_starts[j], ans,
            )
            all_spans.append(spans)

    return all_spans


# ── Metrics ───────────────────────────────────────────────────────────────────

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2*p*r / (p+r)  if (p + r)  > 0 else 0.0
    return p, r, f


def response_metrics(samples, preds):
    """Binary: does the sample contain any hallucination? (response-level)"""
    tp = fp = fn = tn = 0
    for s, pred_raw in zip(samples, preds):
        gold_has = bool([l for l in s.get("labels", []) if not l.get("implicit_true")])
        pred_has = bool(pred_raw)
        if gold_has and pred_has:   tp += 1
        elif pred_has and not gold_has: fp += 1
        elif gold_has and not pred_has: fn += 1
        else:                       tn += 1
    p, r, f = _prf(tp, fp, fn)
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return p, r, f, acc, tp, fp, fn, tn


def char_metrics(samples, preds):
    """Character-level P/R/F1 averaged over samples."""
    ps = rs = fs = 0.0
    n = len(samples)
    for s, pred_raw in zip(samples, preds):
        gold = [(l["start"], l["end"]) for l in s.get("labels", []) if not l.get("implicit_true")]
        pred = [(sp["start"], sp["end"]) for sp in pred_raw] if pred_raw else []
        pred_chars = {c for s2, e2 in pred for c in range(s2, e2)}
        gold_chars = {c for s2, e2 in gold for c in range(s2, e2)}
        if not pred_chars and not gold_chars:
            ps += 1.0; rs += 1.0; fs += 1.0
            continue
        tp = len(pred_chars & gold_chars)
        p  = tp / len(pred_chars) if pred_chars else 0.0
        r  = tp / len(gold_chars)  if gold_chars  else 0.0
        f  = 2*p*r / (p+r) if (p+r) > 0 else 0.0
        ps += p; rs += r; fs += f
    return (ps/n, rs/n, fs/n) if n else (0.0, 0.0, 0.0)


def span_metrics(samples, preds, iou_threshold=0.0):
    """Span-level P/R/F1 — a predicted span matches a gold span if they overlap."""
    tp = fp = fn = 0
    for s, pred_raw in zip(samples, preds):
        gold = [(l["start"], l["end"]) for l in s.get("labels", []) if not l.get("implicit_true")]
        pred = [(sp["start"], sp["end"]) for sp in pred_raw] if pred_raw else []

        matched_gold = set()
        for ps2, pe2 in pred:
            hit = False
            for gi, (gs, ge) in enumerate(gold):
                if gi in matched_gold:
                    continue
                overlap = max(0, min(pe2, ge) - max(ps2, gs))
                if iou_threshold == 0.0:
                    match = overlap > 0
                else:
                    union = max(pe2, ge) - min(ps2, gs)
                    match = (overlap / union) >= iou_threshold if union > 0 else False
                if match:
                    matched_gold.add(gi)
                    hit = True
                    break
            if hit:
                tp += 1
            else:
                fp += 1
        fn += len(gold) - len(matched_gold)
    return _prf(tp, fp, fn)


def print_metrics(label, p, r, f, extra=""):
    print(f"  {label:<12}  P={p:.3f}  R={r:.3f}  F1={f:.3f}{extra}")


def eval_group(samples, preds):
    return char_metrics(samples, preds)


# ── 5-class type classification report ────────────────────────────────────────
# True label = task_type with the singlehop_/multihop_ prefix stripped.
# Lettuce predictions (`pred`) are binary spans only (no sub-type), so the
# predicted label can only ever be "clean" or the sample's own true type:
#
#   Rule A (detection-based collapse): predicted = "clean" if pred is empty,
#       else predicted = true type.
#   Rule B (span-overlap based): predicted = true type if (pred and gold both
#       empty) or a predicted span overlaps a gold span; else "clean".

TYPE_LABELS = ["clean", "answer_mismatch", "missing_tool", "overgeneration", "undergeneration"]


def base_type(task_type):
    for prefix in ("singlehop_", "multihop_"):
        if task_type.startswith(prefix):
            return task_type[len(prefix):]
    return task_type


def _spans_overlap(pred_spans, gold_spans):
    for p in pred_spans:
        for g in gold_spans:
            if max(p["start"], g["start"]) < min(p["end"], g["end"]):
                return True
    return False


def _predicted_label_a(true_label, pred_spans):
    return true_label if pred_spans else "clean"


def _predicted_label_b(true_label, pred_spans, gold_spans):
    if (not pred_spans and not gold_spans) or _spans_overlap(pred_spans, gold_spans):
        return true_label
    return "clean"


def classification_report(y_true, y_pred, labels, digits=2):
    name_width = max(len(l) for l in labels)
    width = max(name_width, len("weighted avg"), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width) + "\n\n"

    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"

    rows = []
    total_correct = 0
    total = len(y_true)
    p_w = r_w = f_w = 0.0
    p_m = r_m = f_m = 0.0

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append((label, precision, recall, f1, support))
        total_correct += tp

        p_m += precision
        r_m += recall
        f_m += f1
        p_w += precision * support
        r_w += recall * support
        f_w += f1 * support

    for label, p, r, f, s in rows:
        report += row_fmt.format(label, p, r, f, s, width=width, digits=digits)
    report += "\n"

    accuracy = total_correct / total if total > 0 else 0.0
    row_fmt_accuracy = "{:>{width}s} " + " {:>9.{digits}}" * 2 + " {:>9.{digits}f} {:>9}\n"
    report += row_fmt_accuracy.format("accuracy", "", "", accuracy, total, width=width, digits=digits)

    n_labels = len(labels)
    report += row_fmt.format("macro avg", p_m / n_labels, r_m / n_labels, f_m / n_labels, total,
                              width=width, digits=digits)
    report += row_fmt.format("weighted avg", p_w / total, r_w / total, f_w / total, total,
                              width=width, digits=digits)
    return report


def print_type_classification_report(samples, preds):
    y_true, y_pred_a, y_pred_b = [], [], []
    skipped = defaultdict(int)

    for s, pred_spans in zip(samples, preds):
        true_label = base_type(s.get("task_type", ""))
        if true_label not in TYPE_LABELS:
            skipped[s.get("task_type", "")] += 1
            continue
        gold_spans = s.get("labels", [])
        y_true.append(true_label)
        y_pred_a.append(_predicted_label_a(true_label, pred_spans))
        y_pred_b.append(_predicted_label_b(true_label, pred_spans, gold_spans))

    if not y_true:
        print("\n(skipping 5-class type classification report: no samples with a "
              "recognized clean/answer_mismatch/missing_tool/overgeneration/undergeneration task_type)")
        return

    if skipped:
        print(f"\n(5-class type report: skipped {sum(skipped.values())} samples with "
              f"unrecognized task_type: {dict(skipped)})")

    print("\n=== 5-class type classification (Rule A: detection-based collapse) ===")
    print(classification_report(y_true, y_pred_a, TYPE_LABELS))

    print("=== 5-class type classification (Rule B: span-overlap based) ===")
    print(classification_report(y_true, y_pred_b, TYPE_LABELS))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_samples(path, split):
    with open(path) as f:
        first = f.read(1); f.seek(0)
        rows = json.load(f) if first == "[" else [json.loads(l) for l in f if l.strip()]

    if rows and "prompt" not in rows[0]:
        # RAGTruth JSONL → convert
        converted = []
        for r in rows:
            output = r.get("output", "")
            if not output:
                continue
            try:
                raw_labs = json.loads(r.get("hallucination_labels", "[]"))
            except Exception:
                raw_labs = []
            labels = [{"start": l["start"], "end": l["end"]}
                      for l in raw_labs
                      if not l.get("implicit_true") and l.get("end", 0) > l.get("start", -1)]
            q   = r.get("query", "")
            ctx = r.get("context", "")
            converted.append({
                "prompt":    f"{q}\n\n{ctx}".strip() if q else ctx,
                "answer":    output,
                "labels":    labels,
                "split":     "test",
                "task_type": r.get("task_type", ""),
                "query":     q,
                "context":   ctx,
            })
        print(f"Converted {len(converted)} RAGTruth rows")
        rows = converted

    has_splits = any(s.get("split") for s in rows)
    samples = [s for s in rows if s.get("split") == split] if has_splits else rows
    return samples, has_splits


# ── Main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="./output_lettuce")
    p.add_argument("--data",         default=str(PROJECT_ROOT / "lettucedetect_data" / "tool_calling_hallucination.json"))
    p.add_argument("--split",        default="test", choices=["test", "dev", "train"])
    p.add_argument("--by-type",      action="store_true")
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--lettuce-path", default=LETTUCE_PATH,
                   help=f"Path to the LettuceDetect repo (default: {LETTUCE_PATH}). "
                        f"Can also be set via the LETTUCE_PATH env variable.")
    p.add_argument("--save-preds",   default=None,
                   help="Save predictions (+ full sample info) to this JSONL file so you can "
                        "re-evaluate / compute new metrics locally without re-running inference")
    p.add_argument("--load-preds",   default=None,
                   help="Load predictions from a previously saved file (skips inference)")
    return p.parse_args()


def main():
    args = get_args()

    print(f"Loading model: {args.model}")
    detector = HallucinationDetector(method="transformer", model_path=args.model)

    print(f"Loading data:  {args.data}")
    samples, has_splits = load_samples(args.data, args.split)

    if not samples:
        print(f"No samples for split='{args.split}'"); return

    print(f"Evaluating {len(samples)} samples (split={'all' if not has_splits else args.split})\n")

    questions = [s.get("query",   s.get("prompt", "")) for s in samples]
    contexts  = [[s.get("context", s.get("prompt", ""))] for s in samples]
    answers   = [s["answer"] for s in samples]

    if args.load_preds:
        print(f"Loading predictions from {args.load_preds} (skipping inference)")
        with open(args.load_preds) as f:
            all_preds = [json.loads(l)["pred"] for l in f if l.strip()]
        assert len(all_preds) == len(samples), \
            f"Prediction count {len(all_preds)} != sample count {len(samples)}"
    else:
        all_preds = predict_batch(detector, questions, contexts, answers, args.batch_size)
        if args.save_preds:
            with open(args.save_preds, "w") as f:
                for s, pred in zip(samples, all_preds):
                    f.write(json.dumps({
                        "query":     s.get("query", ""),
                        "context":   s.get("context", ""),
                        "prompt":    s.get("prompt", ""),
                        "answer":    s.get("answer", ""),
                        "split":     s.get("split", ""),
                        "task_type": s.get("task_type", ""),
                        "gold":      s.get("labels", []),
                        "pred":      pred,
                    }, ensure_ascii=False) + "\n")
            print(f"Predictions saved → {args.save_preds}")

    # ── Overall metrics at all three levels ───────────────────────────────────
    rp, rr, rf, acc, tp, fp, fn, tn = response_metrics(samples, all_preds)
    cp, cr, cf = char_metrics(samples, all_preds)
    sp, sr, sf = span_metrics(samples, all_preds)

    # type_macro_f1: macro-averaged F1 over both binary classes
    neg_p = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    neg_r = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    neg_f = 2 * neg_p * neg_r / (neg_p + neg_r) if (neg_p + neg_r) > 0 else 0.0
    type_macro_f1 = (rf + neg_f) / 2
    pred_hall_rate = (tp + fp) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    score = (type_macro_f1 + sf) / 2

    print(f"\n{'═'*60}")
    print(f"  OVERALL  ({len(samples)} samples)")
    print(f"{'─'*60}")
    print(f"  type_macro_f1   {type_macro_f1:.4f}")
    print(f"  span_f1         {sf:.4f}")
    print(f"  Score           {score:.4f}   (avg of type_macro_f1 & span_f1)")
    print(f"  type_accuracy   {acc:.4f}")
    print(f"  binary_f1       {rf:.4f}")
    print(f"  binary_precision {rp:.4f}")
    print(f"  binary_recall   {rr:.4f}")
    print(f"  char_f1         {cf:.4f}")
    print(f"  pred_hall_rate  {pred_hall_rate:.4f}")
    print(f"{'─'*60}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Char-level      P={cp:.4f}  R={cr:.4f}  F1={cf:.4f}")
    print(f"  Span-level      P={sp:.4f}  R={sr:.4f}  F1={sf:.4f}")
    print(f"{'═'*60}")

    print_type_classification_report(samples, all_preds)

    if args.by_type:
        by_type = defaultdict(list)
        for s, pred in zip(samples, all_preds):
            by_type[s.get("task_type", "?")].append((s, pred))

        print(f"\n{'─'*90}")
        print(f"  {'task_type':<25} {'n':>5}  {'tmacro':>7}  {'span':>7}  {'score':>7}  {'acc':>7}  {'bin_f1':>7}  {'char':>7}  {'hall%':>7}")
        print(f"{'─'*90}")
        for ttype, pairs in sorted(by_type.items()):
            samps, preds = zip(*pairs)
            _, _, rf2, acc2, tp2, fp2, fn2, tn2 = response_metrics(samps, preds)
            _, _, cf2 = char_metrics(samps, preds)
            _, _, sf2 = span_metrics(samps, preds)
            np2 = tn2 / (tn2 + fn2) if (tn2 + fn2) > 0 else 0.0
            nr2 = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0.0
            nf2 = 2 * np2 * nr2 / (np2 + nr2) if (np2 + nr2) > 0 else 0.0
            tm2 = (rf2 + nf2) / 2
            sc2 = (tm2 + sf2) / 2
            hr2 = (tp2 + fp2) / len(samps) if samps else 0.0
            print(f"  {ttype:<25} {len(samps):>5}  {tm2:>7.4f}  {sf2:>7.4f}  {sc2:>7.4f}  {acc2:>7.4f}  {rf2:>7.4f}  {cf2:>7.4f}  {hr2:>7.4f}")


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0 python3 evaluate_save.py \
#     --lettuce-path ./LettuceDetect \
#     --model        results_hallucination_detector \
#     --data         test.json \
#     --save-preds   test_predictions.jsonl \
#     --by-type
