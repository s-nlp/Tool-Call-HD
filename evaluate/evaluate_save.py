from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from toolhace_lettuce_utils import DEFAULT_HF_DATASET, DEFAULT_HF_SPLIT, load_rows, row_to_lettuce_example

TYPE_LABELS = ["clean", "answer_mismatch", "missing_tool", "overgeneration", "undergeneration"]


def _spans_from_token_preds(token_preds, probabilities, labels, offsets, answer_start_token, answer):
    if answer_start_token < offsets.size(0):
        answer_char_offset = offsets[answer_start_token][0].item()
    else:
        answer_char_offset = 0

    spans = []
    current_span = None
    for i in range(answer_start_token, token_preds.size(0)):
        if labels[i].item() == -100:
            continue
        token_start, token_end = offsets[i].tolist()
        if token_start == token_end:
            continue
        rel_start = token_start - answer_char_offset
        rel_end = token_end - answer_char_offset
        is_hal = token_preds[i].item() == 1
        if is_hal:
            if current_span is None:
                current_span = {"start": rel_start, "end": rel_end}
            else:
                current_span["end"] = rel_end
        elif current_span is not None:
            current_span["text"] = answer[current_span["start"]:current_span["end"]]
            spans.append(current_span)
            current_span = None

    if current_span is not None:
        current_span["text"] = answer[current_span["start"]:current_span["end"]]
        spans.append(current_span)
    return spans


def predict_batch(detector, questions, contexts, answers, batch_size):
    import torch
    from tqdm.auto import tqdm
    from lettucedetect.datasets.hallucination_dataset import HallucinationDataset
    from lettucedetect.detectors.prompt_utils import PromptUtils

    inner = detector.detector
    tokenizer = inner.tokenizer
    model = inner.model
    device = inner.device
    max_length = inner.max_length
    pad_id = tokenizer.pad_token_id or 0

    all_spans = []
    for i in tqdm(range(0, len(answers), batch_size), desc="Inference", unit="batch"):
        batch_questions = questions[i:i + batch_size]
        batch_contexts = contexts[i:i + batch_size]
        batch_answers = answers[i:i + batch_size]

        encodings = []
        offsets_list = []
        answer_starts = []
        for question, context_list, answer in zip(batch_questions, batch_contexts, batch_answers):
            prompt = PromptUtils.format_context(context_list, question or None, inner.lang)
            encoding, _, offsets, answer_start = HallucinationDataset.prepare_tokenized_input(
                tokenizer, prompt, answer, max_length
            )
            encodings.append(encoding)
            offsets_list.append(offsets)
            answer_starts.append(answer_start)

        max_seq = max(encoding["input_ids"].shape[1] for encoding in encodings)

        def pad(tensor, value):
            missing = max_seq - tensor.shape[1]
            return torch.cat([tensor, torch.full((1, missing), value, dtype=tensor.dtype)], dim=1)

        input_ids = torch.cat([pad(encoding["input_ids"], pad_id) for encoding in encodings], dim=0).to(device)
        attention_mask = torch.cat(
            [pad(encoding["attention_mask"], 0) for encoding in encodings], dim=0
        ).to(device)

        labels_batch = []
        for answer_start, encoding in zip(answer_starts, encodings):
            seq_len = encoding["input_ids"].shape[1]
            labels = torch.full((max_seq,), -100, dtype=torch.long)
            labels[answer_start:seq_len] = 0
            labels_batch.append(labels)
        labels_tensor = torch.stack(labels_batch).to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        token_preds = torch.where(labels_tensor == -100, labels_tensor, torch.argmax(logits, dim=-1))
        probabilities = torch.softmax(logits, dim=-1)

        for offsets, answer_start, answer, pred_row, prob_row, label_row in zip(
            offsets_list, answer_starts, batch_answers, token_preds, probabilities, labels_tensor
        ):
            all_spans.append(
                _spans_from_token_preds(pred_row, prob_row, label_row, offsets, answer_start, answer)
            )

    return all_spans


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def response_metrics(samples, preds):
    tp = fp = fn = tn = 0
    for sample, pred_spans in zip(samples, preds):
        gold_has = bool(sample["gold_spans"])
        pred_has = bool(pred_spans)
        if gold_has and pred_has:
            tp += 1
        elif pred_has and not gold_has:
            fp += 1
        elif gold_has and not pred_has:
            fn += 1
        else:
            tn += 1
    precision, recall, f1 = prf(tp, fp, fn)
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return precision, recall, f1, accuracy, tp, fp, fn, tn


def char_metrics(samples, preds):
    ps = rs = fs = 0.0
    total = len(samples)
    for sample, pred_spans in zip(samples, preds):
        gold_chars = {c for sp in sample["gold_spans"] for c in range(sp["start"], sp["end"])}
        pred_chars = {c for sp in pred_spans for c in range(sp["start"], sp["end"])}
        if not gold_chars and not pred_chars:
            ps += 1.0
            rs += 1.0
            fs += 1.0
            continue
        tp = len(gold_chars & pred_chars)
        precision = tp / len(pred_chars) if pred_chars else 0.0
        recall = tp / len(gold_chars) if gold_chars else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        ps += precision
        rs += recall
        fs += f1
    return (ps / total, rs / total, fs / total) if total else (0.0, 0.0, 0.0)


def span_metrics(samples, preds):
    tp = fp = fn = 0
    for sample, pred_spans in zip(samples, preds):
        gold = [(span["start"], span["end"]) for span in sample["gold_spans"]]
        pred = [(span["start"], span["end"]) for span in pred_spans]
        matched_gold = set()
        for pred_start, pred_end in pred:
            hit = False
            for gold_idx, (gold_start, gold_end) in enumerate(gold):
                if gold_idx in matched_gold:
                    continue
                overlap = max(0, min(pred_end, gold_end) - max(pred_start, gold_start))
                if overlap > 0:
                    matched_gold.add(gold_idx)
                    hit = True
                    break
            if hit:
                tp += 1
            else:
                fp += 1
        fn += len(gold) - len(matched_gold)
    return prf(tp, fp, fn)


def base_type(value: str | None) -> str:
    if not value:
        return "clean"
    for prefix in ("singlehop_", "multihop_"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def spans_overlap(pred_spans, gold_spans) -> bool:
    for pred in pred_spans:
        for gold in gold_spans:
            if max(pred["start"], gold["start"]) < min(pred["end"], gold["end"]):
                return True
    return False


def classification_report(y_true, y_pred, labels, digits=2):
    name_width = max(len(label) for label in labels)
    width = max(name_width, len("weighted avg"), digits)
    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width) + "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"

    rows = []
    total_correct = 0
    total = len(y_true)
    p_weighted = r_weighted = f_weighted = 0.0
    p_macro = r_macro = f_macro = 0.0

    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        support = sum(1 for truth in y_true if truth == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append((label, precision, recall, f1, support))
        total_correct += tp
        p_macro += precision
        r_macro += recall
        f_macro += f1
        p_weighted += precision * support
        r_weighted += recall * support
        f_weighted += f1 * support

    for label, precision, recall, f1, support in rows:
        report += row_fmt.format(label, precision, recall, f1, support, width=width, digits=digits)
    report += "\n"

    accuracy = total_correct / total if total > 0 else 0.0
    row_fmt_accuracy = "{:>{width}s} " + " {:>9.{digits}}" * 2 + " {:>9.{digits}f} {:>9}\n"
    report += row_fmt_accuracy.format("accuracy", "", "", accuracy, total, width=width, digits=digits)
    report += row_fmt.format(
        "macro avg", p_macro / len(labels), r_macro / len(labels), f_macro / len(labels), total,
        width=width, digits=digits
    )
    report += row_fmt.format(
        "weighted avg", p_weighted / total, r_weighted / total, f_weighted / total, total,
        width=width, digits=digits
    )
    return report


def print_type_classification_report(samples, preds):
    y_true = []
    y_pred_a = []
    y_pred_b = []
    skipped = defaultdict(int)

    for sample, pred_spans in zip(samples, preds):
        true_label = base_type(sample["gold_type"])
        if true_label not in TYPE_LABELS:
            skipped[sample["gold_type"] or ""] += 1
            continue
        gold_spans = sample["gold_spans"]
        y_true.append(true_label)
        y_pred_a.append(true_label if pred_spans else "clean")
        y_pred_b.append(true_label if ((not pred_spans and not gold_spans) or spans_overlap(pred_spans, gold_spans)) else "clean")

    if not y_true:
        print("\nSkipping 5-class type classification report: no recognized labels.")
        return

    if skipped:
        print(f"\nSkipped {sum(skipped.values())} rows with unrecognized labels: {dict(skipped)}")

    print("\n=== 5-class type classification (Rule A: detection-based collapse) ===")
    print(classification_report(y_true, y_pred_a, TYPE_LABELS))
    print("=== 5-class type classification (Rule B: span-overlap based) ===")
    print(classification_report(y_true, y_pred_b, TYPE_LABELS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a LettuceDetect model on ToolHACE unified rows and optionally save predictions."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Local ToolHACE file (.json, .jsonl, or .parquet). If omitted, load from Hugging Face.",
    )
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-split", default=DEFAULT_HF_SPLIT)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--model",
        default="s-nlp/tool-calling-hallucination-modernbert-base-unified-final",
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--row-split",
        default="all",
        choices=["all", "train", "dev", "test"],
        help="Optional filter over a local mixed-split file.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-preds", default=None, help="Save rich predictions to JSONL.")
    parser.add_argument("--load-preds", default=None, help="Load predictions from an existing JSONL file.")
    parser.add_argument("--by-type", action="store_true", help="Also print metrics grouped by gold type.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.uint8
    from lettucedetect.models.inference import HallucinationDetector

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    rows = load_rows(args.input, args.hf_dataset, args.hf_split, hf_token)

    if args.row_split != "all":
        rows = [row for row in rows if row.get("split") == args.row_split]

    samples = [row_to_lettuce_example(row) for row in rows]
    if not samples:
        raise ValueError("No rows available for evaluation.")

    questions = [sample["query"] for sample in samples]
    contexts = [sample["contexts"] for sample in samples]
    answers = [sample["output"] for sample in samples]

    if args.load_preds:
        with Path(args.load_preds).open(encoding="utf-8") as f:
            all_preds = [json.loads(line)["pred"] for line in f if line.strip()]
        if len(all_preds) != len(samples):
            raise ValueError(f"Prediction count {len(all_preds)} != sample count {len(samples)}")
    else:
        print(f"Loading model: {args.model}")
        detector = HallucinationDetector(method="transformer", model_path=args.model)
        all_preds = predict_batch(detector, questions, contexts, answers, args.batch_size)

        if args.save_preds:
            save_path = Path(args.save_preds)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w", encoding="utf-8") as f:
                for sample, pred in zip(samples, all_preds):
                    f.write(json.dumps({
                        "dialogue_id": sample["dialogue_id"],
                        "query": sample["query"],
                        "context": sample["context"],
                        "tool_contexts": sample["contexts"],
                        "answer": sample["output"],
                        "gold_type": sample["gold_type"],
                        "gold_label": sample["gold_label"],
                        "gold": sample["gold_spans"],
                        "pred": pred,
                    }, ensure_ascii=False) + "\n")
            print(f"Predictions saved -> {save_path}")

    rp, rr, rf, acc, tp, fp, fn, tn = response_metrics(samples, all_preds)
    cp, cr, cf = char_metrics(samples, all_preds)
    sp, sr, sf = span_metrics(samples, all_preds)

    neg_p = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    neg_r = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    neg_f = 2 * neg_p * neg_r / (neg_p + neg_r) if (neg_p + neg_r) > 0 else 0.0
    type_macro_f1 = (rf + neg_f) / 2
    pred_hall_rate = (tp + fp) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    score = (type_macro_f1 + sf) / 2

    print(f"\n{'=' * 60}")
    print(f"OVERALL ({len(samples)} samples)")
    print(f"{'-' * 60}")
    print(f"type_macro_f1    {type_macro_f1:.4f}")
    print(f"span_f1          {sf:.4f}")
    print(f"score            {score:.4f}")
    print(f"type_accuracy    {acc:.4f}")
    print(f"binary_f1        {rf:.4f}")
    print(f"binary_precision {rp:.4f}")
    print(f"binary_recall    {rr:.4f}")
    print(f"char_f1          {cf:.4f}")
    print(f"pred_hall_rate   {pred_hall_rate:.4f}")
    print(f"{'-' * 60}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Char-level       P={cp:.4f}  R={cr:.4f}  F1={cf:.4f}")
    print(f"Span-level       P={sp:.4f}  R={sr:.4f}  F1={sf:.4f}")
    print(f"{'=' * 60}")

    print_type_classification_report(samples, all_preds)

    if args.by_type:
        by_type = defaultdict(list)
        for sample, pred in zip(samples, all_preds):
            by_type[sample["gold_type"] or "unknown"].append((sample, pred))

        print(f"\n{'-' * 90}")
        print(f"{'task_type':<25} {'n':>5} {'tmacro':>8} {'span':>8} {'score':>8} {'acc':>8} {'bin_f1':>8} {'char':>8}")
        print(f"{'-' * 90}")
        for task_type, pairs in sorted(by_type.items()):
            task_samples, task_preds = zip(*pairs)
            _, _, rf2, acc2, tp2, fp2, fn2, tn2 = response_metrics(task_samples, task_preds)
            _, _, cf2 = char_metrics(task_samples, task_preds)
            _, _, sf2 = span_metrics(task_samples, task_preds)
            neg_p2 = tn2 / (tn2 + fn2) if (tn2 + fn2) > 0 else 0.0
            neg_r2 = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0.0
            neg_f2 = 2 * neg_p2 * neg_r2 / (neg_p2 + neg_r2) if (neg_p2 + neg_r2) > 0 else 0.0
            tmacro2 = (rf2 + neg_f2) / 2
            score2 = (tmacro2 + sf2) / 2
            print(f"{task_type:<25} {len(task_samples):>5} {tmacro2:>8.4f} {sf2:>8.4f} {score2:>8.4f} {acc2:>8.4f} {rf2:>8.4f} {cf2:>8.4f}")


if __name__ == "__main__":
    main()
