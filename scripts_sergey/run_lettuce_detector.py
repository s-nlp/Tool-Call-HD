"""
Run a hallucination detector over JSONL datasets and save predictions as JSONL.

Input format (data_v2_ragtruth_format/singlehop/qwen_corrupted/*.jsonl)
-----------------------------------------------------------------------
Each line is a JSON object with:
  query   – user question
  context – tool/retrieval response (string)
  output  – generated answer to evaluate

Output
------
JSONL with all original fields plus a `pred` field: a list of span dicts,
e.g. [{"start": 10, "end": 25, "text": "some hallucinated text"}, ...]
Empty list means no hallucinations detected.
shit
Usage
-----
python scripts/run_detector.py \
    --method lettucedetect \
    --checkpoint KRLabsOrg/lettucedect-large-modernbert-en-v1 \
    --data data_v2_ragtruth_format/singlehop/qwen_corrupted/singlehop_type1.jsonl \
    --output predictions/lettuce_large_type1.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Detector wrappers
# ---------------------------------------------------------------------------

def _keep_span_fields(spans: list[dict]) -> list[dict]:
    return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in spans]


def _spans_from_token_preds(token_preds, probabilities, labels, offsets, answer_start_token, answer,
                            threshold: float = 0.5):
    """Replicate lettucedetect span extraction for a single item in a batch."""
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
        is_hal = probabilities[i, 1].item() >= threshold
        confidence = probabilities[i, 1].item() if is_hal else 0.0
        if is_hal:
            if current_span is None:
                current_span = {"start": rel_start, "end": rel_end, "confidence": confidence}
            else:
                current_span["end"] = rel_end
                current_span["confidence"] = max(current_span["confidence"], confidence)
        else:
            if current_span is not None:
                current_span["text"] = answer[current_span["start"]:current_span["end"]]
                spans.append(current_span)
                current_span = None
    if current_span is not None:
        current_span["text"] = answer[current_span["start"]:current_span["end"]]
        spans.append(current_span)
    return spans


class LettuceDetector:
    def __init__(self, checkpoint: str, batch_size: int = 32):
        from lettucedetect.models.inference import HallucinationDetector
        self._det = HallucinationDetector(method="transformer", model_path=checkpoint)
        self.batch_size = batch_size

    def predict(self, question: str, contexts: list[str], answer: str) -> list[dict]:
        spans = self._det.predict(
            context=contexts,
            question=question,
            answer=answer,
            output_format="spans",
        )
        return _keep_span_fields(spans)

    def predict_batch(self, questions: list[str], contexts: list[list[str]], answers: list[str],
                      threshold: float = 0.5) -> list[list[dict]]:
        from lettucedetect.datasets.hallucination_dataset import HallucinationDataset
        from lettucedetect.detectors.prompt_utils import PromptUtils

        inner = self._det.detector
        tokenizer = inner.tokenizer
        model = inner.model
        device = inner.device
        max_length = inner.max_length

        results = []
        for i in range(0, len(answers), self.batch_size):
            batch_questions = questions[i:i + self.batch_size]
            batch_contexts = contexts[i:i + self.batch_size]
            batch_answers = answers[i:i + self.batch_size]

            encodings, offsets_list, answer_starts = [], [], []
            for q, ctx, ans in zip(batch_questions, batch_contexts, batch_answers):
                prompt = PromptUtils.format_context(ctx, q, inner.lang)
                enc, _, offsets, ans_start = HallucinationDataset.prepare_tokenized_input(
                    tokenizer, prompt, ans, max_length
                )
                encodings.append(enc)
                offsets_list.append(offsets)
                answer_starts.append(ans_start)

            pad_id = tokenizer.pad_token_id or 0
            max_len = max(e["input_ids"].shape[1] for e in encodings)

            def _pad(t, pad_val, length):
                sz = length - t.shape[1]
                return torch.cat([t, torch.full((1, sz), pad_val, dtype=t.dtype)], dim=1)

            input_ids = torch.cat(
                [_pad(e["input_ids"], pad_id, max_len) for e in encodings], dim=0
            ).to(device)
            attention_mask = torch.cat(
                [_pad(e["attention_mask"], 0, max_len) for e in encodings], dim=0
            ).to(device)

            labels_batch = []
            for j, enc in enumerate(encodings):
                seq_len = enc["input_ids"].shape[1]
                labels = torch.full((max_len,), -100, dtype=torch.long)
                labels[answer_starts[j]:seq_len] = 0
                labels_batch.append(labels)
            labels_tensor = torch.stack(labels_batch).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            token_preds = torch.argmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            token_preds = torch.where(labels_tensor == -100, labels_tensor, token_preds)

            for j, ans in enumerate(batch_answers):
                spans = _spans_from_token_preds(
                    token_preds[j], probabilities[j], labels_tensor[j],
                    offsets_list[j], answer_starts[j], ans,
                    threshold=threshold,
                )
                results.append(_keep_span_fields(spans))

        return results


class HalDetector(LettuceDetector):
    pass


class AlwaysHalDetector:
    def __init__(self, checkpoint: str): ...

    def predict(self, question: str, contexts: list[str], answer: str) -> list[dict]:
        return [{"start": 0, "end": len(answer), "text": answer}]


class AlwaysNoHalDetector:
    def __init__(self, checkpoint: str): ...

    def predict(self, question: str, contexts: list[str], answer: str) -> list[dict]:
        return []


DETECTORS = {
    "lettucedetect": LettuceDetector,
    "haldetect": HalDetector,
    "always_hal": AlwaysHalDetector,
    "always_no_hal": AlwaysNoHalDetector,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run hallucination detector on JSONL data.")
    p.add_argument("--method", required=True, choices=list(DETECTORS),
                   help="Detector backend to use.")
    p.add_argument("--checkpoint", default="",
                   help="HuggingFace checkpoint or local model path. "
                        "Not required for always_hal / always_no_hal.")
    p.add_argument("--data", required=True,
                   help="Path to input JSONL file.")
    p.add_argument("--output", default=None,
                   help="Path for output JSONL file. "
                        "Defaults to predictions/<method>_<checkpoint_slug>_<input_stem>.jsonl")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for transformer inference (default: 32).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Token-level hallucination probability threshold (default: 0.5).")
    return p.parse_args()


def default_output_path(method: str, checkpoint: str, data_path: Path) -> Path:
    slug = checkpoint.replace("/", "_").replace("\\", "_")
    return Path("predictions") / f"{method}_{slug}_{data_path.stem}.jsonl"


def main():
    args = parse_args()
    data_path = Path(args.data)

    output_path = Path(args.output) if args.output else default_output_path(args.method, args.checkpoint, data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading detector  : {args.method} / {args.checkpoint}")
    detector = DETECTORS[args.method](args.checkpoint, args.batch_size)

    print(f"Loading data      : {data_path}")
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]

    use_batch = hasattr(detector, "predict_batch") and args.batch_size > 1

    print(f"Saving predictions: {output_path}  (batch_size={args.batch_size if use_batch else 1})")
    with output_path.open("w", encoding="utf-8") as fout:
        if use_batch:
            questions = [r["query"] for r in rows]
            contexts  = [[r["context"]] for r in rows]
            answers   = [r["output"] for r in rows]
            all_spans = []
            for i in tqdm(range(0, len(rows), args.batch_size), desc="Running detector"):
                all_spans.extend(detector.predict_batch(
                    questions[i:i + args.batch_size],
                    contexts[i:i + args.batch_size],
                    answers[i:i + args.batch_size],
                    threshold=args.threshold,
                ))
            for row, spans in zip(rows, all_spans):
                row["pred"] = spans
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            for row in tqdm(rows, desc="Running detector"):
                spans = detector.predict(
                    question=row["query"],
                    contexts=[row["context"]],
                    answer=row["output"],
                )
                row["pred"] = spans
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. {len(rows)} predictions saved.")


if __name__ == "__main__":
    main()
