#!/usr/bin/env python3
"""Join generated ToolHACE verdicts to gold data for compute_metrics.py.

The generation pipeline stores exact predicted span text, while the supplied
response- and span-level metric script expects explicit character offsets plus
gold annotations in one JSONL. Its response decision is derived from whether
the normalized prediction contains at least one span. This adapter preserves
the released first-occurrence convention. A
generated span that cannot be located in the answer receives an out-of-answer
virtual interval so it remains a response-level positive and span-level false
positive instead of disappearing from evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from pipeline_common import final_answer


def load_predictions(path: Path) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"prediction line {line_number} is not an object")
            predictions.append(value)
    return predictions


def gold_spans(example: dict[str, Any], answer: str) -> list[dict[str, Any]]:
    result = []
    for span in example.get("span_labels") or []:
        try:
            start, end = int(span["start"]), int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        result.append(
            {
                "start": start,
                "end": end,
                "text": str(span.get("text") or answer[start:end]),
            }
        )
    return result


def predicted_spans(
    prediction: dict[str, Any], answer: str
) -> tuple[list[dict[str, Any]], int]:
    if not prediction.get("parsed") or not isinstance(prediction.get("errors"), list):
        return [], 0

    result = []
    unlocatable = 0
    virtual_cursor = len(answer) + 1
    for error in prediction["errors"]:
        if not isinstance(error, dict):
            continue
        text = error.get("span")
        if not isinstance(text, str) or not text:
            # The scorer is intentionally span based. In particular, a null
            # undergeneration span cannot create a predicted-positive span.
            continue
        start = answer.find(text)
        if start < 0:
            start = virtual_cursor
            end = start + max(1, len(text))
            virtual_cursor = end + 1
            unlocatable += 1
        else:
            end = start + len(text)
        result.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "class": error.get("class"),
            }
        )
    return result, unlocatable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--setting", default="Transfer SFT")
    args = parser.parse_args()

    predictions = load_predictions(args.pred)
    dataset = load_dataset(
        "parquet", data_files=str(args.test_parquet), split="train"
    )
    seen_indices: set[int] = set()
    normalized = []
    unlocatable_total = 0
    parse_failures = 0

    for position, prediction in enumerate(predictions):
        index = int(prediction.get("idx", position))
        if index < 0 or index >= len(dataset):
            raise IndexError(f"prediction index {index} is outside test data")
        if index in seen_indices:
            raise ValueError(f"duplicate prediction index: {index}")
        seen_indices.add(index)

        example = dataset[index]
        answer = final_answer(example)
        pred_spans, unlocatable = predicted_spans(prediction, answer)
        unlocatable_total += unlocatable
        parsed = bool(prediction.get("parsed"))
        parse_failures += int(not parsed)
        normalized.append(
            {
                "idx": index,
                "dialogue_id": example.get("dialogue_id"),
                "answer": answer,
                "gold_type": example.get("type"),
                "gold_spans": gold_spans(example, answer),
                "pred_spans": pred_spans,
                "status": "ok" if parsed else "parse_error",
                "model": args.model,
                "checkpoint_path": args.checkpoint,
                "dataset": "ToolHACE",
                "data_name": "ToolHACE",
                "split": "test",
                "setting": args.setting,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in normalized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"wrote {len(normalized)} rows to {args.out}; "
        f"parse_failures={parse_failures} unlocatable_predicted_spans={unlocatable_total}",
        flush=True,
    )


if __name__ == "__main__":
    main()
