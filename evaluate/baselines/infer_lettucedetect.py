#!/usr/bin/env python3
"""Run a LettuceDetect-compatible checkpoint on the ToolHACE test split.

The test split is loaded from Hugging Face by default.  Every output JSONL
record contains the model input, gold annotations, prediction, and status, so
``compute_metrics.py`` can score the file without downloading the dataset
again.

Examples
--------
    python evaluate/baselines/infer_lettucedetect.py \
        --checkpoint_path KRLabsOrg/lettucedect-large-modernbert-en-v1 \
        --output evaluate/baselines/results/lettucedect_large_test.jsonl

    python evaluate/baselines/infer_lettucedetect.py \
        --checkpoint_path /path/to/local/checkpoint \
        --max-rows 16

The local parquet attached during development is intentionally not used by
the normal command: the benchmark is fetched with ``datasets.load_dataset``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolhace_lettuce_utils import (  # noqa: E402
    DEFAULT_HF_SPLIT,
    flatten_json_to_text,
    normalize_span_labels,
    row_to_lettuce_example,
)


DEFAULT_HF_DATASET = "s-nlp/toolHACE"
DEFAULT_CHECKPOINT = "KRLabsOrg/lettucedect-large-modernbert-en-v1"


def _json_value(value: Any) -> Any:
    """Decode JSON stored as a string and recursively convert array values."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return _json_value(json.loads(stripped))
            except json.JSONDecodeError:
                return value
        return value
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _normalise_spans(value: Any) -> list[dict[str, Any]]:
    value = _json_value(value)
    if not isinstance(value, list):
        return []

    # The shared helper handles the benchmark's normal list-of-dicts format.
    normalised = normalize_span_labels(value)
    if normalised:
        return normalised

    # Keep valid spans even when a source row uses numpy integer scalars or a
    # slightly different field name.  This also makes the saved gold robust
    # to parquet -> Python conversions.
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        start, end = item.get("start"), item.get("end")
        if not isinstance(start, Integral) or not isinstance(end, Integral):
            continue
        start, end = int(start), int(end)
        if end <= start:
            continue
        result.append(
            {
                "start": start,
                "end": end,
                "text": str(item.get("text", "")),
                **({"label_type": item["label_type"]} if "label_type" in item else {}),
            }
        )
    return result


def _context_to_text(value: Any) -> str:
    value = _json_value(value)
    if isinstance(value, str):
        return flatten_json_to_text(value)
    if isinstance(value, (dict, list)):
        return flatten_json_to_text(json.dumps(value, ensure_ascii=False))
    return str(value)


def _first(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _row_to_example(row: dict[str, Any]) -> dict[str, Any]:
    """Convert both the unified conversation schema and flat test schemas."""
    row = {key: _json_value(value) for key, value in row.items()}

    if isinstance(row.get("conversations"), list):
        example = row_to_lettuce_example(row)
        example["gold_spans"] = _normalise_spans(row.get("span_labels", []))
        return example

    question = str(_first(row, "query", "question", "prompt", "user_prompt", default=""))
    answer = str(_first(row, "output", "answer", "final_answer", "response", default=""))
    raw_context = _first(row, "contexts", "context", "tool_contexts", "tool_response", default=[])
    raw_context = _json_value(raw_context)
    if isinstance(raw_context, list):
        contexts = [_context_to_text(item) for item in raw_context]
    else:
        contexts = [_context_to_text(raw_context)] if raw_context not in (None, "") else []

    return {
        "dialogue_id": _first(row, "dialogue_id", "id", "example_id", default=None),
        "query": question,
        "contexts": contexts,
        "context": "\n\n".join(contexts),
        "output": answer,
        "gold_type": _first(row, "gold_type", "type", "task_type", "class", default=None),
        "gold_label": _first(row, "gold_label", "label", default=None),
        "gold_spans": _normalise_spans(
            _first(row, "gold_spans", "span_labels", "gold", default=[])
        ),
        "row_split": _first(row, "split", "row_split", default=None),
    }


def _import_load_dataset():
    """Import the HF package without the repository's ``datasets/`` folder."""
    try:
        from datasets import load_dataset

        if callable(load_dataset):
            return load_dataset
    except ImportError:
        pass

    # The repository has a data directory called datasets/.  When the script
    # is launched from the repo, it can shadow the third-party HF package.
    original_path = list(sys.path)
    original_module = sys.modules.pop("datasets", None)
    try:
        sys.path[:] = [
            entry
            for entry in sys.path
            if entry not in ("", str(PROJECT_ROOT))
            and Path(entry or ".").resolve() != PROJECT_ROOT
        ]
        from datasets import load_dataset

        return load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required to download ToolHACE. "
            "Install it with `pip install datasets`."
        ) from exc
    finally:
        sys.path[:] = original_path
        if original_module is not None:
            sys.modules["datasets"] = original_module
        else:
            sys.modules.pop("datasets", None)


def load_test_rows(dataset_name: str, split: str, token: str | None, cache_dir: str | None) -> list[dict[str, Any]]:
    load_dataset = _import_load_dataset()
    kwargs: dict[str, Any] = {"split": split}
    if token:
        kwargs["token"] = token
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    dataset = load_dataset(dataset_name, **kwargs)
    return [dict(row) for row in dataset]


def _normalise_prediction_span(span: Any, answer: str) -> dict[str, Any] | None:
    if not isinstance(span, dict):
        return None
    start, end = span.get("start"), span.get("end")
    if not isinstance(start, Integral) or not isinstance(end, Integral):
        return None
    start, end = int(start), int(end)
    if start < 0 or end <= start or end > len(answer):
        return None
    text = span.get("text")
    if not text:
        text = answer[start:end]
    result: dict[str, Any] = {
        "start": start,
        "end": end,
        "text": str(text),
    }
    confidence = span.get("confidence", span.get("score"))
    if isinstance(confidence, Real):
        result["confidence"] = float(confidence)
    category = span.get("category", span.get("label_type"))
    if category is not None:
        result["label_type"] = str(category)
    return result


def _normalise_predictions(predictions: Any, answer: str) -> list[dict[str, Any]]:
    if isinstance(predictions, dict):
        predictions = predictions.get("spans", predictions.get("predictions", []))
    if not isinstance(predictions, Iterable) or isinstance(predictions, (str, bytes)):
        return []
    result = []
    for span in predictions:
        normalised = _normalise_prediction_span(span, answer)
        if normalised is not None:
            result.append(normalised)
    return result


def _checkpoint_name(checkpoint_path: str) -> str:
    path = Path(checkpoint_path)
    return path.name if path.name else checkpoint_path.rstrip("/").rsplit("/", 1)[-1]


def run_inference(args: argparse.Namespace) -> Path:
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        # Make an explicitly supplied token visible to the model loader too.
        if args.hf_token:
            os.environ["HF_TOKEN"] = token
    rows = load_test_rows(args.dataset, args.split, token, args.cache_dir)
    if args.start:
        rows = rows[args.start:]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No rows available in the selected ToolHACE split.")

    print(f"Loaded {len(rows)} rows from {args.dataset}/{args.split}")
    print(f"Loading checkpoint: {args.checkpoint_path}")

    import torch

    # Compatibility with older lettucedetect/transformers combinations.
    if not hasattr(torch, "float8_e8m0fnu"):
        torch.float8_e8m0fnu = torch.uint8
    from lettucedetect.models.inference import HallucinationDetector

    detector = HallucinationDetector(method="transformer", model_path=args.checkpoint_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(enumerate(rows), total=len(rows), desc="Inference", unit="row")
    except ImportError:
        iterator = enumerate(rows)

    error_count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for row_index, row in iterator:
            example = _row_to_example(row)
            status = "ok"
            error = None
            pred_spans: list[dict[str, Any]] = []
            try:
                predictions = detector.predict(
                    context=example["contexts"],
                    question=example["query"],
                    answer=example["output"],
                    output_format="spans",
                )
                pred_spans = _normalise_predictions(predictions, example["output"])
            except Exception as exc:  # Preserve the row in the metric denominator.
                status = "error"
                error_count += 1
                error = f"{type(exc).__name__}: {exc}"

            record = {
                "schema_version": 1,
                "model": args.model_name or args.checkpoint_path,
                "checkpoint_path": args.checkpoint_path,
                "dataset": args.dataset,
                "split": args.split,
                "setting": args.setting,
                "data_name": args.data_name,
                "row_index": args.start + row_index,
                "dialogue_id": example["dialogue_id"],
                "query": example["query"],
                "contexts": example["contexts"],
                "context": example["context"],
                "answer": example["output"],
                "gold_type": example["gold_type"],
                "gold_label": example["gold_label"],
                "gold_spans": example["gold_spans"],
                "pred_spans": pred_spans,
                "n_pred_spans": len(pred_spans),
                "status": status,
            }
            if error is not None:
                record["error"] = error
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {output_path}")
    if error_count:
        print(f"Rows with inference errors: {error_count} (kept in output as status=error)")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a LettuceDetect-like checkpoint on the s-nlp/toolHACE test split."
    )
    parser.add_argument(
        "checkpoint_path_positional",
        nargs="?",
        help="Optional positional form of --checkpoint_path.",
    )
    parser.add_argument(
        "--checkpoint_path",
        "--checkpoint-path",
        "--model",
        dest="checkpoint_path",
        default=None,
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument("--output", default=None, help="Output JSONL path.")
    parser.add_argument("--dataset", default=DEFAULT_HF_DATASET, help="HF dataset repo id.")
    parser.add_argument("--split", default=DEFAULT_HF_SPLIT, help="HF split to evaluate.")
    parser.add_argument("--hf-token", default=None, help="HF token; defaults to HF_TOKEN.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--start", type=int, default=0, help="Start row in the selected split.")
    parser.add_argument("--max-rows", type=int, default=None, help="Evaluate only the first N selected rows.")
    parser.add_argument("--model-name", default=None, help="Display name written to each output record.")
    parser.add_argument("--setting", default="Transfer SFT", help="Table metadata: Setting column.")
    parser.add_argument("--data-name", default="ToolHACE", help="Table metadata: Data column.")
    args = parser.parse_args()
    args.checkpoint_path = args.checkpoint_path or args.checkpoint_path_positional or DEFAULT_CHECKPOINT
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    if args.output is None:
        args.output = str(
            Path("evaluate/baselines/results") / f"{_checkpoint_name(args.checkpoint_path)}_{args.split}.jsonl"
        )
    return args


if __name__ == "__main__":
    run_inference(parse_args())
