from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_HF_DATASET = "s-nlp/toolace-unified-hallucinations"
DEFAULT_HF_SPLIT = "test"


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _to_python(value.tolist())
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    return value


def load_rows_from_hf(repo_id: str, split: str, token: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(repo_id, split=split, token=token)
    rows = []
    for row in ds:
        row = dict(row)
        row["conversations"] = _to_python(row.get("conversations", []))
        row["span_labels"] = _to_python(row.get("span_labels", []))
        rows.append(row)
    return rows


def load_rows_from_parquet(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    rows = []
    for row in df.to_dict(orient="records"):
        row["conversations"] = _to_python(row.get("conversations", []))
        row["span_labels"] = _to_python(row.get("span_labels", []))
        rows.append(row)
    return rows


def load_rows_from_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = [rows]

    normalized = []
    for row in rows:
        row = dict(row)
        row["conversations"] = _to_python(row.get("conversations", []))
        row["span_labels"] = _to_python(row.get("span_labels", []))
        normalized.append(row)
    return normalized


def load_rows(
    input_path: str | None,
    hf_dataset: str = DEFAULT_HF_DATASET,
    hf_split: str = DEFAULT_HF_SPLIT,
    hf_token: str | None = None,
) -> list[dict[str, Any]]:
    if input_path:
        path = Path(input_path)
        if path.suffix == ".parquet":
            return load_rows_from_parquet(path)
        return load_rows_from_json(path)
    return load_rows_from_hf(hf_dataset, hf_split, hf_token)


def flatten_json_to_text(json_str: str) -> str:
    try:
        obj = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return str(json_str)

    lines: list[str] = []

    def walk(node: Any, depth: int = 0) -> None:
        pad = "  " * depth
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}{key}:")
                    walk(value, depth + 1)
                else:
                    lines.append(f"{pad}{key}: {value}")
            return

        if isinstance(node, list):
            for item in node:
                walk(item, depth)
            return

        lines.append(f"{pad}{node}")

    walk(obj)
    return "\n".join(lines)


def normalize_span_labels(labels: Any) -> list[dict[str, Any]]:
    if not isinstance(labels, list):
        return []

    normalized = []
    for label in labels:
        if not isinstance(label, dict):
            continue
        start = label.get("start")
        end = label.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            continue
        normalized.append(
            {
                "start": start,
                "end": end,
                "text": str(label.get("text", "")),
            }
        )
    return normalized


def extract_target_turn(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    assistant_idx = None
    for idx in range(len(conversations) - 1, -1, -1):
        if conversations[idx].get("from") == "assistant":
            assistant_idx = idx
            break

    if assistant_idx is None:
        raise ValueError("No assistant answer found in conversations")

    answer = str(conversations[assistant_idx].get("value", ""))
    question = ""
    tool_values: list[str] = []

    for idx in range(assistant_idx - 1, -1, -1):
        role = conversations[idx].get("from")
        if role == "user":
            question = str(conversations[idx].get("value", ""))
            break
        if role == "tool":
            tool_values.append(str(conversations[idx].get("value", "")))

    tool_values.reverse()
    return {
        "question": question,
        "tool_values": tool_values,
        "answer": answer,
    }


def row_to_lettuce_example(row: dict[str, Any]) -> dict[str, Any]:
    turn = extract_target_turn(row.get("conversations", []))
    contexts = [flatten_json_to_text(value) for value in turn["tool_values"]]
    return {
        "dialogue_id": row.get("dialogue_id"),
        "query": turn["question"],
        "contexts": contexts,
        "context": "\n\n".join(contexts),
        "output": turn["answer"],
        "gold_type": row.get("type"),
        "gold_label": row.get("label"),
        "gold_spans": normalize_span_labels(row.get("span_labels", [])),
        "row_split": row.get("split"),
    }
