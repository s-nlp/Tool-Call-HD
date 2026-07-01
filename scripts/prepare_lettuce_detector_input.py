from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from toolhace_lettuce_utils import DEFAULT_HF_DATASET, DEFAULT_HF_SPLIT, load_rows, row_to_lettuce_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ToolHACE unified rows into query/context/output JSONL for LettuceDetect."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Local ToolHACE file (.json, .jsonl, or .parquet). If omitted, load from Hugging Face.",
    )
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        help=f"Hugging Face dataset repo id (default: {DEFAULT_HF_DATASET}).",
    )
    parser.add_argument(
        "--hf-split",
        default=DEFAULT_HF_SPLIT,
        help=f"Hugging Face split to load (default: {DEFAULT_HF_SPLIT}).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. Falls back to HF_TOKEN from the environment.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--start", type=int, default=0, help="Start row index.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to export.")
    parser.add_argument(
        "--include-gold",
        action="store_true",
        help="Also write gold metadata fields (type/label/spans) alongside query/context/output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    rows = load_rows(args.input, args.hf_dataset, args.hf_split, hf_token)

    end = len(rows) if args.limit is None else min(len(rows), args.start + args.limit)
    selected = rows[args.start:end]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for row in selected:
            try:
                item = row_to_lettuce_example(row)
            except Exception:
                skipped += 1
                continue

            payload = {
                "query": item["query"],
                "context": item["context"],
                "output": item["output"],
            }
            if args.include_gold:
                payload.update(
                    {
                        "dialogue_id": item["dialogue_id"],
                        "gold_type": item["gold_type"],
                        "gold_label": item["gold_label"],
                        "gold_spans": item["gold_spans"],
                        "row_split": item["row_split"],
                    }
                )

            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    print(f"Loaded rows    : {len(rows)}")
    print(f"Selected rows  : {len(selected)}")
    print(f"Written rows   : {written}")
    print(f"Skipped rows   : {skipped}")
    print(f"Saved to       : {output_path}")


if __name__ == "__main__":
    main()
