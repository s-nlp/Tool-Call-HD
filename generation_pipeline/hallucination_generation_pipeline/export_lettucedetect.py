"""
Convert unified ToolHACE rows -> LettuceDetect JSON format.

Emits the SAME sample schema as the editing pipeline's
make_lettucedetect_data.py, so the outputs of both pipelines can be
concatenated for training:

  prompt    : question + flattened tool output(s)
  answer    : final assistant answer (span offsets index into this)
  labels    : [{start, end, label:"hallucination"}]  ([] for clean AND for
              undergeneration — omissions have no in-text location)
  split     : train | dev | test
  task_type : unified type (clean/answer_mismatch/overgeneration/
              missing_tool/undergeneration)
  dataset   : "tool_calling_hallucination"
  language  : "en"

IMPORTANT — split assignment is by dialogue_id, not by row. One source
dialogue yields up to five class-rows sharing the same answer scaffolding;
a per-row random split would leak near-duplicates across train/test.

Usage:
    python3 export_lettucedetect.py --input output/final_dataset/RUN \
        --out-dir output/lettucedetect_data
    python3 export_lettucedetect.py --input output/final_dataset/RUN.jsonl \
        --out-dir output/lettucedetect_data --dev-ratio 0.1 --test-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk


# ── minimal copies of toolhace_lettuce_utils helpers (kept dependency-free) ──

def flatten_json_to_text(json_str: str) -> str:
    try:
        obj = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return str(json_str)
    lines: list[str] = []

    def walk(node, depth=0):
        pad = "  " * depth
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}{k}:")
                    walk(v, depth + 1)
                else:
                    lines.append(f"{pad}{k}: {v}")
            return
        if isinstance(node, list):
            for item in node:
                walk(item, depth)
            return
        lines.append(f"{pad}{node}")

    walk(obj)
    return "\n".join(lines)


def extract_target_turn(conversations: list[dict]) -> dict | None:
    """Final answer + the question and tool outputs immediately preceding it."""
    ans_idx = None
    for i in range(len(conversations) - 1, -1, -1):
        t = conversations[i]
        if t.get("turn_role") == "final_answer" or t.get("from") == "assistant":
            ans_idx = i
            if t.get("turn_role") == "final_answer":
                break
            # keep scanning in case an explicit final_answer exists earlier
    if ans_idx is None:
        return None
    answer = str(conversations[ans_idx].get("value", ""))
    question, tools = "", []
    for i in range(ans_idx - 1, -1, -1):
        role = conversations[i].get("from")
        if role == "user":
            question = str(conversations[i].get("value", ""))
            break
        if role == "tool":
            tools.append(str(conversations[i].get("value", "")))
    tools.reverse()
    return {"question": question, "tool_values": tools, "answer": answer}


# ── conversion ────────────────────────────────────────────────────────────────

def load_rows(path: str) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        ds = load_from_disk(str(p))
        if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
            splits = list(ds.keys())
            if len(splits) != 1:
                sys.exit(f"DatasetDict with splits {splits}; expected one split.")
            ds = ds[splits[0]]
        return [dict(r) for r in ds]
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def convert_row(row: dict) -> dict | None:
    target = extract_target_turn(row.get("conversations") or [])
    if target is None or not target["answer"]:
        return None
    context = "\n\n".join(flatten_json_to_text(t) for t in target["tool_values"])
    prompt = f"{target['question']}\n\n{context}".strip()

    labels = []
    for lab in row.get("span_labels") or []:
        s, e = lab.get("start", -1), lab.get("end", -1)
        if not isinstance(s, int) or not isinstance(e, int):
            continue
        if s < 0 or e <= s or e > len(target["answer"]):
            continue
        labels.append({"start": s, "end": e, "label": "hallucination"})

    return {
        "prompt": prompt,
        "answer": target["answer"],
        "labels": labels,
        "split": "train",  # reassigned below
        "task_type": row.get("type", ""),
        "dataset": "tool_calling_hallucination",
        "language": "en",
        "_dialogue_id": str(row.get("dialogue_id", "")),
    }


def assign_splits_by_dialogue(samples: list[dict], dev_ratio: float,
                              test_ratio: float, seed: int = 42) -> None:
    by_dlg = defaultdict(list)
    for s in samples:
        by_dlg[s["_dialogue_id"]].append(s)
    dlg_ids = sorted(by_dlg)
    random.Random(seed).shuffle(dlg_ids)
    n = len(dlg_ids)
    n_test = max(1, int(n * test_ratio))
    n_dev = max(1, int(n * dev_ratio))
    split_of = {}
    for i, d in enumerate(dlg_ids):
        split_of[d] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
    for s in samples:
        s["split"] = split_of[s["_dialogue_id"]]
        del s["_dialogue_id"]


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Unified HF dir or JSONL")
    p.add_argument("--out-dir", default="output/lettucedetect_data")
    p.add_argument("--prefix", default="toolhace_generation",
                   help="Output filename prefix")
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    rows = load_rows(args.input)
    samples = [s for s in (convert_row(r) for r in rows) if s is not None]
    if not samples:
        sys.exit("No convertible rows found.")

    assign_splits_by_dialogue(samples, args.dev_ratio, args.test_ratio, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_split = defaultdict(list)
    for s in samples:
        by_split[s["split"]].append(s)

    for split in ("train", "dev", "test"):
        path = out_dir / f"{args.prefix}_{split}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(by_split.get(split, []), f, ensure_ascii=False, indent=1)
    with open(out_dir / f"{args.prefix}_all.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)

    types = Counter(s["task_type"] for s in samples)
    print(f"\nExported {len(samples)} samples -> {out_dir}/")
    print(f"  splits (by dialogue): "
          f"{ {k: len(v) for k, v in sorted(by_split.items())} }")
    print(f"  types : {dict(types)}")
    print(f"  hallucinated rows with spans: "
          f"{sum(1 for s in samples if s['labels'])}")


if __name__ == "__main__":
    main()
