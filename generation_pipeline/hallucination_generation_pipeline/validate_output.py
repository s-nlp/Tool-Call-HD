"""
Format validation for unified ToolHACE datasets (CI gate).

Checks every row for (stable issue codes in caps — do not rename):
  MISSING_FIELD          required field absent
  BAD_TYPE               type not in the 5-class taxonomy
  LABEL_TYPE_MISMATCH    label=0 must mean type=clean and vice versa
  NO_FINAL_ANSWER        no final_answer (or assistant) turn in conversations
  SPAN_OOR               span offsets out of range / inverted
  SPAN_INVARIANT         answer[start:end] != span.text
  EMPTY_SPANS_REQUIRED   clean / undergeneration row carries spans
  SPANS_REQUIRED         answer_mismatch / overgeneration / missing_tool
                         row has no spans

Exit code 1 on any failure — safe to wire into CI.

Usage:
    python3 validate_output.py output/final_dataset/RUN
    python3 validate_output.py output/final_dataset/RUN.jsonl --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

VALID_TYPES = {"clean", "answer_mismatch", "missing_tool",
               "overgeneration", "undergeneration"}
SPAN_TYPES = {"answer_mismatch", "missing_tool", "overgeneration"}
EMPTY_SPAN_TYPES = {"clean", "undergeneration"}
REQUIRED_FIELDS = ["dialogue_id", "type", "label", "system",
                   "conversations", "span_labels"]


def load_rows(path: str) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        from datasets import load_from_disk
        ds = load_from_disk(str(p))
        if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
            splits = list(ds.keys())
            if len(splits) != 1:
                sys.exit(f"DatasetDict with splits {splits}; expected one split.")
            ds = ds[splits[0]]
        return [dict(r) for r in ds]
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def final_answer_of(conversations) -> str | None:
    if not isinstance(conversations, list):
        return None
    for t in reversed(conversations):
        if isinstance(t, dict) and t.get("turn_role") == "final_answer":
            return t.get("value", "")
    for t in reversed(conversations):
        if isinstance(t, dict) and t.get("from") == "assistant":
            return t.get("value", "")
    return None


def check_row(row: dict) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []

    missing = [k for k in REQUIRED_FIELDS if k not in row]
    if missing:
        errors.append(("MISSING_FIELD", f"{missing}"))
        return errors  # can't check further reliably

    typ, label = row["type"], row["label"]
    if typ not in VALID_TYPES:
        errors.append(("BAD_TYPE", f"type={typ!r}"))
    if (label == 0) != (typ == "clean"):
        errors.append(("LABEL_TYPE_MISMATCH", f"label={label} type={typ}"))

    answer = final_answer_of(row["conversations"])
    if answer is None:
        errors.append(("NO_FINAL_ANSWER", "no final_answer/assistant turn"))
        return errors

    spans = row["span_labels"] or []
    if typ in EMPTY_SPAN_TYPES and spans:
        errors.append(("EMPTY_SPANS_REQUIRED", f"{typ} row has {len(spans)} spans"))
    if typ in SPAN_TYPES and not spans:
        errors.append(("SPANS_REQUIRED", f"{typ} row has no spans"))

    for i, s in enumerate(spans):
        st, en, txt = s.get("start"), s.get("end"), s.get("text", "")
        if not isinstance(st, int) or not isinstance(en, int) \
                or st < 0 or en > len(answer) or st > en:
            errors.append(("SPAN_OOR", f"span {i}: [{st}:{en}] len={len(answer)}"))
            continue
        if answer[st:en] != txt:
            errors.append(("SPAN_INVARIANT",
                           f"span {i}: answer[{st}:{en}]={answer[st:en]!r:.60} "
                           f"!= text={txt!r:.60}"))
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="Unified HF dataset dir or JSONL file")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    rows = load_rows(args.path)
    n_fail = 0
    code_counts: Counter = Counter()
    type_counts: Counter = Counter()

    for idx, row in enumerate(rows):
        type_counts[row.get("type", "?")] += 1
        errors = check_row(row)
        if errors:
            n_fail += 1
            for code, detail in errors:
                code_counts[code] += 1
                if args.verbose:
                    print(f"  row {idx} ({row.get('dialogue_id', '?')}/"
                          f"{row.get('type', '?')}): {code} — {detail}")

    print(f"\nValidated {len(rows)} rows from {args.path}")
    print(f"  types : {dict(type_counts)}")
    if n_fail:
        print(f"  FAILED rows : {n_fail}")
        print(f"  issue codes : {dict(code_counts)}")
        sys.exit(1)
    print("  all rows OK ✔")


if __name__ == "__main__":
    main()
