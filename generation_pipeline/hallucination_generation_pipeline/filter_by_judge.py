"""
Filter a unified dataset by LLM-judge verdicts (judge_verify_annotations.py).

Join key: (dialogue_id, type) == judge JSONL's (dialogue_id, stored_type).
dialogue_id alone is NOT unique — the same source dialogue produces one row
per error class — so both keys are required.

Default policy (all overridable):
  pass                    -> keep
  fail                    -> drop            (--keep-fail keeps them)
  uncertain               -> drop            (--on-uncertain keep)
  not judged / judge err  -> keep + warn     (--on-missing drop)

--apply-suggested-spans rescues span-only failures instead of dropping them:
  a row is rescued when the judge says type_is_correct=true,
  hallucination_present matches the stored label, and every
  suggested_span_labels text occurs exactly ONCE verbatim in the final
  answer (offsets are computed by anchoring; ambiguous or absent text
  means the row is dropped as usual). Never applies to undergeneration
  or clean rows.

Usage:
    python3 filter_by_judge.py --dataset output/fixed/RUN \
        --judge output/judge/RUN.jsonl --output output/final_dataset/RUN
    python3 filter_by_judge.py --dataset output/fixed/RUN \
        --judge output/judge/RUN.jsonl --output output/final_dataset/RUN \
        --apply-suggested-spans --jsonl output/final_dataset/RUN.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk


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


def load_judgments(path: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            key = (str(j.get("dialogue_id")), str(j.get("stored_type")))
            out[key] = j  # last write wins (resume runs may re-judge)
    return out


def final_answer_of(conversations: list[dict]) -> str:
    for t in reversed(conversations or []):
        if isinstance(t, dict) and t.get("turn_role") == "final_answer":
            return t.get("value", "")
    for t in reversed(conversations or []):
        if isinstance(t, dict) and t.get("from") == "assistant":
            return t.get("value", "")
    return ""


def anchor_suggested_spans(answer: str, suggested: list[dict]) -> list[dict] | None:
    """Each suggested span text must occur exactly once verbatim. Returns
    anchored [{start,end,text}] or None if any span can't be anchored."""
    if not suggested:
        return None
    spans = []
    for s in suggested:
        txt = s.get("text")
        if not isinstance(txt, str) or not txt:
            return None
        first = answer.find(txt)
        if first == -1 or answer.find(txt, first + 1) != -1:
            return None
        spans.append({"start": first, "end": first + len(txt), "text": txt})
    return spans


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Unified HF dir or JSONL")
    p.add_argument("--judge", required=True, help="Judge verdicts JSONL")
    p.add_argument("--output", required=True, help="Output HF dataset dir")
    p.add_argument("--jsonl", default=None, help="Also write filtered rows to JSONL")
    p.add_argument("--keep-fail", action="store_true",
                   help="Keep rows the judge failed (default: drop)")
    p.add_argument("--on-uncertain", choices=["drop", "keep"], default="drop")
    p.add_argument("--on-missing", choices=["keep", "drop"], default="keep",
                   help="Rows without a usable verdict (default: keep + warn)")
    p.add_argument("--apply-suggested-spans", action="store_true",
                   help="Rescue span-only failures via judge-suggested spans")
    return p.parse_args()


def main():
    args = get_args()
    rows = load_rows(args.dataset)
    judgments = load_judgments(args.judge)

    kept, out_rows = Counter(), []
    outcome = Counter()

    for row in rows:
        key = (str(row.get("dialogue_id")), str(row.get("type")))
        verdict = judgments.get(key)

        if verdict is None or verdict.get("judgment") is None:
            outcome["missing_or_error"] += 1
            if args.on_missing == "keep":
                out_rows.append(row)
                kept[row["type"]] += 1
            continue

        j = verdict["judgment"]
        decision = j.get("row_judgment")

        if decision == "pass":
            outcome["pass"] += 1
            out_rows.append(row)
            kept[row["type"]] += 1
            continue

        if decision == "uncertain":
            outcome["uncertain"] += 1
            if args.on_uncertain == "keep":
                out_rows.append(row)
                kept[row["type"]] += 1
            continue

        # decision == "fail"
        if (
            args.apply_suggested_spans
            and row.get("type") not in ("undergeneration", "clean")
            and j.get("type_is_correct") is True
            and bool(j.get("hallucination_present")) == (int(row.get("label", 0)) == 1)
        ):
            answer = final_answer_of(row.get("conversations", []))
            anchored = anchor_suggested_spans(answer, j.get("suggested_span_labels") or [])
            if anchored:
                row = dict(row)
                row["span_labels"] = anchored
                outcome["fail_rescued_spans"] += 1
                out_rows.append(row)
                kept[row["type"]] += 1
                continue

        outcome["fail"] += 1
        if args.keep_fail:
            out_rows.append(row)
            kept[row["type"]] += 1

    if not out_rows:
        sys.exit("No rows survived filtering — check the judge file / policies.")

    Dataset.from_list(out_rows).save_to_disk(args.output)
    if args.jsonl:
        Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nFiltered {len(rows)} -> {len(out_rows)} rows  ({args.output})")
    print(f"  verdicts : {dict(outcome)}")
    print(f"  kept/type: {dict(kept)}")
    if outcome["missing_or_error"] and args.on_missing == "keep":
        print(f"  WARNING: {outcome['missing_or_error']} rows had no usable verdict "
              f"and were kept — rerun the judge with --resume to cover them.")


if __name__ == "__main__":
    main()
