"""
Convert raw generate_errors.py output into the unified ToolHACE schema.

This is the glue stage that lets the three original scripts stack:

  generate_errors.py output (raw)          unified schema (downstream)
  ------------------------------           ---------------------------
  class: correct|hallucination|...    -->  type: clean|answer_mismatch|...
  sample_id / id                      -->  dialogue_id / generation_id
  system_prompt                       -->  system
  full: [{role, content}]             -->  conversations: [{from, value, turn_role}]
  spans: [{start, end, text}]         -->  span_labels: [{start, end, text}]
  (derived)                           -->  label: 0|1

Turn-role convention (required by fix_unicode_drift.py / the judge /
toolhace_lettuce_utils.py): the LAST assistant turn gets
turn_role="final_answer"; span offsets index into that turn's `value`.

Span invariant enforced per row:  answer[start:end] == span.text
  * If it holds            -> UNIFY_SPAN_OK
  * If it fails but `text` occurs exactly once verbatim in the answer,
    the offsets are safely re-anchored -> UNIFY_SPAN_REANCHORED
  * Anything else          -> UNIFY_SPAN_BROKEN (row dropped by default;
    keep with --on-broken flag). Punctuation/unicode drift that survives
    this stage is handled downstream by fix_unicode_drift.py.

Class -> (type, label) mapping:
  correct         -> clean            label=0   span_labels must be []
  hallucination   -> answer_mismatch  label=1   spans required
  overgeneration  -> overgeneration   label=1   spans required
  missing_tool    -> missing_tool     label=1   spans required
  undergeneration -> undergeneration  label=1   span_labels=[] BY DESIGN
                                                (omissions kept in omitted_items)

Usage:
    python3 to_unified.py --input output/generated/RUN --output output/unified/RUN \
        --subset singlehop_synthetic --report output/reports/RUN_unify.jsonl
    python3 to_unified.py --input output/generated/RUN --output output/unified/RUN \
        --jsonl output/unified/RUN.jsonl --on-broken flag
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk

# ---------------------------------------------------------------------------
# Constants — stable issue codes (CI-friendly, do not rename)
# ---------------------------------------------------------------------------

SPAN_OK         = "UNIFY_SPAN_OK"
SPAN_REANCHORED = "UNIFY_SPAN_REANCHORED"
SPAN_PUNCT_DRIFT = "UNIFY_SPAN_PUNCT_DRIFT"   # kept as-is; fix_unicode_drift repairs
SPAN_BROKEN     = "UNIFY_SPAN_BROKEN"
NO_FINAL_ANSWER = "UNIFY_NO_FINAL_ANSWER"
UNKNOWN_CLASS   = "UNIFY_UNKNOWN_CLASS"
EMPTY_ANSWER    = "UNIFY_EMPTY_ANSWER"

CLASS_TO_TYPE = {
    "correct": "clean",
    "hallucination": "answer_mismatch",
    "overgeneration": "overgeneration",
    "missing_tool": "missing_tool",
    "undergeneration": "undergeneration",
}

# Types whose span_labels are empty by design
EMPTY_SPAN_TYPES = {"clean", "undergeneration"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_json_maybe(value, default):
    """generate_errors.py JSON-encodes nested fields before save_to_disk."""
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return default
    return default


def build_conversations(full: list[dict]) -> list[dict]:
    """[{role, content}] -> [{from, value, turn_role}], marking the last
    assistant turn as final_answer. All turns carry all three keys so the
    Arrow struct schema stays uniform."""
    turns = []
    for t in full:
        if not isinstance(t, dict):
            continue
        role = t.get("role") or t.get("from") or ""
        value = t.get("content") if "content" in t else t.get("value", "")
        turns.append({"from": str(role), "value": str(value or ""), "turn_role": ""})

    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["from"] == "assistant":
            turns[i]["turn_role"] = "final_answer"
            break
    return turns


def final_answer_of(conversations: list[dict]) -> str | None:
    for t in reversed(conversations):
        if t.get("turn_role") == "final_answer":
            return t.get("value", "")
    return None


def is_punct_drift(sub: str, txt: str) -> bool:
    """Same policy as fix_unicode_drift.py: differ ONLY in non-alphanumeric
    chars, position-aligned. Offsets are provably stable in this case."""
    if len(sub) != len(txt):
        return False
    return all(a == b or (not a.isalnum() and not b.isalnum())
               for a, b in zip(sub, txt))


def reconcile_spans(answer: str, spans: list[dict]) -> tuple[list[dict], list[dict]]:
    """Enforce answer[start:end] == text; re-anchor when safe.
    Returns (new_spans | None-on-broken, reports)."""
    reports = []
    out = []
    broken = False
    for i, s in enumerate(spans):
        st, en, txt = s.get("start"), s.get("end"), s.get("text")
        if not isinstance(txt, str) or not isinstance(st, int) or not isinstance(en, int):
            reports.append({"span_index": i, "code": SPAN_BROKEN,
                            "detail": "missing/typed-wrong start|end|text"})
            broken = True
            continue
        if 0 <= st <= en <= len(answer) and answer[st:en] == txt:
            reports.append({"span_index": i, "code": SPAN_OK, "detail": None})
            out.append({"start": st, "end": en, "text": txt})
            continue
        # Position-aligned punctuation/unicode drift: offsets are stable, only
        # non-alnum chars differ. Keep the span untouched — the fix stage
        # (fix_unicode_drift.py) owns the repair + report for these.
        if 0 <= st <= en <= len(answer) and is_punct_drift(answer[st:en], txt):
            reports.append({"span_index": i, "code": SPAN_PUNCT_DRIFT,
                            "detail": "left for fix_unicode_drift.py"})
            out.append({"start": st, "end": en, "text": txt})
            continue
        # Safe re-anchor: text must appear exactly once verbatim
        first = answer.find(txt)
        if txt and first != -1 and answer.find(txt, first + 1) == -1:
            reports.append({"span_index": i, "code": SPAN_REANCHORED,
                            "detail": f"[{st}:{en}] -> [{first}:{first + len(txt)}]"})
            out.append({"start": first, "end": first + len(txt), "text": txt})
            continue
        n_hits = 0 if not txt else answer.count(txt)
        reports.append({"span_index": i, "code": SPAN_BROKEN,
                        "detail": f"offset mismatch; verbatim hits in answer: {n_hits}"})
        broken = True
    return (None if broken else out), reports


def convert_row(row: dict, subset: str) -> tuple[dict | None, list[dict]]:
    """Returns (unified_row | None, reports). None means the row is dropped
    for a structural reason (reports say why)."""
    reports = []
    cls = row.get("class", "")
    if cls not in CLASS_TO_TYPE:
        return None, [{"code": UNKNOWN_CLASS, "detail": f"class={cls!r}"}]

    typ = CLASS_TO_TYPE[cls]
    label = 0 if typ == "clean" else 1

    full = parse_json_maybe(row.get("full"), [])
    conversations = build_conversations(full)
    answer = final_answer_of(conversations)
    if answer is None:
        return None, [{"code": NO_FINAL_ANSWER, "detail": "no assistant turn in full"}]
    if not answer.strip():
        return None, [{"code": EMPTY_ANSWER, "detail": "final answer is empty"}]

    raw_spans = [] if typ in EMPTY_SPAN_TYPES else parse_json_maybe(row.get("spans"), [])
    span_labels, span_reports = reconcile_spans(answer, raw_spans)
    reports.extend(span_reports)

    omitted = parse_json_maybe(row.get("omitted_items"), [])

    unified = {
        "dialogue_id": str(row.get("sample_id") or row.get("id") or ""),
        "generation_id": str(row.get("id") or ""),
        "subset": subset,
        "type": typ,
        "label": label,
        "system": str(row.get("system_prompt") or ""),
        "conversations": conversations,
        "span_labels": span_labels if span_labels is not None else [],
        "tool_call": str(row.get("tool_call") or ""),
        "tool_response": str(row.get("tool_response") or ""),
        "original_tool_response": str(row.get("original_tool_response") or ""),
        "omitted_items": json.dumps(omitted, ensure_ascii=False),
    }
    # Signal broken spans upward
    unified["_spans_broken"] = span_labels is None
    return unified, reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="generate_errors.py output: HF save_to_disk dir")
    p.add_argument("--output", required=True,
                   help="Output HF dataset dir (save_to_disk)")
    p.add_argument("--jsonl", default=None,
                   help="Also write rows to this JSONL path")
    p.add_argument("--subset", default="",
                   help="Subset tag stored per row, e.g. 'singlehop_synthetic'")
    p.add_argument("--report", default=None,
                   help="JSONL report of all non-OK span events + dropped rows")
    p.add_argument("--on-broken", choices=["drop", "flag"], default="drop",
                   help="Rows with unrepairable spans: drop (default) or keep "
                        "with span_labels=[] and _spans_broken=True")
    return p.parse_args()


def main():
    args = get_args()
    ds = load_from_disk(args.input)
    if hasattr(ds, "keys") and not hasattr(ds, "column_names"):  # DatasetDict
        splits = list(ds.keys())
        if len(splits) != 1:
            sys.exit(f"Input is a DatasetDict with splits {splits}; expected one split.")
        ds = ds[splits[0]]

    rows_out, all_reports = [], []
    dropped = Counter()
    kept = Counter()

    for row in ds:
        unified, reports = convert_row(dict(row), args.subset)
        rid = str(row.get("id") or row.get("sample_id") or "?")
        for r in reports:
            if r.get("code") != SPAN_OK:
                all_reports.append({"generation_id": rid, **r})

        if unified is None:
            dropped[reports[0]["code"]] += 1
            continue
        if unified.pop("_spans_broken"):
            if args.on_broken == "drop":
                dropped[SPAN_BROKEN] += 1
                continue
            unified["span_labels"] = []
        kept[unified["type"]] += 1
        rows_out.append(unified)

    if not rows_out:
        sys.exit("No rows survived conversion — check the input dataset.")

    out_ds = Dataset.from_list(rows_out)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_ds.save_to_disk(args.output)

    if args.jsonl:
        Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            for r in all_reports:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_re = sum(1 for r in all_reports if r["code"] == SPAN_REANCHORED)
    print(f"\nUnified {len(rows_out)} rows -> {args.output}")
    print(f"  by type : {dict(kept)}")
    print(f"  re-anchored spans : {n_re}")
    if dropped:
        print(f"  dropped rows      : {dict(dropped)}")
    if args.report:
        print(f"  report            : {args.report}")


if __name__ == "__main__":
    main()
