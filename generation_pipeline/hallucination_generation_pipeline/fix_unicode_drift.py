#!/usr/bin/env python3
"""
fix_unicode_drift.py
====================
Repair span-text unicode / whitespace drift in tool-hallucination datasets.

Invariant enforced per span:   answer[start:end] == text
where `answer` is the last `final_answer` turn of `conversations`
(fallback: last assistant turn).

Auto-repair policy (conservative):
  * Only rewrites `text` when the offset substring is a *position-aligned
    punctuation/whitespace variant* of the stored text -- i.e. same length,
    every alphanumeric char identical in place, and only non-alphanumeric
    chars differ (U+2011 -> '-', U+2019 -> "'", U+202F/U+00A0 -> ' ', curly
    quotes, etc.). In that case the offsets are provably stable, so we can
    safely overwrite `text` with the exact substring.
  * Anything else (length mismatch, any alnum difference, out-of-range
    offsets, missing offsets, no answer turn) is FLAGGED, never rewritten,
    so genuine offset bugs are surfaced rather than masked.

Conversations / span_labels may be native Python objects (HF datasets) or
stringified Python literals (CSV) -- both are handled.

Usage
-----
  # CSV
  python fix_unicode_drift.py data.csv --out data.fixed.csv --report drift.jsonl
  # HuggingFace dataset saved with save_to_disk (Dataset or DatasetDict)
  python fix_unicode_drift.py ./hf_dataset_dir --out ./hf_fixed --report drift.jsonl
  # detect only, write nothing
  python fix_unicode_drift.py data.csv --check
"""
from __future__ import annotations
import argparse, ast, json, os, sys
from collections import Counter

# ---- stable issue codes (mirror validate_dataset.py style) -------------------
OK              = "SPAN_OK"
FIXED           = "SPAN_FIXED_PUNCT_DRIFT"
GENUINE         = "SPAN_GENUINE_OFFSET_ERROR"
NO_ANSWER       = "SPAN_NO_ANSWER_TURN"
MISSING_OFFSETS = "SPAN_MISSING_OFFSETS"
OOR             = "SPAN_OFFSET_OUT_OF_RANGE"


# ---- parsing helpers ---------------------------------------------------------
def parse_maybe_literal(x):
    """HF -> native object; CSV -> stringified python literal."""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return None
    return x


def resolve_answer(conversations):
    """Offsets index into the last final_answer turn; fall back to last assistant."""
    conv = parse_maybe_literal(conversations)
    if not isinstance(conv, list):
        return None
    finals = [t.get("value") for t in conv
              if isinstance(t, dict) and t.get("turn_role") == "final_answer"]
    if finals:
        return finals[-1]
    assts = [t.get("value") for t in conv
             if isinstance(t, dict) and t.get("from") == "assistant"]
    return assts[-1] if assts else None


def is_punct_drift(sub: str, txt: str) -> bool:
    """True iff sub and txt differ ONLY in non-alphanumeric chars, position-aligned."""
    if len(sub) != len(txt):
        return False
    return all(a == b or (not a.isalnum() and not b.isalnum())
               for a, b in zip(sub, txt))


# ---- core (pure) -------------------------------------------------------------
def fix_row(conversations, span_labels):
    """
    Returns (new_span_labels | None, reports)
      new_span_labels is None when nothing changed.
      reports: list of dicts {span_index, code, old_text, offset_substring}
    """
    spans = parse_maybe_literal(span_labels)
    reports = []
    if not isinstance(spans, list) or not spans:
        return None, reports
    answer = resolve_answer(conversations)
    changed = False
    for i, s in enumerate(spans):
        if not isinstance(s, dict):
            continue
        st, en, txt = s.get("start"), s.get("end"), s.get("text")
        if st is None or en is None or not isinstance(txt, str):
            reports.append(dict(span_index=i, code=MISSING_OFFSETS,
                                old_text=txt, offset_substring=None))
            continue
        if answer is None:
            reports.append(dict(span_index=i, code=NO_ANSWER,
                                old_text=txt, offset_substring=None))
            continue
        if st < 0 or en > len(answer) or st > en:
            reports.append(dict(span_index=i, code=OOR,
                                old_text=txt, offset_substring=None))
            continue
        sub = answer[st:en]
        if sub == txt:
            reports.append(dict(span_index=i, code=OK,
                                old_text=txt, offset_substring=sub))
        elif is_punct_drift(sub, txt):
            s["text"] = sub
            changed = True
            reports.append(dict(span_index=i, code=FIXED,
                                old_text=txt, offset_substring=sub))
        else:
            reports.append(dict(span_index=i, code=GENUINE,
                                old_text=txt, offset_substring=sub))
    return (spans if changed else None), reports


# ---- reporting ---------------------------------------------------------------
def emit(report_records, summary, report_path):
    if report_path:
        # only persist non-OK records to keep the file actionable
        with open(report_path, "w", encoding="utf-8") as fh:
            for r in report_records:
                if r["code"] != OK:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n=== span issue summary ===")
    for code in (OK, FIXED, GENUINE, OOR, MISSING_OFFSETS, NO_ANSWER):
        if summary.get(code):
            print(f"  {code:24s} {summary[code]}")
    genuine = summary.get(GENUINE, 0) + summary.get(OOR, 0)
    if genuine:
        print(f"\n  !! {genuine} span(s) need manual review "
              f"(genuine offset errors; see report).")
    if report_path:
        print(f"  report (non-OK rows): {report_path}")


# ---- CSV path ----------------------------------------------------------------
def run_csv(inp, out, report_path, check_only, span_col, conv_col):
    import pandas as pd

    with open(inp, "r", encoding="utf-8") as fh:
        leading_index = fh.readline().startswith(",")
    df = pd.read_csv(inp, index_col=0 if leading_index else None,
                     keep_default_na=False, dtype=str)
    for c in (span_col, conv_col):
        if c not in df.columns:
            sys.exit(f"ERROR: column '{c}' not in CSV (have: {list(df.columns)})")

    summary, records = Counter(), []
    fixed_rows = 0
    for idx, row in df.iterrows():
        new_spans, reps = fix_row(row[conv_col], row[span_col])
        for r in reps:
            summary[r["code"]] += 1
            r2 = dict(split=None, row=idx, **r)
            records.append(r2)
        if new_spans is not None and not check_only:
            df.at[idx, span_col] = repr(new_spans)
            fixed_rows += 1

    if not check_only:
        df.to_csv(out, index=bool(leading_index))
        print(f"wrote {out}  ({fixed_rows} row(s) modified)")
    else:
        print(f"[check] {summary.get(FIXED,0)} span(s) WOULD be fixed across "
              f"{fixed_rows if not check_only else '...'} rows (dry run)")
    emit(records, summary, report_path)


# ---- HuggingFace path --------------------------------------------------------
def run_hf(inp, out, report_path, check_only, span_col, conv_col):
    from datasets import load_from_disk, DatasetDict

    ds = load_from_disk(inp)
    is_dict = isinstance(ds, DatasetDict)
    items = list(ds.items()) if is_dict else [(None, ds)]

    summary, records = Counter(), []
    new_splits = {}
    for name, split in items:
        if span_col not in split.column_names or conv_col not in split.column_names:
            sys.exit(f"ERROR: split '{name}' missing '{span_col}'/'{conv_col}' "
                     f"(have: {split.column_names})")
        convs = split[conv_col]
        spans = split[span_col]
        new_col, n_fixed = [], 0
        for i in range(len(split)):
            new_spans, reps = fix_row(convs[i], spans[i])
            for r in reps:
                summary[r["code"]] += 1
                records.append(dict(split=name, row=i, **r))
            if new_spans is not None:
                new_col.append(new_spans); n_fixed += 1
            else:
                new_col.append(spans[i])
        if not check_only:
            feats = split.features  # preserve schema exactly
            split = split.map(
                lambda _b, idx: {span_col: [new_col[j] for j in idx]},
                with_indices=True, batched=True, features=feats,
                desc=f"fixing {name or 'dataset'}",
            )
        print(f"  split {name or 'dataset'}: {n_fixed} row(s) modified")
        new_splits[name] = split

    if not check_only:
        out_ds = DatasetDict(new_splits) if is_dict else new_splits[None]
        out_ds.save_to_disk(out)
        print(f"wrote {out}")
    emit(records, summary, report_path)


# ---- entry -------------------------------------------------------------------
def detect_format(path):
    if os.path.isdir(path):
        return "hf"
    if path.lower().endswith((".csv", ".tsv")):
        return "csv"
    return "csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="CSV file or HF dataset directory (save_to_disk)")
    ap.add_argument("--out", help="output path (omit with --check)")
    ap.add_argument("--report", help="JSONL of non-OK spans (fixed + flagged)")
    ap.add_argument("--check", action="store_true", help="detect only, write nothing")
    ap.add_argument("--format", choices=["csv", "hf"], help="override auto-detect")
    ap.add_argument("--span-col", default="span_labels")
    ap.add_argument("--conv-col", default="conversations")
    args = ap.parse_args()

    if not args.check and not args.out:
        ap.error("--out is required unless --check is given")

    fmt = args.format or detect_format(args.input)
    runner = run_hf if fmt == "hf" else run_csv
    runner(args.input, args.out, args.report, args.check, args.span_col, args.conv_col)


if __name__ == "__main__":
    main()
