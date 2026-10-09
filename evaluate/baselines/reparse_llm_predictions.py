#!/usr/bin/env python3
"""Re-parse the raw responses of prompted-LLM prediction files.

Older outputs of ``evaluate/zero_shot.py`` / ``evaluate/few_shot.py`` stored spans
after a broken repair step: when a model's character offsets did not reproduce its
quoted text, the quote was overwritten with whatever sat at those offsets, and
spans with out-of-range offsets were dropped. LLM offsets are almost always off,
so the stored spans are mostly misplaced (few-shot span F1 ~11 instead of ~58 on
the toolHACE test). Every file keeps ``raw_response``, so nothing needs re-running:
this script re-parses it and keeps exactly what the model emitted — its type and
its span list (start / end / text). ``compute_metrics_decoder.py`` then anchors
each span on the quoted text.

    python evaluate/baselines/reparse_llm_predictions.py results/*_test.jsonl \
        --out-dir results/reparsed [--gold-parquet test.parquet]

``--gold-parquet`` fills gold fields for rows written without them (e.g. API
errors); those rows still count as "no prediction" when scored.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DECODER = json.JSONDecoder()
FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def parse_raw(raw: str | None) -> tuple[dict | None, str]:
    """Return (verdict object, note). The first decodable top-level JSON value wins."""
    if not raw or not raw.strip():
        return None, "empty"
    text = FENCE.sub("", raw.strip())
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None, "no_json"
    try:
        obj, end = DECODER.raw_decode(text[min(starts):])
    except json.JSONDecodeError:
        return None, "invalid_or_truncated_json"
    note = "ok" if not text[min(starts) + end:].strip() else "ok_trailing_text"
    if isinstance(obj, list):
        verdicts = [o for o in obj if isinstance(o, dict) and "type" in o]
        return (verdicts[0], "ok_list_wrapped") if len(verdicts) == 1 else (None, "list_not_a_verdict")
    return (obj, note) if isinstance(obj, dict) else (None, "not_an_object")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", type=Path, help="prediction JSONL files with a raw_response field")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--gold-parquet", type=Path, default=None, help="test parquet, to fill rows written without gold")
    args = ap.parse_args()

    test = None
    if args.gold_parquet:
        import pyarrow.parquet as pq
        test = pq.read_table(args.gold_parquet).to_pylist()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        notes, changed = Counter(), Counter()
        with path.open(encoding="utf-8") as src, (args.out_dir / path.name).open("w", encoding="utf-8") as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                if ("gold_spans" not in row or "final_answer" not in row) and test is not None:
                    gold = test[row["row_index"]]
                    if gold.get("dialogue_id") != row.get("dialogue_id"):
                        raise ValueError(f"row_index {row['row_index']} does not match dialogue_id {row.get('dialogue_id')}")
                    row.setdefault("gold_spans", [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in gold["span_labels"]])
                    row.setdefault("final_answer", gold["answer"])
                    row.setdefault("gold_type", gold["type"])
                    changed["gold_filled_from_parquet"] += 1
                api_error = str(row.get("status", "ok")).startswith("error")
                obj, note = (None, "api_error") if api_error else parse_raw(row.get("raw_response"))
                notes[note] += 1
                pred_type = obj.get("type") if obj else None
                spans = (obj.get("span_labels", obj.get("spans")) if obj else None) or []
                spans = [s for s in spans if isinstance(s, dict)] if isinstance(spans, list) else []
                stored = [(s.get("start"), s.get("end"), s.get("text")) for s in row.get("pred_span_labels") or []]
                changed["spans_differ_from_stored"] += [(s.get("start"), s.get("end"), s.get("text")) for s in spans] != stored
                status = ("ok" if obj is not None and pred_type is not None
                          else "error" if api_error else f"parse_error: {note if obj is None else 'no_type_field'}")
                row.update({"pred_type": pred_type, "pred_span_labels": spans, "status": status,
                            "pipeline_status": row.get("status"), "pipeline_pred_type": row.get("pred_type"),
                            "reparse_note": note})
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path.name}: parse={dict(notes)} {dict(changed)} -> {args.out_dir / path.name}")


if __name__ == "__main__":
    main()
