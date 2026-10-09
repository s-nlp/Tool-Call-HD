#!/usr/bin/env python3
"""Chance level of the response-level F1 columns, and each model's lift over it.

A detector that ignores its input and flags each answer with probability q has,
on {class c} ∪ {clean}, precision = n_c / (n_c + n_clean) (independent of q) and
recall = q; for Correct (positive = not flagged, undergeneration rows excluded)
precision = n_clean / (n_clean + n_hallucinated) and recall = 1 - q. Same column
definitions as ``compute_metrics.py`` / ``compute_metrics_decoder.py``. On the
toolHACE test (11,606 rows) no input-independent detector exceeds Avg. ~36.7.

    python evaluate/baselines/chance_baselines.py --gold test.parquet \
        [--metrics results/*_metrics.json]

``--metrics`` takes the ``--output-json`` files of the scorers and prints the
chance Avg. at each model's own flag rate next to its Avg.; a lift near zero
means the model's flags carry no information.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SPAN_TYPES = ["answer_mismatch", "overgeneration", "missing_tool"]


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def chance(q: float, n: Counter) -> dict[str, float]:
    hall = sum(n[c] for c in SPAN_TYPES)
    out = {"Correct": f1(n["clean"] / (n["clean"] + hall), 1 - q)}
    out.update({c: f1(n[c] / (n[c] + n["clean"]), q) for c in SPAN_TYPES})
    out["Avg."] = sum(out.values()) / 4
    out["undergeneration"] = f1(n["undergeneration"] / (n["undergeneration"] + n["clean"]), q)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gold", type=Path, required=True, help="test parquet (column `type`)")
    ap.add_argument("--metrics", type=Path, nargs="*", default=[], help="scorer --output-json files")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    n = Counter(pq.read_table(args.gold, columns=["type"]).column("type").to_pylist())
    hall = sum(n[c] for c in SPAN_TYPES)
    best_q = max((i / 1000 for i in range(1001)), key=lambda q: chance(q, n)["Avg."])
    rows = [("Never flag", 0.0), ("Always flag", 1.0), ("Coin flip", 0.5),
            ("Flag at the hallucination rate", hall / (hall + n["clean"])), ("Best random rate", best_q)]
    print(f"class counts: {dict(n)}\n")
    print("| baseline | flag rate | Correct | Mismatch | Overgen. | Missing Tool | Undergen. | Avg. (w/o Undergen.) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, q in rows:
        c = chance(q, n)
        print(f"| {name} | {100 * q:.1f}% | " + " | ".join(f"{100 * c[k]:.2f}" for k in ("Correct", *SPAN_TYPES, "undergeneration"))
              + f" | **{100 * c['Avg.']:.2f}** |")

    if args.metrics:
        print("\n| model | flag rate (w/o undergen) | Avg. | chance Avg. at that rate | lift |")
        print("|---|---|---|---|---|")
        for path in args.metrics:
            m = json.loads(path.read_text(encoding="utf-8"))
            m = m.get("metrics", m)
            b = m["response_level"]["overall_excluding_undergen"]
            q = (b["tp"] + b["fp"]) / (b["tp"] + b["fp"] + b["fn"] + b["tn"])
            avg = m["response_level"]["macro_class_score"]
            ch = chance(q, n)["Avg."]
            print(f"| {m.get('model') or path.stem} | {100 * q:.1f}% | {100 * avg:.2f} | {100 * ch:.2f} | {100 * (avg - ch):+.2f} |")


if __name__ == "__main__":
    main()
