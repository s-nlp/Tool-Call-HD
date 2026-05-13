"""
Collect generated JSONL files into ragtruth_final/ with canonical naming.

Reads:
  singlehop_new/type{1,2,3}_output.jsonl      → singlehop_type{1,2,3}.jsonl
  pruneddataset/pruned_type{1,2,3}.jsonl       → multistep_type{1,2,3}.jsonl

Drops rows where the type3 overgeneration span doesn't sit at the end of
the output (these are no-content rows the LLM failed to extend).

Usage:
    python3 scripts/collect_final.py
    python3 scripts/collect_final.py --no-drop-bad-type3  # keep broken rows for inspection
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

SOURCES = [
    ("singlehop_type1", PROJECT_ROOT / "" / "singlehop_new" / "type1_output.jsonl"),
    ("singlehop_type2", PROJECT_ROOT / "" / "singlehop_new" / "type2_output.jsonl"),
    ("singlehop_type3", PROJECT_ROOT / "" / "singlehop_new" / "type3_output.jsonl"),
    ("multistep_type1", PROJECT_ROOT / "" / "pruneddataset" / "pruned_type1.jsonl"),
    ("multistep_type2", PROJECT_ROOT / "" / "pruneddataset" / "pruned_type2.jsonl"),
    ("multistep_type3", PROJECT_ROOT / "" / "pruneddataset" / "pruned_type3.jsonl"),
]

FINAL_DIR = PROJECT_ROOT / "ragtruth_final"


def is_bad_type3_row(row: dict) -> bool:
    """True if the overgeneration span doesn't sit at the end of the output."""
    output = row.get("output", "")
    try:
        labels = json.loads(row.get("hallucination_labels", "[]"))
    except Exception:
        return False
    return any(
        not lab.get("implicit_true") and lab.get("end") != len(output)
        for lab in labels
    )


def get_args():
    p = argparse.ArgumentParser(description="Collect generated JSONL into ragtruth_final/")
    p.add_argument("--no-drop-bad-type3", action="store_true",
                   help="Keep type3 rows with misplaced spans (default: drop them)")
    p.add_argument("--final-dir", default=str(FINAL_DIR),
                   help="Output directory (default: ragtruth_final/)")
    return p.parse_args()


def main():
    args = get_args()
    final_dir = Path(args.final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting into {final_dir}/\n")

    total_rows = total_dropped = 0

    for name, src in SOURCES:
        if not src.exists():
            print(f"  MISSING  {src.relative_to(PROJECT_ROOT)}")
            continue

        with open(src, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        dropped = 0
        if "type3" in name and not args.no_drop_bad_type3:
            clean = [r for r in rows if not is_bad_type3_row(r)]
            dropped = len(rows) - len(clean)
            rows = clean

        # Re-id sequentially
        for i, r in enumerate(rows):
            r["id"] = i + 1

        out = final_dir / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        implicit = sum(
            1 for r in rows
            if any(lab.get("implicit_true")
                   for lab in json.loads(r.get("hallucination_labels", "[]")))
        )
        explicit = len(rows) - implicit
        drop_msg = f"  (dropped {dropped} bad-span rows)" if dropped else ""
        print(f"  {name:<25}  {len(rows):>6} rows  "
              f"(explicit={explicit}, implicit={implicit}){drop_msg}")
        total_rows += len(rows)
        total_dropped += dropped

    print(f"\nTotal: {total_rows} rows written to {final_dir}/")
    if total_dropped:
        print(f"Dropped: {total_dropped} bad-span type3 rows")


if __name__ == "__main__":
    main()
