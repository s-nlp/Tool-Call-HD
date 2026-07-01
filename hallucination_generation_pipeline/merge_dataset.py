"""
Merge final_dataset/ into one mega JSONL.

Structure:
  - All hallucinated rows (hall=1) from all 6 splits — kept as-is
  - Clean rows (hall=0) deduplicated by (query, output) — each unique
    clean sample appears only once regardless of how many type files it
    was in

Output:  merged_dataset/merged.jsonl

Usage:
    python3 scripts/merge_dataset.py
    python3 scripts/merge_dataset.py --input-dir final_dataset --out merged_dataset/merged.jsonl
"""
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent / "output"

SINGLEHOP = ["singlehop_incorrect_info_type1", "singlehop_undergeneration_type2", "singlehop_overgeneration_type3"]
MULTISTEP  = ["multistep_incorrect_info_type1", "multistep_undergeneration_type2", "multistep_overgeneration_type3"]
ALL_SPLITS = SINGLEHOP + MULTISTEP


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=str(PROJECT_ROOT / "final_dataset"))
    p.add_argument("--out",       default=str(PROJECT_ROOT / "merged_dataset" / "merged.jsonl"))
    return p.parse_args()


def main():
    args = get_args()
    inp = Path(args.input_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Only pick up JSONL files — skip raw conversation JSON files
    found = sorted(inp.glob("*.jsonl"))
    if not found:
        print(f"No .jsonl/.json files found in {inp}")
        return

    hall_rows = []

    print(f"Reading from {inp}/\n")
    print(f"{'File':<35} {'hall':>6} {'skipped':>8}")
    print("─" * 55)

    for path in found:
        with open(path) as f:
            try:
                rows = json.load(f) if path.suffix == ".json" else [json.loads(l) for l in f if l.strip()]
            except Exception as e:
                print(f"  SKIP {path.name}: {e}")
                continue

        if not rows or not isinstance(rows, list):
            continue

        h = skipped = 0
        for r in rows:
            if "hallucination" in r:
                is_hall = r["hallucination"] == 1
            else:
                try:
                    labs = json.loads(r.get("hallucination_labels", "[]"))
                except Exception:
                    labs = []
                explicit = [l for l in labs if not l.get("implicit_true")
                            and l.get("end", 0) > l.get("start", -1)]
                if not explicit:
                    skipped += 1
                    continue  # empty labels = generation failure
                is_hall = True

            if is_hall:
                hall_rows.append(r)
                h += 1

        print(f"  {path.name:<33} {h:>6} {skipped:>8}")

    print("─" * 55)
    print(f"  {'TOTAL':<33} {len(hall_rows):>6}")

    for i, r in enumerate(hall_rows):
        r["id"] = i + 1

    with open(out, "w") as f:
        for r in hall_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nMerged → {out}")
    print(f"  Total rows : {len(hall_rows)}")
    print(f"\nNext step:")
    print(f"  python3 scripts/make_lettucedetect_data.py --input-dir merged_dataset --out-dir lettucedetect_data")


if __name__ == "__main__":
    main()
