"""
Convert final_dataset/ JSONL files → LettuceDetect JSON format.

LettuceDetect schema per sample:
  prompt    : grounding context shown to the model
  answer    : model output to check for hallucinations
  labels    : [{start, end, label}] with char offsets into answer ([] for clean)
  split     : "train" | "dev" | "test"
  task_type : original task type string
  dataset   : "tool_calling_hallucination"
  language  : "en"

Usage:
    python3 scripts/make_lettucedetect_data.py
    python3 scripts/make_lettucedetect_data.py --input-dir final_dataset --out-dir lettucedetect_data
    python3 scripts/make_lettucedetect_data.py --dev-ratio 0.1 --test-ratio 0.1
"""

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent / "output"

SPLITS = [
    "singlehop_incorrect_info_type1",
    "singlehop_undergeneration_type2",
    "singlehop_overgeneration_type3",
    "multistep_incorrect_info_type1",
    "multistep_undergeneration_type2",
    "multistep_overgeneration_type3",
]

# ── Override these on the server if needed ───────────────────────────────────
# PROJECT_ROOT = Path("/workspace/dimabsa/workingsolution/output_glaive_jsonl")
# SPLITS = ["type1_output", "type2_output", "type3_output"]


def convert_row(row: dict) -> dict | None:
    """Convert one row to a LettuceDetect sample dict.

    Handles two source formats:
      - final_dataset format : has 'hallucination' (0|1) field
      - ragtruth JSONL format : has 'hallucination_labels' with spans, no 'hallucination' field
    """
    output = row.get("output", "")
    context = row.get("context", "")
    query = row.get("query", "")

    if not output:
        return None

    prompt = f"{query}\n\n{context}".strip() if query else context

    # ── Detect format and extract spans ──────────────────────────────────────
    try:
        raw_labels = json.loads(row.get("hallucination_labels", "[]"))
    except Exception:
        raw_labels = []

    # final_dataset format: explicit hallucination field
    if "hallucination" in row:
        is_hall = row["hallucination"] == 1
        candidate_labels = raw_labels if is_hall else []
    else:
        # ragtruth format: treat non-empty, non-implicit labels as hallucinated
        candidate_labels = raw_labels
        is_hall = any(not lab.get("implicit_true") and lab.get("end", 0) > lab.get("start", -1)
                      for lab in raw_labels)

    labels = []
    for lab in candidate_labels:
        s, e = lab.get("start", -1), lab.get("end", -1)
        if s < 0 or e <= s or e > len(output):
            continue
        if lab.get("implicit_true"):
            continue
        labels.append({"start": s, "end": e, "label": "hallucination"})

    return {
        "prompt":    prompt,
        "answer":    output,
        "labels":    labels,
        "split":     "train",
        "task_type": row.get("task_type", ""),
        "dataset":   "tool_calling_hallucination",
        "language":  "en",
    }


def assign_splits(samples: list, dev_ratio: float, test_ratio: float,
                  seed: int = 42) -> list:
    """Shuffle and assign train/dev/test splits in-place."""
    rng = random.Random(seed)
    rng.shuffle(samples)
    n = len(samples)
    n_test = max(1, int(n * test_ratio))
    n_dev  = max(1, int(n * dev_ratio))
    for i, s in enumerate(samples):
        if i < n_test:
            s["split"] = "test"
        elif i < n_test + n_dev:
            s["split"] = "dev"
        else:
            s["split"] = "train"
    return samples


def get_args():
    p = argparse.ArgumentParser(description="Convert final_dataset → LettuceDetect format")
    p.add_argument("--input-dir",    default=str(PROJECT_ROOT / "final_dataset"),
                   help="Directory containing hallucinated JSONL files")
    p.add_argument("--clean-source", default=None,
                   help="Path to original source dataset (.json or .jsonl) used as clean examples. "
                        "Rows must have user_prompt/tool_response/original_answer fields.")
    p.add_argument("--out-dir",      default=str(PROJECT_ROOT / "lettucedetect_data"),
                   help="Output directory (default: lettucedetect_data/)")
    p.add_argument("--dev-ratio",    type=float, default=0.1)
    p.add_argument("--test-ratio",   type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--no-clean",     action="store_true",
                   help="Exclude clean rows entirely")
    return p.parse_args()


def main():
    args = get_args()
    inp = Path(args.input_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_samples = []
    print(f"Loading from {inp}/\n")

    # Accept either a single merged.jsonl or the 6-file directory
    if inp.is_file() and inp.suffix == ".jsonl":
        input_files = [(inp.stem, inp)]
    else:
        input_files = [(name, inp / f"{name}.jsonl") for name in SPLITS]
        # Also accept merged.jsonl inside the directory
        merged = inp / "merged.jsonl"
        if merged.exists() and not any(p.exists() for _, p in input_files):
            input_files = [("merged", merged)]

    for split_name, path in input_files:
        if not path.exists():
            print(f"  MISSING  {path.name} — skipped")
            continue

        with open(path) as f:
            rows = [json.loads(l) for l in f]

        converted = 0
        skipped = 0
        for row in rows:
            if args.no_clean and row.get("hallucination", 0) == 0:
                continue
            sample = convert_row(row)
            if sample is None:
                skipped += 1
                continue
            all_samples.append(sample)
            converted += 1

        hall  = sum(1 for r in rows if r.get("hallucination") == 1)
        clean = sum(1 for r in rows if r.get("hallucination") == 0)
        print(f"  {split_name:<25}  {len(rows):>6} rows  "
              f"(hall={hall}, clean={clean})  →  {converted} converted, {skipped} skipped")

    # ── Load clean source (original dataset) ─────────────────────────────────
    if args.clean_source and not args.no_clean:
        clean_path = Path(args.clean_source)
        print(f"\nLoading clean source: {clean_path}")
        if clean_path.suffix == ".jsonl":
            with open(clean_path) as f:
                clean_src = [json.loads(l) for l in f]
        else:
            with open(clean_path) as f:
                clean_src = json.load(f)

        clean_added = 0
        seen = {(s["prompt"], s["answer"]) for s in all_samples}
        for row in clean_src:
            # Support both ragtruth format and raw generation format
            query   = row.get("user_prompt") or row.get("query", "")
            context = row.get("tool_response") or row.get("context", "")
            output  = row.get("original_answer") or row.get("output", "")
            if not output:
                continue
            prompt = f"{query}\n\n{context}".strip() if query else context
            key = (prompt, output)
            if key in seen:
                continue
            seen.add(key)
            all_samples.append({
                "prompt":    prompt,
                "answer":    output,
                "labels":    [],
                "split":     "train",
                "task_type": "clean",
                "dataset":   "tool_calling_hallucination",
                "language":  "en",
            })
            clean_added += 1

        print(f"  Added {clean_added} unique clean samples from source")

    print(f"\nTotal samples: {len(all_samples)}")

    # Assign splits
    all_samples = assign_splits(all_samples, args.dev_ratio, args.test_ratio, args.seed)

    train = [s for s in all_samples if s["split"] == "train"]
    dev   = [s for s in all_samples if s["split"] == "dev"]
    test  = [s for s in all_samples if s["split"] == "test"]

    print(f"Split: train={len(train)}  dev={len(dev)}  test={len(test)}")

    # ── Save combined file (what LettuceDetect train.py loads) ───────────────
    combined_path = out / "tool_calling_hallucination.json"
    with open(combined_path, "w") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"\nCombined → {combined_path}  ({len(all_samples)} samples)")

    # ── Also save per-split files for convenience ────────────────────────────
    for name, subset in [("train", train), ("dev", dev), ("test", test)]:
        p = out / f"{name}.json"
        with open(p, "w") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"           → {p}  ({len(subset)} samples)")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print("\nLabel distribution:")
    for name, subset in [("train", train), ("dev", dev), ("test", test)]:
        hall_s  = sum(1 for s in subset if s["labels"])
        clean_s = sum(1 for s in subset if not s["labels"])
        print(f"  {name:<6}: {len(subset):>6} total  "
              f"hall={hall_s} ({100*hall_s//len(subset) if subset else 0}%)  "
              f"clean={clean_s}")

    print(f"\nDone. Load in LettuceDetect with:")
    print(f"  HallucinationData.from_json(json.loads(Path('{combined_path}').read_text()))")


if __name__ == "__main__":
    main()
