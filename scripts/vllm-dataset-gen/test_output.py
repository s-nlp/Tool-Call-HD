"""
Format validation for generated ragtruth_final/ JSONL files.

Checks every row in every split for:
  - Required fields present
  - hallucination_labels parseable and non-empty
  - Label start/end are valid offsets into output
  - Type3: overgeneration span sits at the END of output (not mid-output)
  - Type3: hall span is short (< 1000 chars) — catches the original-answer-echoing bug
  - No <hall> tags left in output field (should have been stripped during export)

Usage:
    python3 scripts/test_output.py                       # validate ragtruth_final/
    python3 scripts/test_output.py --dir some/other/dir  # validate a different directory
    python3 scripts/test_output.py --verbose             # print details on each failure
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REQUIRED_FIELDS = [
    "id", "query", "context", "output", "task_type",
    "hallucination_labels", "hallucination_labels_processed", "input_str",
]
HALL_TAG_RE = re.compile(r"</?hall>")


def check_file(path: Path, verbose: bool) -> tuple[int, int]:
    """Returns (rows_checked, failures)."""
    rows_checked = failures = 0
    is_type3 = "type3" in path.name

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            row = json.loads(line)
            rows_checked += 1
            row_errors = []

            # 1. Required fields
            missing = [k for k in REQUIRED_FIELDS if k not in row]
            if missing:
                row_errors.append(f"missing fields: {missing}")

            # 2. No <hall> tags in output
            output = row.get("output", "")
            if HALL_TAG_RE.search(output):
                row_errors.append("<hall> tag found in output (should be stripped)")

            # 3. Labels parseable and non-empty
            try:
                labels = json.loads(row.get("hallucination_labels", "[]"))
            except Exception as e:
                row_errors.append(f"hallucination_labels not valid JSON: {e}")
                labels = []

            if not labels:
                row_errors.append("no hallucination_labels")

            # 4. Offset validity
            for lab in labels:
                s, e = lab.get("start", -1), lab.get("end", -1)
                if s < 0 or e < 0:
                    row_errors.append(f"label has negative offset: start={s} end={e}")
                elif e > len(output):
                    row_errors.append(f"label end={e} > output length={len(output)}")
                elif s > e:
                    row_errors.append(f"label start={s} > end={e}")

                # 5. Type3-specific: span must be at end, must be short
                if is_type3 and not lab.get("implicit_true"):
                    if e != len(output):
                        span_text = output[s:e] if s >= 0 and e <= len(output) else ""
                        row_errors.append(
                            f"type3 span not at end: [{s}:{e}] (output_len={len(output)}) "
                            f"— '{span_text[:60]}...'"
                        )
                    if (e - s) > 1000:
                        row_errors.append(
                            f"type3 span too long ({e - s} chars) — likely original answer echo"
                        )

            if row_errors:
                failures += 1
                if verbose:
                    print(f"  FAIL row {lineno} (id={row.get('id','?')}): "
                          + " | ".join(row_errors))

    return rows_checked, failures


def get_args():
    p = argparse.ArgumentParser(description="Validate ragtruth JSONL format")
    p.add_argument("--dir", default=str(PROJECT_ROOT / "ragtruth_final"),
                   help="Directory to validate (default: ragtruth_final/)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print details on each failed row")
    return p.parse_args()


def main():
    args = get_args()
    target = Path(args.dir)

    if not target.exists():
        print(f"ERROR: {target} does not exist")
        sys.exit(1)

    files = sorted(target.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {target}")
        sys.exit(1)

    print(f"Validating {len(files)} files in {target}/\n")

    total_rows = total_failures = 0
    all_ok = True

    for p in files:
        rows, fails = check_file(p, args.verbose)
        total_rows += rows
        total_failures += fails
        status = "OK" if fails == 0 else f"FAIL ({fails} bad rows)"
        print(f"  {p.name:<30}  {rows:>6} rows   {status}")
        if fails:
            all_ok = False

    print(f"\n{'='*55}")
    if all_ok:
        print(f"ALL PASS — {total_rows} rows checked, 0 failures")
    else:
        print(f"FAILURES: {total_failures}/{total_rows} rows have issues")
        print("Run with --verbose to see details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
