#!/usr/bin/env python3
"""Concatenate table rows from compute_metrics.py / compute_metrics_decoder.py.

    python evaluate/baselines/merge_tables.py results/*_table.csv \
        --output-csv results/all_models.csv --markdown

Rows keep the shared column order from ``toolhace_metrics_common.TABLE_COLUMNS``;
``--markdown`` also prints a Markdown table (``—`` marks columns that are not
applicable to a detector, e.g. ``Undergen.`` for span-only models).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from toolhace_metrics_common import TABLE_COLUMNS  # noqa: E402


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({column: row.get(column, "—") or "—" for column in TABLE_COLUMNS})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-model table rows into one table.")
    parser.add_argument("tables", type=Path, nargs="+", help="CSV files written with --output-csv")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table.")
    parser.add_argument(
        "--sort-by", default=None, help="Column to sort by, descending (e.g. 'Response-level Avg. (w/o Undergen.)')."
    )
    args = parser.parse_args()

    rows = read_rows(args.tables)
    if args.sort_by:
        def key(row: dict[str, str]) -> float:
            try:
                return float(row.get(args.sort_by, ""))
            except ValueError:
                return float("-inf")

        rows.sort(key=key, reverse=True)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Merged {len(rows)} rows into {args.output_csv}")

    if args.markdown or not args.output_csv:
        print("| " + " | ".join(TABLE_COLUMNS) + " |")
        print("|" + "---|" * len(TABLE_COLUMNS))
        for row in rows:
            print("| " + " | ".join(row[column] for column in TABLE_COLUMNS) + " |")


if __name__ == "__main__":
    main()
