import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


FOLLOWUP_PATTERNS = [
    r"what would you like to do next\??",
    r"is there anything else you would like to know\??",
    r"if you need .*? please let me know!?",
    r"if you need .*? feel free to ask!?",
    r"let me know if you need .*",
    r"feel free to ask!?",
    r"would you like .*?\??",
]


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "changed": {"type": "boolean"},
        "original_answer_clean": {"type": "string"},
        "tagged_answer_clean": {"type": "string"},
    },
    "required": ["changed", "original_answer_clean", "tagged_answer_clean"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove trailing assistant follow-up questions and generic call-to-action "
            "phrases from dataset rows using parallel OpenAI API calls."
        )
    )
    parser.add_argument(
        "--input",
        default="dataset_v3_tagged.json",
        help="Path to the source JSON dataset.",
    )
    parser.add_argument(
        "--output",
        default="dataset_v3_tagged_cleaned.json",
        help="Path for the cleaned JSON dataset.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model to use for cleaning.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel API requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N candidate rows.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per row if an API call fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write output. Print a short preview instead.",
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Send every row to the model instead of prefiltering likely candidates.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in {path}")
    return data


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def looks_like_candidate(text: str) -> bool:
    if not text.strip():
        return False

    lowered_tail = text.lower()[-350:]
    if "?" in lowered_tail:
        return True

    return any(re.search(pattern, lowered_tail, flags=re.IGNORECASE) for pattern in FOLLOWUP_PATTERNS)


def candidate_indices(rows: list[dict[str, Any]], process_all: bool, limit: int | None) -> list[int]:
    if process_all:
        indices = list(range(len(rows)))
    else:
        indices = [
            idx
            for idx, row in enumerate(rows)
            if looks_like_candidate(str(row.get("original_answer", "")))
        ]
    if limit is not None:
        indices = indices[:limit]
    return indices


def build_prompt(row: dict[str, Any]) -> str:
    payload = {
        "user_prompt": row.get("user_prompt", ""),
        "tool_call": row.get("tool_call", ""),
        "tool_response": row.get("tool_response", ""),
        "original_answer": row.get("original_answer", ""),
        "tagged_answer": row.get("tagged_answer", ""),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def clean_row(
    client: OpenAI,
    row: dict[str, Any],
    model: str,
    max_retries: int,
) -> dict[str, Any]:
    instructions = (
        "You clean dataset rows for a hallucination-detection project.\n"
        "Remove only trailing assistant follow-up questions, conversational invitations, "
        "or generic call-to-action endings such as asking what to do next, asking whether "
        "the user needs more help, or saying 'let me know' / 'feel free to ask'.\n"
        "Keep all substantive answer content grounded in the tool response.\n"
        "Do not add new content. Do not rewrite earlier sentences unless a tiny punctuation "
        "or whitespace fix is required after removing the trailing follow-up.\n"
        "Clean original_answer and tagged_answer consistently. Preserve all existing tags in "
        "tagged_answer and only remove the mirrored follow-up segment there.\n"
        "If there is no trailing follow-up or generic assistant outro, return both answers unchanged.\n"
        "Return JSON only."
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=build_prompt(row),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "cleaned_answers",
                        "strict": True,
                        "schema": OUTPUT_SCHEMA,
                    }
                },
            )
            parsed = json.loads(response.output_text)
            return {
                "changed": bool(parsed["changed"]),
                "original_answer_clean": parsed["original_answer_clean"],
                "tagged_answer_clean": parsed["tagged_answer_clean"],
            }
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError("Unexpected retry loop exit")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")

    rows = load_dataset(input_path)
    original_rows = json.loads(json.dumps(rows, ensure_ascii=False))
    indices = candidate_indices(rows, process_all=args.process_all, limit=args.limit)

    print(f"Loaded {len(rows)} rows from {input_path}")
    print(f"Selected {len(indices)} candidate rows for cleaning")

    if not indices:
        if not args.dry_run:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            print(f"No candidates found. Wrote unchanged dataset to {output_path}")
        return

    client = OpenAI()
    changed_count = 0
    failed_count = 0
    lock = threading.Lock()
    failures: list[tuple[int, str]] = []

    def worker(idx: int) -> tuple[int, dict[str, Any]]:
        row = rows[idx]
        result = clean_row(
            client=client,
            row=row,
            model=args.model,
            max_retries=args.max_retries,
        )
        return idx, result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, idx): idx for idx in indices}

        for completed, future in enumerate(as_completed(futures), start=1):
            idx = futures[future]
            try:
                row_idx, result = future.result()
                original_clean = result["original_answer_clean"].strip()
                tagged_clean = result["tagged_answer_clean"].strip()

                row_was_changed = (
                    normalize_whitespace(original_clean)
                    != normalize_whitespace(str(original_rows[row_idx].get("original_answer", "")))
                    or normalize_whitespace(tagged_clean)
                    != normalize_whitespace(str(original_rows[row_idx].get("tagged_answer", "")))
                )

                if result["changed"] or row_was_changed:
                    rows[row_idx]["original_answer"] = original_clean
                    rows[row_idx]["tagged_answer"] = tagged_clean
                    with lock:
                        changed_count += 1

                if completed % 25 == 0 or completed == len(indices):
                    print(
                        f"Processed {completed}/{len(indices)} candidates | "
                        f"changed={changed_count} failed={failed_count}"
                    )
            except Exception as exc:
                with lock:
                    failed_count += 1
                    failures.append((idx, str(exc)))

    if args.dry_run:
        print("\nDry run preview:")
        preview_count = 0
        for idx in indices:
            original = rows[idx].get("original_answer", "")
            tagged = rows[idx].get("tagged_answer", "")
            if (
                normalize_whitespace(original)
                != normalize_whitespace(str(original_rows[idx].get("original_answer", "")))
                or normalize_whitespace(tagged)
                != normalize_whitespace(str(original_rows[idx].get("tagged_answer", "")))
            ):
                print(f"\nRow {idx}")
                print("original_answer:", original)
                print("tagged_answer:", tagged)
                preview_count += 1
            if preview_count >= 5:
                break
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\nWrote cleaned dataset to {output_path}")

    print(f"Rows changed: {changed_count}")
    print(f"Rows failed: {failed_count}")
    if failures:
        print("Failure samples:")
        for idx, message in failures[:10]:
            print(f"  - row {idx}: {message}")


if __name__ == "__main__":
    main()
