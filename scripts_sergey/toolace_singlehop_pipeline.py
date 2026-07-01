import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

from toolace_multistep_pipeline import (
    OpenAI,
    add_input_args,
    clean_last_assistant_message,
    extract_examples,
    find_completed_tool_turns,
    heuristic_clean_text,
    is_tool_call_message,
    load_source_rows,
    looks_like_followup_candidate,
    normalize_messages,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract single-hop ToolACE dialogues and optionally clean last assistant follow-ups."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Keep only dialogues with exactly one completed tool-use turn and exactly one tool call in the dialogue.",
    )
    add_input_args(extract_parser)
    extract_parser.add_argument(
        "--output",
        required=True,
        help="Where to write the extracted dataset JSON.",
    )
    extract_parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional path for a JSON summary file.",
    )
    extract_parser.add_argument(
        "--examples-output",
        default=None,
        help="Optional path for a JSON file with presentation-ready examples.",
    )
    extract_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only inspect the first N rows from the source dataset.",
    )
    extract_parser.add_argument(
        "--examples-count",
        type=int,
        default=5,
        help="How many examples to save.",
    )

    clean_parser = subparsers.add_parser(
        "clean-tool-answer-followup",
        help="Remove trailing follow-up questions from the assistant answer for the only tool-use turn.",
    )
    clean_parser.add_argument(
        "--input",
        required=True,
        help="Path to the extracted single-hop dataset JSON.",
    )
    clean_parser.add_argument(
        "--output",
        required=True,
        help="Path for the cleaned dataset JSON.",
    )
    clean_parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional path for a JSON cleaning summary file.",
    )
    clean_parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model used for cleaning.",
    )
    clean_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel OpenAI requests for cleaning.",
    )
    clean_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per row for OpenAI requests.",
    )
    clean_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N extracted rows.",
    )
    clean_parser.add_argument(
        "--process-all",
        action="store_true",
        help="Send every row to the model instead of prefiltering likely follow-up candidates.",
    )
    clean_parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Use a local regex-based cleaner instead of the OpenAI API.",
    )

    return parser


def count_tool_calls(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if is_tool_call_message(message))


def build_extract_summary(
    total_rows: int,
    singlehop_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "total_rows_seen": total_rows,
        "singlehop_rows_kept": len(singlehop_rows),
        "selection_rule": (
            "exactly one tool call message in the whole dialogue"
        ),
        "examples": examples,
    }


def run_extract(args: argparse.Namespace) -> None:
    rows = load_source_rows(args)
    if args.limit is not None:
        rows = rows[:args.limit]

    singlehop_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        messages = normalize_messages(row)
        completed_turns = find_completed_tool_turns(messages)
        tool_call_count = count_tool_calls(messages)

        if tool_call_count != 1:
            continue

        enriched_row = deepcopy(row)
        enriched_row["analysis"] = {
            "dialogue_id": row.get("id", f"toolace_{row_index:05d}"),
            "source_row_index": row_index,
            "completed_tool_turn_count": len(completed_turns),
            "completed_tool_turns": completed_turns,
            "tool_call_count_in_dialogue": tool_call_count,
            "normalized_messages": messages,
            "selection_type": "singlehop_exactly_one_tool_call",
        }
        singlehop_rows.append(enriched_row)

    print(f"Loaded {len(rows)} rows")
    print(
        "Found "
        f"{len(singlehop_rows)} single-hop rows "
        "with exactly one tool call"
    )

    examples = extract_examples(singlehop_rows, args.examples_count)
    summary = build_extract_summary(
        total_rows=len(rows),
        singlehop_rows=singlehop_rows,
        examples=examples,
    )

    write_json(Path(args.output), singlehop_rows)
    print(f"Wrote extracted dataset to {args.output}")

    if args.summary_output:
        write_json(Path(args.summary_output), summary)
        print(f"Wrote summary to {args.summary_output}")

    if args.examples_output:
        write_json(Path(args.examples_output), examples)
        print(f"Wrote examples to {args.examples_output}")


def run_clean_tool_answer_followup(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.limit is not None:
        rows = rows[:args.limit]

    candidate_indices: list[int] = []
    target_answer_indices: list[int | None] = []

    for idx, row in enumerate(rows):
        analysis = row.get("analysis", {})
        completed_turns = analysis.get("completed_tool_turns", [])
        if len(completed_turns) != 1:
            target_answer_indices.append(None)
            continue

        answer_index = completed_turns[0].get("assistant_answer_index")
        if not isinstance(answer_index, int):
            target_answer_indices.append(None)
            continue

        target_answer_indices.append(answer_index)

        messages = normalize_messages(row)
        text = messages[answer_index]["content"]
        if args.process_all or looks_like_followup_candidate(text):
            candidate_indices.append(idx)

    print(f"Loaded {len(rows)} extracted rows")
    print(f"Selected {len(candidate_indices)} candidate rows for tool-answer cleaning")

    if not candidate_indices:
        write_json(Path(args.output), rows)
        print(f"No candidates found. Wrote unchanged dataset to {args.output}")
        return

    use_openai = not args.heuristic_only and OpenAI is not None and bool(os.getenv("OPENAI_API_KEY"))
    if args.heuristic_only:
        print("Cleaning mode: heuristic-only")
    elif use_openai:
        print(f"Cleaning mode: OpenAI model {args.model}")
    else:
        print("Cleaning mode: heuristic fallback (OPENAI_API_KEY not set)")

    client = OpenAI() if use_openai else None
    changed_count = 0
    failures: list[tuple[int, str]] = []
    lock = threading.Lock()
    completed_count = 0

    def worker(row_index: int) -> tuple[int, dict[str, Any]]:
        row = rows[row_index]
        messages = normalize_messages(row)
        answer_index = target_answer_indices[row_index]
        if answer_index is None:
            raise ValueError("Missing tool-answer index")
        if use_openai:
            result = clean_last_assistant_message(
                client=client,
                model=args.model,
                row=row,
                messages=messages,
                target_index=answer_index,
                max_retries=args.max_retries,
            )
        else:
            result = heuristic_clean_text(messages[answer_index]["content"])
        return row_index, result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, idx): idx for idx in candidate_indices}
        for future in as_completed(futures):
            row_index = futures[future]
            try:
                _, result = future.result()
                answer_index = target_answer_indices[row_index]
                if answer_index is None:
                    continue
                if result["changed"]:
                    changed_count += 1
                    rows[row_index]["analysis"] = rows[row_index].get("analysis", {})
                    rows[row_index]["analysis"]["last_assistant_followup_removed"] = True
                    rows[row_index]["analysis"]["last_assistant_followup_cleaning_method"] = (
                        "openai" if use_openai else "heuristic"
                    )
                    rows[row_index]["analysis"]["last_assistant_answer_index"] = answer_index

                    original_messages = normalize_messages(rows[row_index])
                    original_text = original_messages[answer_index]["content"]
                    rows[row_index]["analysis"]["last_assistant_answer_original"] = original_text
                    rows[row_index]["analysis"]["last_assistant_answer_cleaned"] = result["cleaned_text"]

                    message_key = "conversations" if isinstance(rows[row_index].get("conversations"), list) else "messages"
                    content_key = "value" if message_key == "conversations" else "content"
                    original_index = original_messages[answer_index]["original_index"]
                    rows[row_index][message_key][original_index][content_key] = result["cleaned_text"]
                with lock:
                    completed_count += 1
                    if completed_count % 25 == 0 or completed_count == len(futures):
                        print(
                            f"Cleaned {completed_count}/{len(futures)} candidates | "
                            f"changed={changed_count} failed={len(failures)}"
                        )
            except Exception as exc:
                with lock:
                    failures.append((row_index, str(exc)))
                    completed_count += 1
                    if completed_count % 25 == 0 or completed_count == len(futures):
                        print(
                            f"Cleaned {completed_count}/{len(futures)} candidates | "
                            f"changed={changed_count} failed={len(failures)}"
                        )

    write_json(Path(args.output), rows)
    print(f"Wrote cleaned dataset to {args.output}")

    if args.summary_output:
        summary = {
            "total_rows_seen": len(rows),
            "candidate_rows": len(candidate_indices),
            "changed_rows": changed_count,
            "failed_rows": len(failures),
            "failure_samples": [
                {"row_index": idx, "error": message}
                for idx, message in failures[:10]
            ],
            "target_message": "assistant answer attached to the only completed tool-use turn",
        }
        write_json(Path(args.summary_output), summary)
        print(f"Wrote cleaning summary to {args.summary_output}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        run_extract(args)
        return

    if args.command == "clean-tool-answer-followup":
        run_clean_tool_answer_followup(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
