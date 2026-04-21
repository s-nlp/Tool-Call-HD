import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


FOLLOWUP_PATTERNS = [
    r"what would you like to do next\??",
    r"is there anything else you would like to know\??",
    r"is there anything else .*?(?:need|like)\??",
    r"if you need .*? please let me know!?",
    r"if you need .*? feel free to ask!?",
    r"let me know if you need .*",
    r"feel free to ask!?",
    r"would you like .*?\??",
    r"you may also check .*",
    r"i can also .*",
]

EXTRACT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": [
                "same_task_multihop",
                "same_dialog_but_unrelated",
                "not_multistep",
            ],
        },
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}

CLEAN_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "changed": {"type": "boolean"},
        "cleaned_text": {"type": "string"},
    },
    "required": ["changed", "cleaned_text"],
    "additionalProperties": False,
}

FOLLOWUP_FULL_PATTERNS = [
    r"(?:what would you like to do next\??)",
    r"(?:is there anything else(?: you would like to know)?\??)",
    r"(?:would you like(?: me)? to .*?\??)",
    r"(?:do you want me to .*?\??)",
    r"(?:if you need .*?(?:please )?let me know!?)+",
    r"(?:if you need .*?feel free to ask!?)+",
    r"(?:let me know if you need .*?)+",
    r"(?:feel free to ask!?)+",
    r"(?:you may also check .*?)+",
    r"(?:i can also .*?)+",
    r"(?:if you'd like,? i can .*?)+",
    r"(?:if you want,? i can .*?)+",
    r"(?:please let me know if .*?)+",
    r"(?:can i help with anything else\??)",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract multi-step dialogues from ToolACE and clean last assistant follow-ups."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Find dialogues with at least N completed tool-use user turns.",
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
        "--min-completed-tool-turns",
        type=int,
        default=2,
        help="Minimum number of completed tool-use user turns per dialogue.",
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
        help="How many examples to save for presentation.",
    )
    extract_parser.add_argument(
        "--llm-filter",
        action="store_true",
        help=(
            "Use an OpenAI model to keep only same-task multi-hop dialogues and "
            "drop unrelated multi-turn conversations."
        ),
    )
    extract_parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model used for --llm-filter and clean-last-followup.",
    )
    extract_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel OpenAI requests for LLM filtering.",
    )
    extract_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per row for OpenAI requests.",
    )

    clean_parser = subparsers.add_parser(
        "clean-last-followup",
        help="Remove trailing follow-up questions from the last assistant answer.",
    )
    clean_parser.add_argument(
        "--input",
        required=True,
        help="Path to the extracted multi-step dataset JSON.",
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


def add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        default=None,
        help="Local input dataset path (.json or .jsonl).",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help='Optional Hugging Face dataset name, for example "ankush13r/ToolACE".',
    )
    parser.add_argument(
        "--hf-split",
        default="train",
        help="Split to load from the Hugging Face dataset.",
    )


def require_openai() -> None:
    if OpenAI is None:
        raise ImportError("openai is not installed")
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")


def load_source_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input:
        return load_rows_from_path(Path(args.input))

    if args.hf_dataset:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'Install "datasets" to load ToolACE directly from Hugging Face.'
            ) from exc

        dataset = load_dataset(args.hf_dataset, split=args.hf_split)
        return [dict(row) for row in dataset]

    raise ValueError("Provide either --input or --hf-dataset")


def load_rows_from_path(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected top-level JSON array in {path}")
        return data

    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object on line {line_number} in {path}")
                rows.append(row)
        return rows

    raise ValueError(f"Unsupported input format for {path}; use .json or .jsonl")


def normalize_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_messages = row.get("conversations")
    if raw_messages is None:
        raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("Row has no conversations/messages list")

    normalized: list[dict[str, Any]] = []
    for idx, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message #{idx} is not an object")
        role = message.get("from", message.get("role", ""))
        content = message.get("value", message.get("content", ""))
        normalized.append(
            {
                "role": str(role).strip().lower(),
                "content": str(content),
                "original_index": idx,
            }
        )
    return normalized


def looks_like_tool_call_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return False
    inner = stripped[1:-1]
    return bool(re.search(r'[A-Za-z0-9_./ -]+\([^][]*\)', inner))


def is_tool_call_message(message: dict[str, Any]) -> bool:
    role = message["role"]
    content = message["content"]
    if role in {"tool_call", "function_call"}:
        return True
    return role == "assistant" and looks_like_tool_call_text(content)


def is_tool_result_message(message: dict[str, Any]) -> bool:
    role = message["role"]
    if role in {"tool", "tool_output", "function", "observation"}:
        return True
    return False


def is_user_message(message: dict[str, Any]) -> bool:
    return message["role"] == "user"


def is_plain_assistant_message(message: dict[str, Any]) -> bool:
    return message["role"] == "assistant" and not is_tool_call_message(message)


def find_completed_tool_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    user_indices = [idx for idx, message in enumerate(messages) if is_user_message(message)]
    completed_turns: list[dict[str, Any]] = []

    for turn_number, user_index in enumerate(user_indices):
        next_user_index = user_indices[turn_number + 1] if turn_number + 1 < len(user_indices) else len(messages)
        segment = messages[user_index + 1:next_user_index]

        tool_call_indices = [user_index + 1 + idx for idx, msg in enumerate(segment) if is_tool_call_message(msg)]
        if not tool_call_indices:
            continue

        first_tool_call_index = tool_call_indices[0]
        tool_result_indices = [
            user_index + 1 + idx
            for idx, msg in enumerate(segment)
            if is_tool_result_message(msg) and (user_index + 1 + idx) > first_tool_call_index
        ]
        if not tool_result_indices:
            continue

        last_tool_result_index = tool_result_indices[-1]
        assistant_answer_index = None
        for idx in range(last_tool_result_index + 1, next_user_index):
            if is_plain_assistant_message(messages[idx]):
                assistant_answer_index = idx
                break

        if assistant_answer_index is None:
            continue

        completed_turns.append(
            {
                "turn_number": len(completed_turns) + 1,
                "user_index": user_index,
                "assistant_tool_call_indices": tool_call_indices,
                "tool_result_indices": tool_result_indices,
                "assistant_answer_index": assistant_answer_index,
                "user_text": messages[user_index]["content"],
                "tool_call_texts": [messages[idx]["content"] for idx in tool_call_indices],
                "tool_result_texts": [messages[idx]["content"] for idx in tool_result_indices],
                "assistant_answer_text": messages[assistant_answer_index]["content"],
            }
        )

    return completed_turns


def summarize_dialogue_for_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for idx, message in enumerate(messages, start=1):
        role = message["role"]
        content = message["content"].strip()
        if len(content) > 1200:
            content = content[:1200].rstrip() + "\n...[truncated]"
        parts.append(f"{idx}. {role}: {content}")
    return "\n".join(parts)


def build_multihop_prompt(
    row_index: int,
    row: dict[str, Any],
    messages: list[dict[str, Any]],
    completed_turns: list[dict[str, Any]],
) -> str:
    payload = {
        "row_index": row_index,
        "system_excerpt": str(row.get("system", ""))[:800],
        "completed_tool_turn_count": len(completed_turns),
        "completed_tool_turns": [
            {
                "turn_number": turn["turn_number"],
                "user_text": turn["user_text"],
                "tool_call_texts": turn["tool_call_texts"],
                "assistant_answer_text": turn["assistant_answer_text"],
            }
            for turn in completed_turns
        ],
        "dialogue": summarize_dialogue_for_prompt(messages),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def classify_multihop_dialogue(
    client: Any,
    model: str,
    row_index: int,
    row: dict[str, Any],
    messages: list[dict[str, Any]],
    completed_turns: list[dict[str, Any]],
    max_retries: int,
) -> dict[str, str]:
    instructions = (
        "You classify ToolACE dialogues.\n"
        "Focus on whether the dialogue is a true same-task multi-hop conversation.\n"
        "Definitions:\n"
        "- same_task_multihop: at least two completed tool-use user turns are follow-up steps inside the same broader task.\n"
        "- same_dialog_but_unrelated: the dialogue contains multiple completed tool-use turns, but they are separate topics or unrelated requests inside one chat.\n"
        "- not_multistep: it does not actually contain two completed tool-use user turns.\n"
        "Return JSON only."
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=build_multihop_prompt(
                    row_index=row_index,
                    row=row,
                    messages=messages,
                    completed_turns=completed_turns,
                ),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "toolace_multihop_classification",
                        "strict": True,
                        "schema": EXTRACT_OUTPUT_SCHEMA,
                    }
                },
            )
            parsed = json.loads(response.output_text)
            return {
                "label": parsed["label"],
                "reason": parsed["reason"],
            }
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError("Unexpected retry loop exit")


def extract_examples(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows[:count]:
        completed_turns = row["analysis"]["completed_tool_turns"]
        examples.append(
            {
                "dialogue_id": row["analysis"]["dialogue_id"],
                "source_row_index": row["analysis"]["source_row_index"],
                "completed_tool_turn_count": row["analysis"]["completed_tool_turn_count"],
                "llm_label": row["analysis"].get("llm_label"),
                "llm_reason": row["analysis"].get("llm_reason"),
                "turns": [
                    {
                        "user": turn["user_text"],
                        "tool_calls": turn["tool_call_texts"],
                        "assistant_answer": turn["assistant_answer_text"],
                    }
                    for turn in completed_turns[:2]
                ],
            }
        )
    return examples


def build_extract_summary(
    total_rows: int,
    structural_candidates: list[dict[str, Any]],
    kept_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    llm_counts: dict[str, int] = {}
    for row in structural_candidates:
        label = row["analysis"].get("llm_label", "not_run")
        llm_counts[label] = llm_counts.get(label, 0) + 1

    return {
        "total_rows_seen": total_rows,
        "structural_multi_step_candidates": len(structural_candidates),
        "final_rows_kept": len(kept_rows),
        "llm_label_counts": llm_counts,
        "examples": examples,
    }


def last_assistant_answer_index(messages: list[dict[str, Any]]) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        if is_plain_assistant_message(messages[idx]):
            return idx
    return None


def looks_like_followup_candidate(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered_tail = stripped.lower()[-400:]
    if "?" in lowered_tail:
        return True
    return any(re.search(pattern, lowered_tail, flags=re.IGNORECASE) for pattern in FOLLOWUP_PATTERNS)


def build_clean_prompt(row: dict[str, Any], messages: list[dict[str, Any]], target_index: int) -> str:
    payload = {
        "dialogue_id": row.get("analysis", {}).get("dialogue_id"),
        "last_assistant_index": target_index,
        "last_assistant_text": messages[target_index]["content"],
        "dialogue": summarize_dialogue_for_prompt(messages),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_followup_match_text(text: str) -> str:
    normalized = text.strip().strip("-* \n\t")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.lower()


def is_followup_fragment(text: str) -> bool:
    normalized = normalize_followup_match_text(text)
    if not normalized:
        return False

    return any(
        re.fullmatch(pattern, normalized, flags=re.IGNORECASE)
        for pattern in FOLLOWUP_FULL_PATTERNS
    )


def remove_trailing_followup_paragraph(text: str) -> tuple[str, bool]:
    parts = re.split(r"(\n\s*\n)", text.rstrip())
    if len(parts) < 3:
        return text, False

    trailing_text = parts[-1]
    if not is_followup_fragment(trailing_text):
        return text, False

    candidate = "".join(parts[:-2]).rstrip()
    return candidate, candidate != text.rstrip()


def split_sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r".*?(?:[.!?](?=\s|$)|$)", text, flags=re.DOTALL):
        start, end = match.span()
        if start == end:
            continue
        if not text[start:end].strip():
            continue
        spans.append((start, end))
    return spans


def remove_trailing_followup_sentence(text: str) -> tuple[str, bool]:
    stripped = text.rstrip()
    spans = split_sentence_spans(stripped)
    if not spans:
        return text, False

    start, end = spans[-1]
    trailing_sentence = stripped[start:end]
    if not is_followup_fragment(trailing_sentence):
        return text, False

    candidate = stripped[:start].rstrip()
    return candidate, candidate != stripped


def heuristic_clean_text(text: str) -> dict[str, Any]:
    cleaned = text.rstrip()
    changed = False

    while True:
        next_text, paragraph_changed = remove_trailing_followup_paragraph(cleaned)
        if paragraph_changed:
            cleaned = next_text
            changed = True
            continue

        next_text, sentence_changed = remove_trailing_followup_sentence(cleaned)
        if sentence_changed:
            cleaned = next_text
            changed = True
            continue

        break

    return {
        "changed": changed,
        "cleaned_text": cleaned,
    }


def clean_last_assistant_message(
    client: Any,
    model: str,
    row: dict[str, Any],
    messages: list[dict[str, Any]],
    target_index: int,
    max_retries: int,
) -> dict[str, Any]:
    instructions = (
        "You clean ToolACE assistant answers.\n"
        "Remove only trailing assistant follow-up questions, conversational invitations, or generic "
        "call-to-action endings from the last assistant message.\n"
        "Examples to remove: 'Would you like me to...', 'Let me know if you need...', "
        "'You may also check...', 'Feel free to ask...'.\n"
        "Keep all substantive answer content unchanged.\n"
        "If there is no trailing follow-up, return the original text unchanged.\n"
        "Return JSON only."
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=build_clean_prompt(row=row, messages=messages, target_index=target_index),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "toolace_last_answer_clean",
                        "strict": True,
                        "schema": CLEAN_OUTPUT_SCHEMA,
                    }
                },
            )
            parsed = json.loads(response.output_text)
            return {
                "changed": bool(parsed["changed"]),
                "cleaned_text": parsed["cleaned_text"],
            }
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError("Unexpected retry loop exit")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_extract(args: argparse.Namespace) -> None:
    rows = load_source_rows(args)
    if args.limit is not None:
        rows = rows[:args.limit]

    structural_candidates: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        messages = normalize_messages(row)
        completed_turns = find_completed_tool_turns(messages)
        if len(completed_turns) < args.min_completed_tool_turns:
            continue

        enriched_row = deepcopy(row)
        enriched_row["analysis"] = {
            "dialogue_id": row.get("id", f"toolace_{row_index:05d}"),
            "source_row_index": row_index,
            "completed_tool_turn_count": len(completed_turns),
            "completed_tool_turns": completed_turns,
            "normalized_messages": messages,
        }
        structural_candidates.append(enriched_row)

    print(f"Loaded {len(rows)} rows")
    print(
        "Found "
        f"{len(structural_candidates)} structural multi-step candidates "
        f"with >= {args.min_completed_tool_turns} completed tool-use turns"
    )

    kept_rows = structural_candidates

    if args.llm_filter:
        require_openai()
        client = OpenAI()
        failures: list[tuple[int, str]] = []
        lock = threading.Lock()
        completed_count = 0
        kept_rows = []

        def worker(local_index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            messages = row["analysis"]["normalized_messages"]
            completed_turns = row["analysis"]["completed_tool_turns"]
            result = classify_multihop_dialogue(
                client=client,
                model=args.model,
                row_index=row["analysis"]["source_row_index"],
                row=row,
                messages=messages,
                completed_turns=completed_turns,
                max_retries=args.max_retries,
            )
            return local_index, result

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(worker, local_index, row): local_index
                for local_index, row in enumerate(structural_candidates)
            }
            for future in as_completed(futures):
                local_index = futures[future]
                row = structural_candidates[local_index]
                try:
                    _, result = future.result()
                    row["analysis"]["llm_label"] = result["label"]
                    row["analysis"]["llm_reason"] = result["reason"]
                    if result["label"] == "same_task_multihop":
                        kept_rows.append(row)
                except Exception as exc:
                    failures.append((local_index, str(exc)))
                with lock:
                    completed_count += 1
                    if completed_count % 25 == 0 or completed_count == len(futures):
                        print(
                            f"LLM filtered {completed_count}/{len(futures)} candidates | "
                            f"kept={len(kept_rows)} failed={len(failures)}"
                        )

        kept_rows.sort(key=lambda row: row["analysis"]["source_row_index"])
        if failures:
            print("LLM filtering failures:")
            for local_index, message in failures[:10]:
                print(f"  - candidate {local_index}: {message}")

    examples = extract_examples(kept_rows, args.examples_count)
    summary = build_extract_summary(
        total_rows=len(rows),
        structural_candidates=structural_candidates,
        kept_rows=kept_rows,
        examples=examples,
    )

    write_json(Path(args.output), kept_rows)
    print(f"Wrote extracted dataset to {args.output}")

    if args.summary_output:
        write_json(Path(args.summary_output), summary)
        print(f"Wrote summary to {args.summary_output}")

    if args.examples_output:
        write_json(Path(args.examples_output), examples)
        print(f"Wrote examples to {args.examples_output}")


def run_clean_last_followup(args: argparse.Namespace) -> None:
    rows = load_rows_from_path(Path(args.input))
    if args.limit is not None:
        rows = rows[:args.limit]

    candidate_indices: list[int] = []
    row_messages: list[list[dict[str, Any]]] = []
    last_answer_indices: list[int | None] = []

    for idx, row in enumerate(rows):
        messages = normalize_messages(row)
        row_messages.append(messages)
        answer_index = last_assistant_answer_index(messages)
        last_answer_indices.append(answer_index)
        if answer_index is None:
            continue
        text = messages[answer_index]["content"]
        if args.process_all or looks_like_followup_candidate(text):
            candidate_indices.append(idx)

    print(f"Loaded {len(rows)} extracted rows")
    print(f"Selected {len(candidate_indices)} candidate rows for last-answer cleaning")

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
        messages = row_messages[row_index]
        answer_index = last_answer_indices[row_index]
        if answer_index is None:
            raise ValueError("Missing last assistant answer index")
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
                answer_index = last_answer_indices[row_index]
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
                    rows[row_index]["analysis"]["last_assistant_answer_original"] = row_messages[row_index][answer_index]["content"]
                    rows[row_index]["analysis"]["last_assistant_answer_cleaned"] = result["cleaned_text"]

                    message_key = "conversations" if isinstance(rows[row_index].get("conversations"), list) else "messages"
                    rows[row_index][message_key][row_messages[row_index][answer_index]["original_index"]]["value" if message_key == "conversations" else "content"] = result["cleaned_text"]
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
        }
        write_json(Path(args.summary_output), summary)
        print(f"Wrote cleaning summary to {args.summary_output}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        run_extract(args)
        return

    if args.command == "clean-last-followup":
        run_clean_last_followup(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
