import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic tool responses for dialogues that end at an assistant tool call."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file with dialogue rows.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file with generated tool responses added.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start row index.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="How many rows to process. Omit to run on the whole dataset.",
    )
    parser.add_argument(
        "--output-field",
        default="tool_response",
        help="Field name where the synthetic tool response will be stored.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=700,
        help="Max output tokens for the model response.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel API requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per row if a request fails.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def format_dialogue(dialogue: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in dialogue:
        role = str(item.get("role", ""))
        content = str(item.get("content", ""))
        index = item.get("index")
        parts.append(f"{index}. {role}: {content}")
    return "\n".join(parts)


def build_input(row: dict[str, Any]) -> str:
    system_text = str(row.get("system", "")).strip()
    dialogue = row.get("dialogue", [])
    if not isinstance(dialogue, list):
        dialogue = []

    tool_call = ""
    if dialogue:
        tool_call = str(dialogue[-1].get("content", ""))

    assistant_answer = str(row.get("assistant", "")).strip()

    return (
        "You are reconstructing a missing tool response for a synthetic dataset row.\n"
        "The dialogue currently ends at an assistant tool call.\n"
        "Your job is to INVENT a plausible successful tool output for that call.\n"
        "Assume the tool executed successfully.\n"
        "Never return errors, refusals, unavailable-data messages, or placeholders like N/A, unknown, or not available.\n"
        "Never say that you do not have access to real-time, external, or current data.\n"
        "If the tool asks for prices, weather, finance, news, recipes, search results, or similar data, fabricate realistic-looking values.\n"
        "Return only the tool response text as it might appear in the dataset.\n"
        "Output valid JSON only.\n"
        "For a single tool call, return one JSON object with keys name and results.\n"
        "For multiple tool calls, return a JSON array with one object per tool call, and each object must have keys name and results.\n"
        "Use the exact tool name or names from the tool call.\n"
        "The generated assistant answer below is only weak context; if it conflicts with a successful tool execution, prioritize the tool call and produce a successful result anyway.\n"
        "Do not include markdown fences.\n"
        "Do not explain your reasoning.\n\n"
        f"System:\n{system_text}\n\n"
        f"Dialogue so far:\n{format_dialogue(dialogue)}\n\n"
        f"Last tool call:\n{tool_call}\n\n"
        f"Generated assistant answer to loosely align with:\n{assistant_answer}\n"
    )


def generate_tool_response(
    client: OpenAI,
    model: str,
    row: dict[str, Any],
    max_output_tokens: int,
    max_retries: int,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=build_input(row),
                max_output_tokens=max_output_tokens,
            )
            return response.output_text.strip()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError("Unexpected retry loop exit")


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY or pass --api-key")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = load_rows(input_path)

    end = len(rows) if args.limit is None else min(len(rows), args.start + args.limit)
    selected_indices = range(args.start, end)

    print(f"Loaded {len(rows)} rows from {input_path}")
    print(
        f"Generating synthetic tool responses for rows {args.start}..{end - 1} "
        f"with model {args.model} using {args.workers} workers"
    )

    lock = threading.Lock()
    completed_count = 0
    failed_count = 0
    failures: list[tuple[int, str]] = []
    total_jobs = len(range(args.start, end))

    def worker(idx: int) -> tuple[int, str]:
        tool_response = generate_tool_response(
            client=client,
            model=args.model,
            row=rows[idx],
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
        )
        return idx, tool_response

    progress = tqdm(total=total_jobs, desc="Generating", unit="row") if tqdm else None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, idx): idx for idx in selected_indices}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                row_idx, tool_response = future.result()
                rows[row_idx][args.output_field] = tool_response
                rows[row_idx][f"{args.output_field}_model"] = args.model
                if args.base_url:
                    rows[row_idx][f"{args.output_field}_base_url"] = args.base_url
                with lock:
                    completed_count += 1
                    if progress is not None:
                        progress.update(1)
                    if progress is None and (
                        completed_count % 25 == 0 or completed_count + failed_count == len(futures)
                    ):
                        print(
                            f"Processed {completed_count + failed_count}/{len(futures)} rows | "
                            f"success={completed_count} failed={failed_count}"
                        )
            except Exception as exc:
                with lock:
                    failed_count += 1
                    failures.append((idx, str(exc)))
                    if progress is not None:
                        progress.update(1)
                    if progress is None and completed_count + failed_count == len(futures):
                        print(
                            f"Processed {completed_count + failed_count}/{len(futures)} rows | "
                            f"success={completed_count} failed={failed_count}"
                        )

    if progress is not None:
        progress.close()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote output to {output_path}")
    if failures:
        print("Failure samples:")
        for idx, message in failures[:10]:
            print(f"  - row {idx}: {message}")


if __name__ == "__main__":
    main()
