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
        description="Generate answers for dataset rows using an OpenAI-compatible API."
    )
    parser.add_argument(
        "--input",
        default="dataset_v3_tagged.json",
        help="Input dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file with generated answers added.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name, for example gpt-4.1-mini.",
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
        default="generated_answer",
        help="Field name where the model answer will be stored.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
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


def build_input(row: dict[str, Any]) -> str:
    return (
        "You are given a user request, the tool call selected by an assistant, and the tool result.\n"
        "Write the final assistant answer for the user.\n"
        "Use the tool response as the source of truth.\n"
        "Do not mention hidden reasoning.\n"
        "Do not mention function names unless they are user-facing.\n"
        "Return only the answer text.\n\n"
        f"User prompt:\n{row.get('user_prompt', '')}\n\n"
        f"Tool call:\n{row.get('tool_call', '')}\n\n"
        f"Tool response:\n{row.get('tool_response', '')}\n"
    )


def generate_answer(
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
        f"Generating answers for rows {args.start}..{end - 1} "
        f"with model {args.model} using {args.workers} workers"
    )

    lock = threading.Lock()
    completed_count = 0
    failed_count = 0
    failures: list[tuple[int, str]] = []
    total_jobs = len(range(args.start, end))

    def worker(idx: int) -> tuple[int, str]:
        answer = generate_answer(
            client=client,
            model=args.model,
            row=rows[idx],
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
        )
        return idx, answer

    progress = tqdm(total=total_jobs, desc="Generating", unit="row") if tqdm else None

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, idx): idx for idx in selected_indices}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                row_idx, answer = future.result()
                rows[row_idx][args.output_field] = answer
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
