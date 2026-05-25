from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "artifacts" / "toolace_unified_for_hf" / "toolace_unified_records_test.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluate" / "outputs" / "gpt52_test_split_audit.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "evaluate" / "outputs" / "gpt52_test_split_audit_summary.json"
PROMPTS_DIR = PROJECT_ROOT / "evaluate" / "prompts"
DEFAULT_CLEAN_PROMPT = PROMPTS_DIR / "judge_clean_system_prompt.txt"
DEFAULT_HALL_PROMPT = PROMPTS_DIR / "judge_hallucination_system_prompt.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the ToolACE unified test split with gpt-5.2."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input parquet file, default: test split parquet.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSONL with one audit result per row.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY),
        help="Output JSON summary file.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="Judge model name.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key. Defaults to OPENAI_API_KEY or .env OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--clean-prompt",
        default=str(DEFAULT_CLEAN_PROMPT),
        help="Path to clean-row judge system prompt.",
    )
    parser.add_argument(
        "--hall-prompt",
        default=str(DEFAULT_HALL_PROMPT),
        help="Path to hallucination-row judge system prompt.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start row index in the parquet dataframe.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="How many rows to process. Omit to process all rows from start.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel API requests.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1400,
        help="Max output tokens for each judge response.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retries per row if the API call or JSON parsing fails.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort for the judge model.",
    )
    return parser.parse_args()


def to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_python(value.tolist())
    if isinstance(value, list):
        return [to_python(item) for item in value]
    if isinstance(value, dict):
        return {key: to_python(item) for key, item in value.items()}
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["conversations"] = to_python(row.get("conversations", []))
                row["span_labels"] = to_python(row.get("span_labels", []))
                rows.append(row)
        return rows
    df = pd.read_parquet(path)
    rows = []
    for row in df.to_dict(orient="records"):
        row["conversations"] = to_python(row.get("conversations", []))
        row["span_labels"] = to_python(row.get("span_labels", []))
        rows.append(row)
    return rows


def find_last_assistant_answer(conversations: list[dict[str, Any]]) -> str:
    for message in reversed(conversations):
        if message.get("from") == "assistant":
            return message.get("value", "")
    return ""


def find_last_tool_response(conversations: list[dict[str, Any]]) -> str:
    for message in reversed(conversations):
        if message.get("from") == "tool":
            return message.get("value", "")
    return ""


def format_row_payload(row: dict[str, Any]) -> str:
    final_answer = find_last_assistant_answer(row.get("conversations", []))
    last_tool_response = find_last_tool_response(row.get("conversations", []))
    payload = {
        "dialogue_id": row.get("dialogue_id"),
        "label": row.get("label"),
        "type": row.get("type"),
        "subset": row.get("subset"),
        "system_prompt": row.get("system", ""),
        "conversations": row.get("conversations", []),
        "final_assistant_answer": final_answer,
        "last_tool_response": last_tool_response,
        "span_labels": row.get("span_labels", []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def build_messages(row: dict[str, Any], clean_prompt: str, hall_prompt: str) -> list[dict[str, str]]:
    is_clean = row.get("label") == 0 or row.get("type") == "clean"
    system_prompt = clean_prompt if is_clean else hall_prompt
    row_type = row.get("type", "unknown")
    row_label = row.get("label", "unknown")
    user_prompt = (
        f"Row type: {row_type}\n"
        f"Row label: {row_label}\n"
        "Audit the following dataset row.\n"
        "Return JSON only.\n\n"
        f"{format_row_payload(row)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def judge_row(
    client: OpenAI,
    model: str,
    row: dict[str, Any],
    clean_prompt: str,
    hall_prompt: str,
    max_output_tokens: int,
    max_retries: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=build_messages(row, clean_prompt, hall_prompt),
                max_output_tokens=max_output_tokens,
                reasoning={"effort": reasoning_effort},
            )
            parsed = extract_json(response.output_text)
            return {
                "judge_response": parsed,
                "judge_raw_text": response.output_text,
                "judge_model": model,
            }
        except Exception as exc:  # pragma: no cover
            last_error = str(exc)
            if attempt == max_retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(last_error or "Unknown judge failure")


def build_triage_df(audit_rows: list[dict[str, Any]]) -> "pd.DataFrame":
    """
    Build a flat triage DataFrame with one row per audited sample.

    verdict column values:
    - "verified_correct"  — judge said row_judgment="pass"  (annotation looks right)
    - "verified_wrong"    — judge said row_judgment="fail"  (annotation is wrong)
    - "uncertain"         — judge said row_judgment="uncertain" or response missing
    - "error"             — API / parse failure, no judge response
    """
    records = []
    for row in audit_rows:
        jr = row.get("judge_response", {}) or {}
        judgment = jr.get("row_judgment")

        if row.get("status") != "ok":
            verdict = "error"
        elif judgment == "pass":
            verdict = "verified_correct"
        elif judgment == "fail":
            verdict = "verified_wrong"
        else:
            verdict = "uncertain"

        record: dict[str, Any] = {
            "row_index": row.get("row_index"),
            "dialogue_id": row.get("dialogue_id"),
            "label": row.get("label"),
            "type": row.get("type"),
            "subset": row.get("subset"),
            "verdict": verdict,
            "judge_judgment": judgment,
            "judge_reasoning": jr.get("reasoning_short"),
            # clean-prompt specific fields
            "contains_hallucination": jr.get("contains_hallucination"),
            "recommended_label": jr.get("recommended_label"),
            # hall-prompt specific fields
            "hallucination_present": jr.get("hallucination_present"),
            "type_is_correct": jr.get("type_is_correct"),
            "span_labels_are_correct": jr.get("span_labels_are_correct"),
            # shared
            "recommended_type": jr.get("recommended_type"),
            "error": row.get("error"),
        }
        records.append(record)
    return pd.DataFrame(records)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    success = [row for row in results if row.get("status") == "ok"]
    failed = [row for row in results if row.get("status") != "ok"]

    def count_by(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in success:
            value = row.get("judge_response", {}).get(key, "missing")
            counts[str(value)] = counts.get(str(value), 0) + 1
        return counts

    return {
        "total_rows": total,
        "successful_rows": len(success),
        "failed_rows": len(failed),
        "row_judgment_counts": count_by("row_judgment"),
        "recommended_label_counts": count_by("recommended_label"),
        "recommended_type_counts": count_by("recommended_type"),
        "contains_hallucination_counts": count_by("contains_hallucination"),
        "hallucination_present_counts": count_by("hallucination_present"),
        "span_labels_are_correct_counts": count_by("span_labels_are_correct"),
        "type_is_correct_counts": count_by("type_is_correct"),
        "failures": [
            {
                "row_index": row.get("row_index"),
                "dialogue_id": row.get("dialogue_id"),
                "error": row.get("error"),
            }
            for row in failed[:50]
        ],
    }


def main() -> None:
    args = parse_args()

    env = dotenv_values(PROJECT_ROOT / ".env")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or env.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY, put it in .env, or pass --api-key")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    clean_prompt = Path(args.clean_prompt).read_text(encoding="utf-8")
    hall_prompt = Path(args.hall_prompt).read_text(encoding="utf-8")
    rows = load_rows(Path(args.input))

    end = len(rows) if args.limit is None else min(len(rows), args.start + args.limit)
    selected_indices = list(range(args.start, end))

    print(f"Loaded {len(rows)} rows from {args.input}")
    print(
        f"Auditing rows {args.start}..{end - 1} with {args.model} "
        f"using {args.workers} workers"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: load already-processed row indices from existing output
    already_done: set[int] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    already_done.add(rec["row_index"])
                except Exception:
                    pass
        if already_done:
            print(f"Resuming: skipping {len(already_done)} already-processed rows")

    selected_indices = [i for i in selected_indices if i not in already_done]

    lock = threading.Lock()
    audit_rows: list[dict[str, Any]] = []
    progress = tqdm(total=len(selected_indices), desc="Auditing", unit="row") if tqdm else None

    def worker(idx: int) -> dict[str, Any]:
        row = rows[idx]
        result = {
            "row_index": idx,
            "dialogue_id": row.get("dialogue_id"),
            "label": row.get("label"),
            "type": row.get("type"),
            "subset": row.get("subset"),
        }
        judged = judge_row(
            client=client,
            model=args.model,
            row=row,
            clean_prompt=clean_prompt,
            hall_prompt=hall_prompt,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
            reasoning_effort=args.reasoning_effort,
        )
        result.update(judged)
        result["status"] = "ok"
        return result

    # Open output file in append mode so each result is flushed immediately
    with output_path.open("a", encoding="utf-8") as out_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(worker, idx): idx for idx in selected_indices}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover
                    row = rows[idx]
                    result = {
                        "row_index": idx,
                        "dialogue_id": row.get("dialogue_id"),
                        "label": row.get("label"),
                        "type": row.get("type"),
                        "subset": row.get("subset"),
                        "status": "error",
                        "error": str(exc),
                    }
                with lock:
                    audit_rows.append(result)
                    out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_handle.flush()
                    if progress is not None:
                        progress.update(1)
                    elif len(audit_rows) % 25 == 0 or len(audit_rows) == len(selected_indices):
                        print(f"Processed {len(audit_rows)}/{len(selected_indices)} rows")

    if progress is not None:
        progress.close()

    # For triage/summary, merge newly processed rows with already-done rows from file
    all_audit_rows: list[dict[str, Any]] = []
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                all_audit_rows.append(json.loads(line))
            except Exception:
                pass
    all_audit_rows.sort(key=lambda r: r["row_index"])

    # --- triage DataFrame ---
    triage_df = build_triage_df(all_audit_rows)
    triage_parquet = output_path.with_name(output_path.stem + "_triage.parquet")
    triage_csv = output_path.with_name(output_path.stem + "_triage.csv")
    triage_df.to_parquet(triage_parquet, index=False)
    triage_df.to_csv(triage_csv, index=False)

    verdict_counts = triage_df["verdict"].value_counts().to_dict()
    print(f"\nTriage verdict counts: {verdict_counts}")
    print(f"Wrote triage parquet to {triage_parquet}")
    print(f"Wrote triage CSV    to {triage_csv}")

    summary = summarize(all_audit_rows)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote audit rows to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
