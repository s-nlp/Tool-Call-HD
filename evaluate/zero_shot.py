"""
Zero-shot hallucination detection on the ToolACE unified test split.

For each row the model must output:
  - type:        clean | answer_mismatch | missing_tool | overgeneration | undergeneration
  - span_labels: list of {start, end, text} character-offset spans in the final assistant answer
  - reasoning:   short explanation

Usage
-----
# Default: loads test split from HuggingFace s-nlp/toolace-unified-hallucinations
python evaluate/zero_shot.py --model gpt-4o

# With explicit HF token (otherwise read from .env HF_TOKEN)
python evaluate/zero_shot.py --model gpt-4o --hf-token hf_...

# Fallback: load from local parquet instead of HF
python evaluate/zero_shot.py --model gpt-4o --input path/to/test.parquet

Environment
-----------
OPENAI_API_KEY  must be set (or placed in .env at project root).
HF_TOKEN        must be set if the dataset is private (or placed in .env).
"""
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
except ImportError:
    tqdm = None

# local
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import ZERO_SHOT_SYSTEM, build_user_message

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HF_DATASET = "s-nlp/toolace-unified-hallucinations"
DEFAULT_HF_SPLIT = "test"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluate" / "results" / "zero_shot_predictions.jsonl"
VALID_TYPES = {"clean", "answer_mismatch", "missing_tool", "overgeneration", "undergeneration"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot hallucination detection eval.")
    p.add_argument(
        "--input", default=None,
        help="Local parquet file to use instead of HuggingFace dataset.",
    )
    p.add_argument(
        "--hf-dataset", default=DEFAULT_HF_DATASET,
        help=f"HuggingFace dataset repo id (default: {DEFAULT_HF_DATASET}).",
    )
    p.add_argument(
        "--hf-split", default=DEFAULT_HF_SPLIT,
        help="Dataset split to evaluate on (default: test).",
    )
    p.add_argument(
        "--hf-token", default=None,
        help="HuggingFace token (falls back to HF_TOKEN in env/.env).",
    )
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--model", default="gpt-4o", help="OpenAI chat model to use.")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Max rows to evaluate.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _to_python(v: Any) -> Any:
    if hasattr(v, "tolist"):
        return _to_python(v.tolist())
    if isinstance(v, list):
        return [_to_python(i) for i in v]
    if isinstance(v, dict):
        return {k: _to_python(vv) for k, vv in v.items()}
    return v


def load_rows_from_hf(repo_id: str, split: str, token: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset  # lazy import
    ds = load_dataset(repo_id, split=split, token=token)
    rows = []
    for row in ds:
        row = dict(row)
        row["conversations"] = _to_python(row.get("conversations", []))
        row["span_labels"] = _to_python(row.get("span_labels", []))
        rows.append(row)
    return rows


def load_rows_from_parquet(path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    rows = []
    for row in df.to_dict(orient="records"):
        row["conversations"] = _to_python(row.get("conversations", []))
        row["span_labels"] = _to_python(row.get("span_labels", []))
        rows.append(row)
    return rows


def load_rows(args: argparse.Namespace, env: dict) -> list[dict[str, Any]]:
    if args.input:
        print(f"[zero_shot] Loading from local parquet: {args.input}")
        return load_rows_from_parquet(Path(args.input))
    hf_token = args.hf_token or os.getenv("HF_TOKEN") or env.get("HF_TOKEN")
    print(f"[zero_shot] Loading from HuggingFace: {args.hf_dataset} / split={args.hf_split}")
    rows = load_rows_from_hf(args.hf_dataset, args.hf_split, hf_token)
    print(f"[zero_shot] Loaded {len(rows)} rows from HF")
    return rows


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict[str, Any]:
    """Balanced-brace scan for the first top-level JSON object. The old
    find("{")..rfind("}") slice breaks whenever the model emits braces in
    surrounding prose or trailing text — do not reintroduce it."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise json.JSONDecodeError("no parseable JSON object found", text, 0)


def _nearest_occurrence(answer: str, text: str, anchor: int) -> int:
    """Start of the occurrence of ``text`` in ``answer`` closest to ``anchor`` (-1 if absent)."""
    best, pos = -1, answer.find(text)
    while pos >= 0:
        if best < 0 or abs(pos - anchor) < abs(best - anchor):
            best = pos
        pos = answer.find(text, pos + 1)
    return best


def _anchor_span(sp: dict[str, Any], final_answer: str, i: int, issues: list[str]) -> bool:
    """Place one predicted span in the answer; return False to drop it.

    The quoted text is the anchor — LLM character offsets are almost always off,
    and overwriting the quote with whatever sits at those offsets (as this step
    used to do) misplaces nearly every span. Offsets are used only when the quote
    is not in the answer. A quote that cannot be placed at all is kept without
    offsets, so scorers count it as a false positive instead of losing it.
    """
    s, e, t = sp.get("start"), sp.get("end"), sp.get("text", "")
    has_offsets = isinstance(s, int) and isinstance(e, int)
    in_range = has_offsets and 0 <= s < e <= len(final_answer)
    if isinstance(t, str) and t:
        if in_range and final_answer[s:e] == t:
            return True
        idx = _nearest_occurrence(final_answer, t, s if has_offsets else 0)
        if idx >= 0:
            issues.append(f"span[{i}] offsets corrected via text search (was [{s},{e}])")
            sp["start"], sp["end"] = idx, idx + len(t)
        elif in_range:
            issues.append(f"span[{i}] text not found in answer, using offsets [{s},{e}]")
            sp["model_text"], sp["text"] = t, final_answer[s:e]
        else:
            issues.append(f"span[{i}] text not found and offsets unusable [{s},{e}]; kept without offsets")
            sp.pop("start", None)
            sp.pop("end", None)
        return True
    if in_range:
        sp["text"] = final_answer[s:e]
        return True
    issues.append(f"span[{i}] has no text and no usable offsets [{s},{e}]; dropped")
    return False


def validate_prediction(pred: dict[str, Any], final_answer: str) -> dict[str, Any]:
    """
    Validate and normalise a model prediction dict.
    Returns the pred dict with an added 'parse_issues' list.
    """
    issues = []

    # type
    ptype = pred.get("type", "")
    if ptype not in VALID_TYPES:
        issues.append(f"unknown type '{ptype}'")
        pred["type"] = "unknown"

    # span_labels
    spans = pred.get("span_labels", [])
    if not isinstance(spans, list):
        issues.append("span_labels is not a list")
        pred["span_labels"] = []
        spans = []

    # For undergeneration / clean: spans must be empty
    if pred.get("type") in ("clean", "undergeneration") and spans:
        issues.append(f"type={pred['type']} but non-empty span_labels — clearing spans")
        pred["span_labels"] = []
        spans = []

    # Verify each span
    validated_spans = []
    for i, sp in enumerate(spans):
        if not isinstance(sp, dict):
            issues.append(f"span[{i}] is not a dict")
            continue
        if _anchor_span(sp, final_answer, i, issues):
            validated_spans.append(sp)

    pred["span_labels"] = validated_spans
    pred["parse_issues"] = issues
    return pred


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    max_retries: int,
) -> str:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_err = str(exc)
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(last_err)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def process_row(
    client: OpenAI,
    model: str,
    row: dict[str, Any],
    row_index: int,
    max_tokens: int,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    # Get the final answer text for span validation
    final_answer = ""
    for msg in reversed(row.get("conversations", [])):
        if msg.get("from") == "assistant":
            final_answer = msg.get("value", "")
            break

    messages = [
        {"role": "system", "content": ZERO_SHOT_SYSTEM},
        {"role": "user", "content": build_user_message(row)},
    ]

    raw_text = call_model(client, model, messages, max_tokens, temperature, max_retries)

    try:
        pred = extract_json(raw_text)
        pred = validate_prediction(pred, final_answer)
        status = "ok"
    except Exception as exc:
        pred = {}
        status = f"parse_error: {exc}"

    return {
        "row_index": row_index,
        "dialogue_id": row.get("dialogue_id"),
        "gold_type": row.get("type"),
        "gold_label": row.get("label"),
        "gold_spans": row.get("span_labels", []),
        "final_answer": final_answer,
        "pred_type": pred.get("type"),
        "pred_span_labels": pred.get("span_labels", []),
        "pred_reasoning": pred.get("reasoning", ""),
        "pred_parse_issues": pred.get("parse_issues", []),
        "raw_response": raw_text,
        "status": status,
        "model": model,
        "mode": "zero_shot",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    env = dotenv_values(PROJECT_ROOT / ".env")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or env.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY in env or .env file.")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    rows = load_rows(args, env)
    end = len(rows) if args.limit is None else min(len(rows), args.start + args.limit)
    indices = list(range(args.start, end))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[zero_shot] model={args.model}  rows={len(indices)}  workers={args.workers}")

    lock = threading.Lock()
    results: list[dict[str, Any]] = []
    bar = tqdm(total=len(indices), desc="zero-shot", unit="row") if tqdm else None

    def worker(idx: int) -> dict[str, Any]:
        return process_row(
            client=client,
            model=args.model,
            row=rows[idx],
            row_index=idx,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, i): i for i in indices}
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as exc:
                idx = futures[fut]
                row = rows[idx]
                res = {
                    "row_index": idx,
                    "dialogue_id": row.get("dialogue_id"),
                    "gold_type": row.get("type"),
                    "gold_label": row.get("label"),
                    "status": f"error: {exc}",
                    "model": args.model,
                    "mode": "zero_shot",
                }
            with lock:
                results.append(res)
                if bar:
                    bar.update(1)
                elif len(results) % 50 == 0:
                    print(f"  {len(results)}/{len(indices)} done")

    if bar:
        bar.close()

    results.sort(key=lambda r: r["row_index"])

    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quick stats
    ok = [r for r in results if r.get("status") == "ok"]
    errors = len(results) - len(ok)
    type_correct = sum(1 for r in ok if r["pred_type"] == r["gold_type"])
    print(f"\n[zero_shot] Done. ok={len(ok)}  errors={errors}")
    print(f"  Type accuracy: {type_correct}/{len(ok)} = {type_correct/len(ok):.3f}" if ok else "  No successful rows.")
    print(f"  Results: {output_path}")


if __name__ == "__main__":
    main()
