"""
LLM-judge for verifying synthetic error annotations.

Takes rows from the synthetic error dataset (HuggingFace dataset OR the small
CSV) and asks a judge model whether each row's annotation is correct:
  - is the `type` field right?
  - are the `span_labels` right (correct text spans, no missing spans, no
    spurious spans)?
  - is the binary `label` (hallucination present yes/no) right?

The judge writes one JSON object per row to a JSONL file. The schema is
fixed (see JUDGE_SCHEMA below) so downstream analysis can aggregate cleanly.

Designed to be provider-agnostic via OpenAI-compatible endpoints — runs
against OpenRouter for hosted models or a local vLLM server for self-hosted
ones. Concurrency is bounded by asyncio.Semaphore.

Usage
-----
    # OpenRouter (hosted)
    export OPENROUTER_API_KEY=sk-or-...
    python judge_verify_annotations.py \\
        --csv small_test_for_verifiaction.csv \\
        --model qwen/qwen-3.6-plus \\
        --base-url https://openrouter.ai/api/v1 \\
        --output judgments.jsonl \\
        --concurrency 8

    # Local vLLM
    python judge_verify_annotations.py \\
        --hf-dataset your-org/synthetic-errors \\
        --model Qwen/Qwen2.5-14B-Instruct \\
        --base-url http://localhost:8000/v1 \\
        --api-key EMPTY \\
        --output judgments.jsonl

    # Quick smoke test on the first 5 rows
    python judge_verify_annotations.py --csv ... --limit 5
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, APIError

# ---------------------------------------------------------------------------
# Class definitions (kept verbatim from the project spec so the judge prompt
# is the single source of truth).
# ---------------------------------------------------------------------------

CLASS_DEFINITIONS = """\
- "answer_mismatch": The final assistant answer states information that
  directly contradicts or substitutes values from the tool response (e.g.
  wrong company name, wrong number, wrong entity). The tool response IS
  present and the assistant misread or swapped data from it. Span labels
  should point to the exact contradicted text.

- "missing_tool": The final assistant answer includes content that is
  unsolicited or fabricated beyond the tool output scope. This covers two
  sub-patterns:
    (a) Baseless Info — the answer reports concrete facts (IDs, amounts,
        names) that are not present or cannot be derived from the tool
        response;
    (b) Proactive follow-up — the answer appends an unprompted offer or
        question ("Would you like me to...") not grounded in any tool
        result or user request.
  Note: a tool response IS always present in the dialogue for this type.
  Do NOT interpret "missing_tool" as "no tool was called."

- "overgeneration": The final assistant answer adds unsupported
  elaboration, commentary, interpretation, or claims that go beyond what
  any tool response returned, even if not directly contradicting it.

- "undergeneration": The final assistant answer OMITS information that the
  tool response returned. The hallucination is an absence, not a
  fabrication. Therefore span_labels will be EMPTY ([]) for all
  undergeneration rows — this is correct by design. When type=
  "undergeneration" and span_labels=[], set span_labels_are_correct=true.

- "clean": No hallucination. The assistant answer faithfully reflects the
  tool response. span_labels must be []."""

UNDERGENERATION_RULE = """\
CRITICAL RULE — undergeneration:
If the row's type is "undergeneration", an EMPTY span_labels list ([]) is
the EXPECTED and CORRECT annotation. You MUST set
span_labels_are_correct=true for undergeneration rows with empty
span_labels. Do not treat the absence of spans as a labeling error for
this class. Suggested_span_labels MUST also be [] for undergeneration."""

JUDGE_SCHEMA = """\
{
  "row_judgment": "pass" | "fail" | "uncertain",
  "hallucination_present": true | false,
  "type_is_correct": true | false,
  "span_labels_are_correct": true | false,
  "reasoning_short": "brief explanation (<= 2 sentences)",
  "recommended_type": "answer_mismatch" | "missing_tool" | "overgeneration" | "undergeneration" | "clean" | "unknown",
  "missing_hallucination_spans": [
    {"text": "hallucinated text missing from labels", "reason": "why it should be labeled"}
  ],
  "incorrect_span_labels": [
    {"label_text": "text currently labeled", "issue": "not hallucination | too broad | too narrow | wrong target | other"}
  ],
  "suggested_span_labels": [
    {"text": "suggested text span", "reason": "why this text should be labeled"}
  ]
}"""

JUDGE_SEMANTICS = """\
row_judgment semantics:
  - "pass":      type_is_correct AND span_labels_are_correct AND the binary
                 hallucination_present matches the row's label.
  - "fail":      any of the three above is wrong.
  - "uncertain": you genuinely cannot decide from the dialogue (e.g.
                 ambiguous tool response, missing context). Prefer "fail"
                 over "uncertain" if you can articulate a concrete error."""

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class JudgeConfig:
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 1024
    concurrency: int = 8
    request_timeout: float = 120.0
    max_retries: int = 3
    # Some providers don't support response_format={"type":"json_object"}.
    # If False, we rely on prompt + regex extraction.
    use_response_format_json: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _parse_maybe_literal(value: Any) -> Any:
    """span_labels / conversations are stringified Python literals in the CSV
    but native lists when loaded from a HF dataset. Normalise both."""
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Try JSON as a fallback (some pipelines export as JSON strings)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value


def load_rows_from_csv(path: Path, limit: int | None = None) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "dialogue_id": r["dialogue_id"],
                "system": r["system"],
                "conversations": _parse_maybe_literal(r["conversations"]),
                "span_labels": _parse_maybe_literal(r["span_labels"]),
                "label": int(r["label"]),
                "type": r["type"],
                "subset": r.get("subset", ""),
            }
        )
    return rows


def load_rows_from_hf(
    name: str, split: str = "train", limit: int | None = None
) -> list[dict]:
    from datasets import load_dataset, load_from_disk

    # Local save_to_disk directories (Dataset or DatasetDict) must go through
    # load_from_disk — load_dataset() cannot read them.
    p = Path(name)
    if p.is_dir() and ((p / "dataset_info.json").exists()
                       or (p / "dataset_dict.json").exists()):
        ds = load_from_disk(str(p))
        if hasattr(ds, "keys") and not hasattr(ds, "column_names"):  # DatasetDict
            ds = ds[split]
    else:
        ds = load_dataset(name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = []
    for r in ds:
        rows.append(
            {
                "dialogue_id": r["dialogue_id"],
                "system": r["system"],
                "conversations": _parse_maybe_literal(r["conversations"]),
                "span_labels": _parse_maybe_literal(r["span_labels"]),
                "label": int(r["label"]),
                "type": r["type"],
                "subset": r.get("subset", ""),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def format_conversation(conv: list[dict]) -> str:
    """Render the conversations list as a readable transcript for the judge."""
    out = []
    for turn in conv:
        role = turn.get("from", "?").upper()
        value = turn.get("value", "")
        out.append(f"[{role}]\n{value}")
    return "\n\n".join(out)


def format_span_labels(spans: list[dict]) -> str:
    if not spans:
        return "[] (no span labels)"
    lines = []
    for i, s in enumerate(spans):
        text = s.get("text", "")
        start = s.get("start")
        end = s.get("end")
        label_type = s.get("label_type") or "—"
        meta = s.get("meta") or "—"
        lines.append(
            f"  {i + 1}. text={text!r}\n"
            f"     offsets=[{start}, {end})  label_type={label_type}  meta={meta!r}"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = f"""\
You are a strict quality-control judge for a synthetic error-annotation
dataset. Each row contains a tool-using dialogue and an annotation that
labels whether the final assistant turn contains an error, what type of
error, and which spans of text are the error.

Your job: decide whether the annotation is correct.

ERROR CLASS DEFINITIONS
{CLASS_DEFINITIONS}

{UNDERGENERATION_RULE}

{JUDGE_SEMANTICS}

OUTPUT FORMAT
Return JSON only — no prose, no markdown fences. The object must have
exactly these fields and no others:
{JUDGE_SCHEMA}

Field rules:
  - hallucination_present must match what you actually see in the
    assistant's final turn, regardless of the row's stored label.
  - type_is_correct compares the row's stored `type` against the true
    error class.
  - For undergeneration with empty span_labels, span_labels_are_correct is
    ALWAYS true (see CRITICAL RULE above).
  - suggested_span_labels should be your proposed corrected set of spans
    (use the EXACT verbatim substring as it appears in the assistant's
    final turn). For undergeneration this list must be [].
  - missing_hallucination_spans lists hallucinated text that the
    annotators failed to mark.
  - incorrect_span_labels lists currently-labeled spans that you believe
    are wrong (with a short issue tag).
  - reasoning_short: at most 2 sentences."""


USER_PROMPT_TEMPLATE = """\
=== ROW ===
dialogue_id: {dialogue_id}
stored label (1=hallucination, 0=clean): {label}
stored type: {type}

=== SYSTEM PROMPT GIVEN TO THE ASSISTANT ===
{system}

=== FULL CONVERSATION ===
{conversation}

=== CURRENT SPAN LABELS ===
{span_labels}

=== TASK ===
Judge whether the annotation is correct and return the required JSON object."""


def build_user_prompt(row: dict) -> str:
    return USER_PROMPT_TEMPLATE.format(
        dialogue_id=row["dialogue_id"],
        label=row["label"],
        type=row["type"],
        system=row["system"],
        conversation=format_conversation(row["conversations"]),
        span_labels=format_span_labels(row["span_labels"]),
    )


# ---------------------------------------------------------------------------
# JSON extraction & schema validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "row_judgment",
    "hallucination_present",
    "type_is_correct",
    "span_labels_are_correct",
    "reasoning_short",
    "recommended_type",
    "missing_hallucination_spans",
    "incorrect_span_labels",
    "suggested_span_labels",
}

_ROW_JUDGMENT_VALUES = {"pass", "fail", "uncertain"}
_RECOMMENDED_TYPE_VALUES = {
    "answer_mismatch",
    "answer_missmatch",  # legacy misspelling — normalized below
    "missing_tool",
    "overgeneration",
    "undergeneration",
    "clean",
    "unknown",
}


def extract_json(text: str) -> dict:
    """Parse JSON from the model output. Falls back to extracting the first
    {...} block if the model wrapped it in prose or fences."""
    text = text.strip()
    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Best-effort: greedy match the first top-level object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    return json.loads(match.group(0))


def validate_judgment(j: dict, row: dict) -> tuple[dict, list[str]]:
    """Return (normalized_judgment, warnings). Warnings flag soft problems
    that don't invalidate the response but should be tracked."""
    warnings: list[str] = []

    missing = _REQUIRED_KEYS - j.keys()
    if missing:
        raise ValueError(f"Judgment missing required keys: {sorted(missing)}")

    extra = set(j.keys()) - _REQUIRED_KEYS
    if extra:
        warnings.append(f"extra keys ignored: {sorted(extra)}")
        for k in extra:
            j.pop(k, None)

    if j["row_judgment"] not in _ROW_JUDGMENT_VALUES:
        raise ValueError(f"Bad row_judgment: {j['row_judgment']!r}")
    if j["recommended_type"] not in _RECOMMENDED_TYPE_VALUES:
        raise ValueError(f"Bad recommended_type: {j['recommended_type']!r}")
    if j["recommended_type"] == "answer_missmatch":
        j["recommended_type"] = "answer_mismatch"
    for boolfield in (
        "hallucination_present",
        "type_is_correct",
        "span_labels_are_correct",
    ):
        if not isinstance(j[boolfield], bool):
            raise ValueError(f"Field {boolfield} must be bool, got {j[boolfield]!r}")

    # Enforce the undergeneration rule defensively, in case the model
    # forgets despite the prompt.
    if row["type"] == "undergeneration" and not row["span_labels"]:
        if not j["span_labels_are_correct"]:
            warnings.append(
                "undergeneration rule auto-corrected: span_labels_are_correct -> true"
            )
            j["span_labels_are_correct"] = True
        if j["suggested_span_labels"]:
            warnings.append(
                "undergeneration rule auto-corrected: suggested_span_labels -> []"
            )
            j["suggested_span_labels"] = []

    return j, warnings


# ---------------------------------------------------------------------------
# Async judge call
# ---------------------------------------------------------------------------


async def judge_one(
    client: AsyncOpenAI,
    cfg: JudgeConfig,
    row: dict,
    sem: asyncio.Semaphore,
) -> dict:
    user_prompt = build_user_prompt(row)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "timeout": cfg.request_timeout,
    }
    if cfg.use_response_format_json:
        kwargs["response_format"] = {"type": "json_object"}
    if cfg.extra_headers:
        kwargs["extra_headers"] = cfg.extra_headers

    last_err: Exception | None = None
    async with sem:
        for attempt in range(1, cfg.max_retries + 1):
            try:
                resp = await client.chat.completions.create(**kwargs)
                # OpenAI SDK auto-isolates reasoning into .reasoning for
                # models that emit it (Qwen think-tags, etc.); .content
                # stays clean of <think> blocks.
                content = resp.choices[0].message.content or ""
                judgment = extract_json(content)
                judgment, warnings = validate_judgment(judgment, row)
                return {
                    "dialogue_id": row["dialogue_id"],
                    "stored_type": row["type"],
                    "stored_label": row["label"],
                    "judgment": judgment,
                    "warnings": warnings,
                    "error": None,
                }
            except (APIError, ValueError, json.JSONDecodeError) as e:
                last_err = e
                if attempt < cfg.max_retries:
                    await asyncio.sleep(min(2**attempt, 8))
                continue

    return {
        "dialogue_id": row["dialogue_id"],
        "stored_type": row["type"],
        "stored_label": row["label"],
        "judgment": None,
        "warnings": [],
        "error": f"{type(last_err).__name__}: {last_err}",
    }


def load_done_keys(output_path: Path) -> set[tuple[str, str]]:
    """(dialogue_id, stored_type) pairs already judged successfully."""
    done: set[tuple[str, str]] = set()
    if not output_path.exists():
        return done
    with output_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None and r.get("judgment") is not None:
                done.add((str(r.get("dialogue_id")), str(r.get("stored_type"))))
    return done


async def run_judge(
    rows: list[dict], cfg: JudgeConfig, output_path: Path, resume: bool = False
) -> list[dict]:
    client = AsyncOpenAI(
        api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.request_timeout
    )
    sem = asyncio.Semaphore(cfg.concurrency)

    results: list[dict] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "w"
    if resume:
        done = load_done_keys(output_path)
        before = len(rows)
        rows = [r for r in rows
                if (str(r["dialogue_id"]), str(r["type"])) not in done]
        print(f"Resume: {before - len(rows)} rows already judged, "
              f"{len(rows)} remaining")
        if not rows:
            return []
        mode = "a"

    # Write progressively so we don't lose progress on a crash.
    with output_path.open(mode) as fh:
        tasks = [asyncio.create_task(judge_one(client, cfg, r, sem)) for r in rows]
        for i, fut in enumerate(asyncio.as_completed(tasks), start=1):
            result = await fut
            results.append(result)
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            status = "ok" if result["error"] is None else "ERR"
            sys.stdout.write(
                f"\r[{i}/{len(rows)}] {status}  dialogue_id={result['dialogue_id']}     "
            )
            sys.stdout.flush()
    sys.stdout.write("\n")
    await client.close()
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: list[dict]) -> None:
    n = len(results)
    n_err = sum(1 for r in results if r["error"])
    n_ok = n - n_err
    if n_ok == 0:
        print(f"\nAll {n} rows errored.")
        return

    judgments = [r for r in results if r["judgment"] is not None]
    by_row = {"pass": 0, "fail": 0, "uncertain": 0}
    type_correct = 0
    spans_correct = 0
    hallu_present_match = 0

    for r in judgments:
        j = r["judgment"]
        by_row[j["row_judgment"]] += 1
        if j["type_is_correct"]:
            type_correct += 1
        if j["span_labels_are_correct"]:
            spans_correct += 1
        stored_has_hallu = r["stored_label"] == 1
        if j["hallucination_present"] == stored_has_hallu:
            hallu_present_match += 1

    print("\n=== JUDGE SUMMARY ===")
    print(f"rows total:     {n}")
    print(f"rows judged ok: {n_ok}")
    print(f"rows errored:   {n_err}")
    print()
    print(f"row_judgment   pass: {by_row['pass']:>4} ({by_row['pass']/n_ok:.1%})")
    print(f"row_judgment   fail: {by_row['fail']:>4} ({by_row['fail']/n_ok:.1%})")
    print(
        f"row_judgment   uncert: {by_row['uncertain']:>3} "
        f"({by_row['uncertain']/n_ok:.1%})"
    )
    print()
    print(f"type_is_correct:          {type_correct}/{n_ok} ({type_correct/n_ok:.1%})")
    print(f"span_labels_are_correct:  {spans_correct}/{n_ok} ({spans_correct/n_ok:.1%})")
    print(
        f"hallucination_present matches stored label: "
        f"{hallu_present_match}/{n_ok} ({hallu_present_match/n_ok:.1%})"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="Path to a CSV like small_test_for_verifiaction.csv")
    src.add_argument("--hf-dataset", type=str, help="HuggingFace dataset name")

    p.add_argument("--hf-split", default="train", help="HF split (default: train)")
    p.add_argument("--limit", type=int, default=None, help="Only judge the first N rows")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("judgments.jsonl"),
        help="Output JSONL path (default: judgments.jsonl)",
    )
    p.add_argument(
        "--model",
        default="qwen/qwen-3.6-plus",
        help="Model name (provider-prefixed for OpenRouter, e.g. qwen/qwen-3.6-plus)",
    )
    p.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible endpoint",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key; defaults to $OPENROUTER_API_KEY or $OPENAI_API_KEY",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip (dialogue_id, type) pairs already judged successfully in "
             "--output; append new verdicts instead of overwriting",
    )
    p.add_argument(
        "--no-json-response-format",
        action="store_true",
        help="Disable response_format={'type':'json_object'} (use for models that don't support it)",
    )

    args = p.parse_args()

    api_key = (
        args.api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"  # vLLM local servers accept any string
    )

    cfg = JudgeConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        use_response_format_json=not args.no_json_response_format,
    )

    if args.csv:
        rows = load_rows_from_csv(args.csv, limit=args.limit)
    else:
        rows = load_rows_from_hf(args.hf_dataset, split=args.hf_split, limit=args.limit)

    print(f"Loaded {len(rows)} rows. Judging with {cfg.model} @ {cfg.base_url}")
    print(f"Writing judgments to {args.output}")

    results = asyncio.run(run_judge(rows, cfg, args.output, resume=args.resume))
    print_summary(results)


if __name__ == "__main__":
    main()
