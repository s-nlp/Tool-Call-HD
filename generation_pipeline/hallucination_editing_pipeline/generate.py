"""
Singlehop generation entry point.

Supports seven generation types:

  undergeneration_legacy    — flat leaf deletion (no LLM)
  undergeneration           — cascade deletion (no LLM, cleaner spans) [preferred]
  answer_mismatch_legacy    — schema-based hallucination via vLLM guided decoding
  answer_mismatch_gated     — schema-based hallucination + undergeneration span matching
                              (only generates for rows that passed undergeneration;
                              needs undergeneration output first)
  answer_mismatch_targeted  — span-targeted schema hallucination, unlocks only
                              answer-quoted leaves (preferred over answer_mismatch_gated
                              — higher yield; needs undergeneration first)
  overgeneration_legacy     — tool-constrained overgeneration via vLLM
  overgeneration            — overgeneration with filler filtering + comment
                              validation (cleaner) [preferred]

By default the async batched path is used for the LLM-driven types
(answer_mismatch_legacy/gated/targeted, overgeneration_legacy/overgeneration).
Pass --sync to fall back to the slow one-row-at-a-time versions
(answer_mismatch_legacy, overgeneration_legacy only).

Usage:
    python3 generate.py                                                    # async, default types
    python3 generate.py --types undergeneration overgeneration             # skip answer-mismatch
    python3 generate.py --types answer_mismatch_legacy undergeneration_legacy overgeneration_legacy  # legacy types
    python3 generate.py --types undergeneration answer_mismatch_gated      # undergeneration then gated answer-mismatch
    python3 generate.py --sync --types answer_mismatch_legacy overgeneration_legacy  # sync path (legacy)
    python3 generate.py --fresh                                            # delete existing outputs first
"""
import json
import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI, AsyncOpenAI
import httpx
from hallucination_auto import HallucinationAuto

# Word names are the canonical CLI interface. Each maps to the internal
# numeric type key used throughout this file and the rest of the pipeline.
#   answer_mismatch_legacy / _gated / _targeted → Type 1 / 1.1 / 1.2
#   undergeneration_legacy / undergeneration     → Type 2 / 2.1 (preferred)
#   overgeneration_legacy / overgeneration       → Type 3 / 3.1 (preferred)
WORD_ALIASES = {
    "answer_mismatch_legacy":    "1",
    "answer_mismatch_gated":     "1.1",
    "answer_mismatch_targeted":  "1.2",
    "undergeneration_legacy":    "2",
    "undergeneration":           "2.1",
    "overgeneration_legacy":     "3",
    "overgeneration":            "3.1",
}
VALID_TYPES = list(WORD_ALIASES)
DEFAULT_TYPES = ["undergeneration", "answer_mismatch_gated", "overgeneration"]
NUMERIC_TO_WORD = {v: k for k, v in WORD_ALIASES.items()}  # for display only

# Maps type key → output filename (no extension)
OUTPUT_NAMES = {
    "1":   "type1_output",
    "2":   "type2_output",
    "2.1": "type2_1_output",
    "3":   "type3_output",
    "1.1": "type1_1_output",
    "1.2": "type1_2_output",
    "3.1": "type3_1_output",
}


def _type_arg(value: str) -> str:
    """Normalize a --types entry to its internal numeric key.

    Accepts either the word name (e.g. "undergeneration") or, for backward
    compatibility, the legacy numeric key (e.g. "2.1") directly.
    """
    if value in WORD_ALIASES:
        return WORD_ALIASES[value]
    if value in OUTPUT_NAMES:  # legacy numeric key, still accepted
        return value
    raise argparse.ArgumentTypeError(
        f"invalid type {value!r}. Choices: {', '.join(VALID_TYPES)} "
        f"(legacy numeric keys {', '.join(OUTPUT_NAMES)} also accepted)"
    )


def get_args():
    p = argparse.ArgumentParser(
        description="Generate hallucination datasets (async by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "data" / "singlehop_synthetic_toolace_converted.json"),
        help="Input dataset path (converted ToolACE JSON)",
    )
    p.add_argument(
        "--type2-1-path",
        default=None,
        help=(
            "Override the undergeneration input file for answer_mismatch_gated / "
            "answer_mismatch_targeted generation. Defaults to <out-dir>/type2_1_output.json. "
            "Use this to pass a pre-filtered subset (e.g. type1_2_input.json)."
        ),
    )
    p.add_argument("--base-url",   default="http://172.17.0.1:8000/v1", help="vLLM server URL")
    p.add_argument("--api-key",    default="dummy",                       help="API key")
    p.add_argument("--model",      default="Qwen/Qwen2.5-14B-Instruct",  help="Model name")
    p.add_argument("--timeout",    type=float, default=300.0,             help="Request timeout (s)")
    p.add_argument(
        "--types", nargs="+", type=_type_arg,
        default=[WORD_ALIASES[t] for t in DEFAULT_TYPES],
        metavar="TYPE",
        help=(
            f"Which types to generate. Choices: {', '.join(VALID_TYPES)}. "
            f"Default: {' '.join(DEFAULT_TYPES)}. "
            "Note: answer_mismatch_gated/answer_mismatch_targeted require "
            "undergeneration to be run first (or already exist in --out-dir)."
        ),
    )
    p.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "output" / "singlehop_new"),
        help="Output directory (JSON + JSONL written here)",
    )
    p.add_argument(
        "--sync", action="store_true",
        help="Disable async mode (one row at a time). Only affects answer_mismatch_legacy and overgeneration_legacy.",
    )
    p.add_argument(
        "--batch-size", type=int, default=10,
        help="Concurrent async requests per batch (default: 10). Async only.",
    )
    p.add_argument(
        "--fresh", action="store_true",
        help=(
            "Delete existing output files for the requested types before starting. "
            "Does NOT delete the undergeneration output when only --fresh for "
            "answer_mismatch_gated is set."
        ),
    )
    return p.parse_args()


def _out(out_dir: str, type_key: str) -> str:
    return f"{out_dir}/{OUTPUT_NAMES[type_key]}.json"


def _try_parse_json(s: str):
    """Try several recovery strategies to parse a malformed JSON string.

    Returns a parsed dict/list on success, None on failure.
    Strategies (in order):
      1. Direct json.loads (catches the 240 clean double-encoded records)
      2. Strip a single trailing spurious quote character
      3. Strip a single trailing spurious closing brace
      4. Find the longest valid JSON prefix (handles mid-string truncation)
    """
    s = s.strip()
    if not s.startswith(("{", "[")):
        return None
    # Strategy 1: clean double-encoding
    try:
        v = json.loads(s)
        if isinstance(v, (dict, list)):
            return v
    except Exception:
        pass
    # Strategy 2: trailing spurious quote
    if s[-1] in ('"', "'"):
        try:
            v = json.loads(s[:-1])
            if isinstance(v, (dict, list)):
                return v
        except Exception:
            pass
    # Strategy 3: trailing spurious closing brace
    if s[-1] == "}":
        try:
            v = json.loads(s[:-1])
            if isinstance(v, (dict, list)):
                return v
        except Exception:
            pass
    # Strategy 4: longest valid JSON prefix
    for end in range(len(s) - 1, 0, -1):
        try:
            v = json.loads(s[:end])
            if isinstance(v, (dict, list)):
                return v
        except Exception:
            continue
    return None


def _unwrap_string_results(dataset: list) -> list:
    """Fix double-encoded or malformed JSON in tool_response.results.

    Handles:
    - Clean double-encoding: results = "{\"key\": \"val\"}"  (240 records)
    - Trailing spurious quote: results = "{...}\""           (3 records)
    - Trailing extra brace: results = "{...}}"               (1 record)
    - Truncated JSON: recovers longest valid prefix           (1 record)
    Only replaces results when the recovered value is a dict or list.
    """
    fixed = 0
    for row in dataset:
        try:
            tr = json.loads(row["tool_response"])
        except Exception:
            continue
        if not isinstance(tr, dict):
            continue
        results = tr.get("results")
        if not isinstance(results, str):
            continue
        parsed = _try_parse_json(results)
        if parsed is None:
            continue
        tr["results"] = parsed
        row["tool_response"] = json.dumps(tr, ensure_ascii=False)
        fixed += 1
    if fixed:
        print(f"  [preprocess] unwrapped/repaired {fixed} results fields")
    return dataset


def _check_answer_mismatch_prereq(out_dir: str, label: str, override: str = None) -> str:
    """Return path to the undergeneration output, or raise if it doesn't exist.

    If *override* is given, uses that path instead of the default.
    """
    p = Path(override) if override else Path(_out(out_dir, "2.1"))
    if not p.exists() or p.stat().st_size == 0:
        raise SystemExit(
            f"\n  [{label}] {label} requires undergeneration output at:\n"
            f"        {p}\n"
            f"  Run with --types undergeneration first, or pass --type2-1-path."
        )
    return str(p)


def _check_answer_mismatch_gated_prereq(out_dir: str, override: str = None) -> str:
    return _check_answer_mismatch_prereq(out_dir, "answer_mismatch_gated", override)


def run_sync(args, dataset, ha, out):
    """Legacy slow path — one row at a time (types 1 and 3 only)."""
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=httpx.Timeout(args.timeout, connect=10.0),
    )

    if "2" in args.types:
        print("\n" + "=" * 60)
        print("UNDERGENERATION_LEGACY — flat deletion (no LLM)")
        print("=" * 60)
        ha.generate_type2_dataset(dataset, output_path=_out(out, "2"))

    if "2.1" in args.types:
        print("\n" + "=" * 60)
        print("UNDERGENERATION — cascade deletion (no LLM)")
        print("=" * 60)
        ha.generate_type2_1_dataset(dataset, output_path=_out(out, "2.1"))

    if "1" in args.types:
        print("\n" + "=" * 60)
        print("ANSWER_MISMATCH_LEGACY — schema-based hallucination (sync)")
        print("=" * 60)
        ha.generate_answer_mismatch_dataset(
            client, args.model, dataset, output_path=_out(out, "1"),
        )

    if "3" in args.types:
        print("\n" + "=" * 60)
        print("OVERGENERATION_LEGACY — tool-constrained overgeneration (sync)")
        print("=" * 60)
        ha.generate_type3_dataset(
            client, args.model, dataset, output_path=_out(out, "3"),
        )

    for t in ("1.1", "3.1"):
        if t in args.types:
            print(f"\n  [WARNING] --sync does not support type {NUMERIC_TO_WORD.get(t, t)}; use async mode.")


async def run_async(args, dataset, ha, out):
    """Default fast path — batched async LLM requests."""
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=httpx.Timeout(args.timeout, connect=10.0),
    )

    # ── No-LLM types first ────────────────────────────────────────────────────

    if "2" in args.types:
        print("\n" + "=" * 60)
        print("UNDERGENERATION_LEGACY — flat deletion (no LLM)")
        print("=" * 60)
        ha.generate_type2_dataset(dataset, output_path=_out(out, "2"))

    if "2.1" in args.types:
        print("\n" + "=" * 60)
        print("UNDERGENERATION — cascade deletion (no LLM, cleaner spans)")
        print("=" * 60)
        ha.generate_type2_1_dataset(dataset, output_path=_out(out, "2.1"))

    # ── LLM types ─────────────────────────────────────────────────────────────

    if "1" in args.types:
        print("\n" + "=" * 60)
        print(f"ANSWER_MISMATCH_LEGACY — schema-based hallucination (async, batch={args.batch_size})")
        print("=" * 60)
        await ha.generate_answer_mismatch_dataset_async(
            client, args.model, dataset,
            output_path=_out(out, "1"),
            batch_size=args.batch_size,
        )

    if "1.1" in args.types:
        t21_path = _check_answer_mismatch_gated_prereq(out, getattr(args, "type2_1_path", None))
        print("\n" + "=" * 60)
        print(f"ANSWER_MISMATCH_GATED — schema hallucination + undergeneration spans (async, batch={args.batch_size})")
        print(f"           reading undergeneration gate from: {t21_path}")
        print("=" * 60)
        await ha.generate_answer_mismatch_gated_dataset_async(
            client, args.model,
            type2_1_output_path=t21_path,
            output_path=_out(out, "1.1"),
            batch_size=args.batch_size,
        )

    if "1.2" in args.types:
        t21_path = _check_answer_mismatch_prereq(out, "answer_mismatch_targeted", getattr(args, "type2_1_path", None))
        print("\n" + "=" * 60)
        print(f"ANSWER_MISMATCH_TARGETED — span-targeted schema hallucination (async, batch={args.batch_size})")
        print(f"           reading undergeneration gate from: {t21_path}")
        print("=" * 60)
        await ha.generate_answer_mismatch_targeted_dataset_async(
            client, args.model,
            type2_1_output_path=t21_path,
            output_path=_out(out, "1.2"),
            batch_size=args.batch_size,
        )

    if "3" in args.types:
        print("\n" + "=" * 60)
        print(f"OVERGENERATION_LEGACY — tool overgeneration (async, batch={args.batch_size})")
        print("=" * 60)
        await ha.generate_type3_dataset_async(
            client, args.model, dataset,
            output_path=_out(out, "3"),
            batch_size=args.batch_size,
        )

    if "3.1" in args.types:
        print("\n" + "=" * 60)
        print(f"OVERGENERATION — overgeneration + quality gates (async, batch={args.batch_size})")
        print("=" * 60)
        await ha.generate_type3_1_dataset_async(
            client, args.model, dataset,
            output_path=_out(out, "3.1"),
            batch_size=args.batch_size,
        )


def main():
    args = get_args()

    # ── Validate type ordering: 1.1 needs 2.1 to appear before it ─────────────
    if "1.1" in args.types and "2.1" not in args.types:
        # 2.1 must already exist from a prior run
        pass  # _check_answer_mismatch_gated_prereq() will catch it at runtime

    with open(args.dataset) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} rows from {args.dataset}")
    dataset = _unwrap_string_results(dataset)

    ha  = HallucinationAuto()
    out = args.out_dir.rstrip("/")
    Path(out).mkdir(parents=True, exist_ok=True)

    if args.fresh:
        import os
        for t in args.types:
            p = Path(_out(out, t))
            if p.exists():
                os.remove(p)
                print(f"  [fresh] removed {p}")
            # also remove the skipped sidecar for 2 and 2.1
            if t in ("2.1",):
                skipped = Path(_out(out, t).replace(".json", ".skipped.json"))
                if skipped.exists():
                    os.remove(skipped)
                    print(f"  [fresh] removed {skipped}")

    print(f"\nTypes to run : {' '.join(NUMERIC_TO_WORD.get(t, t) for t in args.types)}")
    print(f"Out dir      : {out}")
    print(f"Server       : {args.base_url}")
    print(f"Model        : {args.model}")
    print(f"Batch size   : {args.batch_size}" + (" (ignored — sync mode)" if args.sync else ""))

    if args.sync:
        print("\nMode: SYNC (one row at a time — slow)")
        run_sync(args, dataset, ha, out)
    else:
        print(f"\nMode: ASYNC (batch_size={args.batch_size})")
        asyncio.run(run_async(args, dataset, ha, out))

    print("\nDone!")


if __name__ == "__main__":
    main()
