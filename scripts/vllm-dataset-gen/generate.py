"""
Singlehop generation entry point.

By DEFAULT runs the ASYNC batched versions of type1/type3 generation
(batch_size concurrent requests against the vLLM server). Pass --sync
to fall back to the slow one-row-at-a-time versions.

Type 2 has no LLM call so always runs instantly regardless of mode.

Usage:
    python3 generate.py                       # async, all 3 types
    python3 generate.py --types 2 3           # async, types 2 and 3
    python3 generate.py --sync                # disable async (legacy slow mode)
    python3 generate.py --batch-size 20       # increase concurrency
"""
import json
import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI, AsyncOpenAI
import httpx
from hallucination_auto import HallucinationAuto


def get_args():
    p = argparse.ArgumentParser(description="Generate hallucination datasets (async by default)")
    p.add_argument("--dataset",
                   default=str(Path(__file__).parent / "dataset_v3_tagged_cleaned_sys.json"),
                   help="Input dataset path")
    p.add_argument("--base-url", default="http://172.17.0.1:8000/v1", help="vLLM server URL")
    p.add_argument("--api-key", default="dummy", help="API key")
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="Model name")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="Request timeout in seconds (default: 300)")
    p.add_argument("--types", nargs="+", default=["1", "2", "3"],
                   choices=["1", "2", "3"], help="Which types to generate (default: all)")
    p.add_argument("--out-dir",
                   default=str(PROJECT_ROOT / "singlehop_new"),
                   help="Output directory")
    p.add_argument("--sync", action="store_true",
                   help="Disable async mode (process rows one-by-one). Much slower.")
    p.add_argument("--batch-size", type=int, default=10,
                   help="Concurrent async requests per batch (default: 10). Async only.")
    p.add_argument("--fresh", action="store_true",
                   help="Delete any existing output files before starting (forces full regeneration).")
    return p.parse_args()


def run_sync(args, dataset, ha, out):
    """Legacy slow path — one row at a time."""
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=httpx.Timeout(args.timeout, connect=10.0),
    )

    if "2" in args.types:
        print("\n" + "=" * 60)
        print("TYPE 2: Deletion-based overgeneration (no LLM)")
        print("=" * 60)
        ha.generate_type2_dataset(dataset, output_path=f"{out}/type2_output.json")

    if "1" in args.types:
        print("\n" + "=" * 60)
        print("TYPE 1: Schema-based hallucination (sync)")
        print("=" * 60)
        ha.generate_type1_dataset(
            client, args.model, dataset, output_path=f"{out}/type1_output.json",
        )

    if "3" in args.types:
        print("\n" + "=" * 60)
        print("TYPE 3: Tool-constrained overgeneration (sync)")
        print("=" * 60)
        ha.generate_type3_dataset(
            client, args.model, dataset, output_path=f"{out}/type3_output.json",
        )


async def run_async(args, dataset, ha, out):
    """Default fast path — batched async LLM requests."""
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=httpx.Timeout(args.timeout, connect=10.0),
    )

    # Type 2 — no LLM, sync is fine and instant
    if "2" in args.types:
        print("\n" + "=" * 60)
        print("TYPE 2: Deletion-based overgeneration (no LLM)")
        print("=" * 60)
        ha.generate_type2_dataset(dataset, output_path=f"{out}/type2_output.json")

    if "1" in args.types:
        print("\n" + "=" * 60)
        print(f"TYPE 1: Schema-based hallucination (async, batch_size={args.batch_size})")
        print("=" * 60)
        await ha.generate_type1_dataset_async(
            client, args.model, dataset,
            output_path=f"{out}/type1_output.json",
            batch_size=args.batch_size,
        )

    if "3" in args.types:
        print("\n" + "=" * 60)
        print(f"TYPE 3: Tool-constrained overgeneration (async, batch_size={args.batch_size})")
        print("=" * 60)
        await ha.generate_type3_dataset_async(
            client, args.model, dataset,
            output_path=f"{out}/type3_output.json",
            batch_size=args.batch_size,
        )


def main():
    args = get_args()

    with open(args.dataset) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} rows from {args.dataset}")

    ha = HallucinationAuto()
    out = args.out_dir.rstrip("/")
    Path(out).mkdir(parents=True, exist_ok=True)

    if args.fresh:
        import os
        for t in args.types:
            p = Path(out) / f"type{t}_output.json"
            if p.exists():
                os.remove(p)
                print(f"  [fresh] removed {p}")

    if args.sync:
        print("Mode: SYNC (one row at a time — slow)")
        run_sync(args, dataset, ha, out)
    else:
        print(f"Mode: ASYNC (batch_size={args.batch_size})")
        asyncio.run(run_async(args, dataset, ha, out))

    print("\nDone!")


if __name__ == "__main__":
    main()
