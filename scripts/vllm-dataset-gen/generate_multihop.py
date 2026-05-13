"""
MULTIHOP generation — pruning-based hallucination injection.

Takes a multistep dialogue dataset and generates hallucinated versions at
every prefix depth (N turns down to 2 turns).  Single-hop depth is skipped
since we already have plenty of singlehop data.

4-turn dialogue example:
  [turn1, turn2, turn3, hall@turn4]   depth=4
  [turn1, turn2, hall@turn3]          depth=3
  [turn1, hall@turn2]                 depth=2
  depth=1 -> STOP (single-hop)

For SINGLEHOP generation use generate.py / run.sh instead.

Usage:
    python3 generate_multihop.py
    python3 generate_multihop.py --types 2
    python3 generate_multihop.py --types 1 3 --batch-size 10
    python3 generate_multihop.py --multistep other_multistep.json --out-dir my_results/
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from openai import AsyncOpenAI
import httpx

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from hallucination_auto import HallucinationAuto


def get_args():
    p = argparse.ArgumentParser(
        description="Multihop pruning-based hallucination generation"
    )
    p.add_argument(
        "--multistep",
        default=str(Path(__file__).parent / "toolace_multistep_clean.json"),
        help="Multistep dialogue dataset",
    )
    p.add_argument(
        "--base-url",
        default="http://172.17.0.1:8000/v1",
        help="vLLM server URL (needed for type1 and type3)",
    )
    p.add_argument("--api-key", default="dummy", help="API key")
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-14B-Instruct",
        help="Model name",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds (default: 300)",
    )
    p.add_argument(
        "--types",
        nargs="+",
        default=["1", "2", "3"],
        choices=["1", "2", "3"],
        help="Which hallucination types to generate (default: all)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Concurrent async requests per batch for type1/type3 (default: 5)",
    )
    p.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "pruneddataset"),
        help="Output directory for pruned_type*.json",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Delete any existing output files before starting (forces full regeneration).",
    )
    return p.parse_args()


async def run(args):
    multistep_path = Path(args.multistep)
    with open(multistep_path) as f:
        multistep = json.load(f)

    from collections import Counter
    turn_dist = Counter(d["analysis"]["completed_tool_turn_count"] for d in multistep)
    print(f"Loaded {len(multistep)} multistep dialogues")
    print(f"Turn distribution: {dict(sorted(turn_dist.items()))}")

    ha = HallucinationAuto()

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=httpx.Timeout(args.timeout, connect=10.0),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        import os
        for t in args.types:
            p = out_dir / f"pruned_type{t}.json"
            if p.exists():
                os.remove(p)
                print(f"  [fresh] removed {p}")

    types = args.types
    print(f"\nGenerating types: {types}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Batch size: {args.batch_size} (type1/type3 only)\n")

    if types == ["2"]:
        print("=" * 60)
        print("TYPE 2 only — no LLM required")
        print("=" * 60)
        all_t2 = []
        for dlg in multistep:
            n = dlg["analysis"]["completed_tool_turn_count"]
            for turn_k in range(n - 1, 0, -1):
                row = ha.extract_turn_as_row(dlg, turn_k)
                pruned = ha.prune_dialogue(dlg, turn_k)
                t2_row = ha.generate_type2_for_row(row)
                if t2_row:
                    injected = ha.inject_row_into_pruned(pruned, t2_row, "type2")
                    injected["_pruning_depth"] = turn_k + 1
                    injected["_dialogue_id"] = dlg["analysis"].get("dialogue_id", "")
                    all_t2.append(injected)

        out_path = out_dir / "pruned_type2.json"
        with open(out_path, "w") as f:
            json.dump(all_t2, f, ensure_ascii=False, indent=2)
        print(f"Done: {len(all_t2)} type2 dialogues -> {out_path}")
    else:
        t1, t2, t3 = await ha.generate_pruned_multistep(
            client=client,
            model=args.model,
            multistep=multistep,
            output_dir=str(out_dir),
            batch_size=args.batch_size,
            types=types,
        )
        if "1" in types:
            print(f"type1 -> {len(t1)} dialogues")
        if "2" in types:
            print(f"type2 -> {len(t2)} dialogues")
        if "3" in types:
            print(f"type3 -> {len(t3)} dialogues")

    print("\nDone!")


if __name__ == "__main__":
    args = get_args()
    asyncio.run(run(args))
