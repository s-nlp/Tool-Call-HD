"""Quick viewer for pruned hallucination datasets.

Usage:
    python3 scripts/show_samples.py                        # 3 random type2
    python3 scripts/show_samples.py --type type1           # type1
    python3 scripts/show_samples.py --type type3 -n 5      # 5 samples
    python3 scripts/show_samples.py --idx 7                # specific index
    python3 scripts/show_samples.py --seed 99              # different random pick
    python3 scripts/show_samples.py --dir path/to/dataset  # custom folder
"""

import argparse, json, random, textwrap
from pathlib import Path

# ── Colours ──────────────────────────────────────────────────────────────────
RED   = "\033[1;31m"
RESET = "\033[0m"
BOLD  = "\033[1m"

SEP  = "═" * 72
SEP2 = "─" * 72
ROLE = {"user": "USER", "assistant": "ASSISTANT", "tool": "TOOL RESPONSE"}


def show_dialogue(d: dict, idx: int, total: int, dataset: str):
    convs = d["conversations"]
    print(f"\n{SEP}")
    print(f"  {BOLD}Sample index: {idx}/{total-1}  |  {dataset}{RESET}")
    print(f"  Dialogue : {d['analysis'].get('dialogue_id','?')}")
    print(f"  Hall type: {d.get('hallucination_type','?')}  |  "
          f"Depth: {d['analysis']['completed_tool_turn_count']}  |  "
          f"Skipped: {d.get('skipped', False)}")
    print(SEP)

    for i, c in enumerate(convs):
        role    = c.get("from", "?")
        value   = c.get("value", "")
        label   = ROLE.get(role, role.upper())
        is_hall = "<hall>" in value

        marker = f"  {RED}★ HALLUCINATED{RESET}" if is_hall else ""
        print(f"\n  [{i}] {BOLD}{label}{RESET}{marker}")
        print(f"  {SEP2}")

        display = (value
                   .replace("<hall>",  f"{RED}❬")
                   .replace("</hall>", f"❭{RESET}"))

        for line in display.split("\n"):
            if len(line) > 76:
                line = textwrap.fill(line, width=76, subsequent_indent="      ")
            print(f"    {line}")

    hall_turns = [i for i, c in enumerate(convs) if "<hall>" in c.get("value", "")]
    print(f"\n  Hallucinated turn(s): {hall_turns}")
    print(SEP)


def main():
    p = argparse.ArgumentParser(description="Quick viewer for pruned hallucination datasets")
    p.add_argument("--type",  default="type2", choices=["type1","type2","type3"],
                   help="Dataset type (default: type2)")
    p.add_argument("-n",      type=int, default=3,
                   help="Number of random samples to show (default: 3)")
    p.add_argument("--idx",   type=int, default=None,
                   help="Show a specific index instead of random samples")
    p.add_argument("--seed",  type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--dir",   default=None,
                   help="Path to folder containing pruned_type*.json files")
    args = p.parse_args()

    # Resolve dataset path
    script_dir = Path(__file__).parent
    data_dir = Path(args.dir) if args.dir else script_dir.parent / "" / "pruneddataset"
    fpath = data_dir / f"pruned_{args.type}.json"

    if not fpath.exists():
        print(f"File not found: {fpath}")
        return

    with open(fpath) as f:
        data = json.load(f)

    print(f"\nLoaded {len(data)} dialogues from {fpath.name}")

    if args.idx is not None:
        if args.idx >= len(data):
            print(f"Index {args.idx} out of range (max {len(data)-1})")
            return
        show_dialogue(data[args.idx], args.idx, len(data), fpath.name)
    else:
        random.seed(args.seed)
        picks = random.sample(range(len(data)), min(args.n, len(data)))
        for idx in picks:
            show_dialogue(data[idx], idx, len(data), fpath.name)


if __name__ == "__main__":
    main()
