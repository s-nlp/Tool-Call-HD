import json
import argparse
from openai import OpenAI
import httpx
from old_trash.old_code.corrupt.hallucination_auto import HallucinationAuto


def main():
    parser = argparse.ArgumentParser(description="Generate hallucination datasets")
    parser.add_argument("--dataset", default="dataset_v3_tagged_cleaned_sys.json", help="Input dataset path")
    parser.add_argument("--base-url", default="http://10.16.88.77:9091/v1", help="vLLM server URL")
    parser.add_argument("--api-key", default="bottle-of-water-14b-instruct", help="API key")
    parser.add_argument("--model", default="qwen2.5-14b-instruct", help="Model name")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout (seconds)")
    parser.add_argument("--types", nargs="+", default=["1", "2", "3"],
                        choices=["1", "2", "3"], help="Which types to generate (default: all)")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} rows from {args.dataset}")

    ha = HallucinationAuto()

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=httpx.Timeout(args.timeout, connect=10.0),
    )

    out = args.out_dir.rstrip("/")

    # ── Type 2: no LLM, instant ──
    if "2" in args.types:
        print("\n" + "=" * 60)
        print("TYPE 2: Deletion-based overgeneration (no LLM)")
        print("=" * 60)
        ha.generate_type2_dataset(dataset, output_path=f"{out}/type2_output.json")

    # ── Type 1: schema-based corruption ──
    if "1" in args.types:
        print("\n" + "=" * 60)
        print("TYPE 1: Schema-based hallucination (guided decoding)")
        print("=" * 60)
        ha.generate_type1_dataset(
            client, args.model, dataset,
            output_path=f"{out}/type1_output.json",
        )

    # ── Type 3: subtle overgeneration ──
    if "3" in args.types:
        print("\n" + "=" * 60)
        print("TYPE 3: Tool-constrained overgeneration")
        print("=" * 60)
        ha.generate_type3_dataset(
            client, args.model, dataset,
            output_path=f"{out}/type3_output.json",
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
