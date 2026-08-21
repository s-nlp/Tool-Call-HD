"""Greedy vLLM evaluation of a tuned checkpoint on the ToolHACE test file."""

import argparse
import json
import os
import sys

from datasets import load_dataset
from vllm import LLM, SamplingParams

from pipeline_common import SYSTEM, build_prompt, parse_verdict

sys.stdout.reconfigure(errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="local base snapshot/tokenizer")
    parser.add_argument("--checkpoint", required=True, help="complete tuned checkpoint")
    parser.add_argument("--test-parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    args = parser.parse_args()

    dataset = load_dataset("parquet", data_files=args.test_parquet, split="train")
    count = min(args.limit, len(dataset)) if args.limit else len(dataset)
    indices = list(range(count))
    messages = [
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(dataset[index])},
        ]
        for index in indices
    ]

    llm = LLM(
        model=args.checkpoint,
        tokenizer=args.base_model,
        max_model_len=16384,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=600)
    kwargs = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        outputs = llm.chat(messages, sampling, **kwargs)
    except TypeError:
        outputs = llm.chat(messages, sampling)

    os.makedirs(args.out, exist_ok=True)
    output_path = os.path.join(args.out, "verdicts.jsonl")
    with open(output_path, "w", encoding="utf-8") as handle:
        for index, output in zip(indices, outputs):
            example = dataset[index]
            raw = output.outputs[0].text
            verdict = parse_verdict(raw)
            handle.write(
                json.dumps(
                    {
                        "idx": index,
                        "dialogue_id": example["dialogue_id"],
                        "type": example["type"],
                        "label": int(example["label"]),
                        "parsed": verdict is not None,
                        "errors": (verdict or {}).get("errors"),
                        "raw": raw[-400:],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print("wrote", output_path, flush=True)


if __name__ == "__main__":
    main()
