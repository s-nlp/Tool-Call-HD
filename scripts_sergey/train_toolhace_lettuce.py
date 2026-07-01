from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from toolhace_lettuce_utils import DEFAULT_HF_DATASET, load_rows, row_to_lettuce_example


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LettuceDetect-style token classifier on ToolHACE unified rows."
    )
    parser.add_argument("--train-input", default=None, help="Local train file (.json, .jsonl, or .parquet).")
    parser.add_argument("--dev-input", default=None, help="Optional local dev file (.json, .jsonl, or .parquet).")
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help=f"Hugging Face dataset repo id. Example: {DEFAULT_HF_DATASET}",
    )
    parser.add_argument("--hf-train-split", default="train", help="Train split name for Hugging Face loading.")
    parser.add_argument("--hf-dev-split", default="dev", help="Dev split name for Hugging Face loading.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--output-dir", default="outputs/toolhace_lettuce_modernbert_base")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lang", default="en", help="Prompt language passed to LettuceDetect prompt templates.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Used when no explicit dev split is provided.")
    return parser.parse_args()


def load_train_dev_rows(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    if args.train_input:
        train_rows = load_rows(args.train_input, hf_token=hf_token)
        if args.dev_input:
            dev_rows = load_rows(args.dev_input, hf_token=hf_token)
            return train_rows, dev_rows
        return train_rows, []

    if not args.hf_dataset:
        raise ValueError("Provide --train-input or --hf-dataset.")

    train_rows = load_rows(None, args.hf_dataset, args.hf_train_split, hf_token)
    try:
        dev_rows = load_rows(None, args.hf_dataset, args.hf_dev_split, hf_token)
    except Exception:
        dev_rows = []
    return train_rows, dev_rows


def rows_to_samples(rows: list[dict], default_split: str, lang: str):
    from lettucedetect.datasets.hallucination_dataset import HallucinationSample
    from lettucedetect.detectors.prompt_utils import PromptUtils

    samples = []
    skipped = 0
    for row in rows:
        if "type" not in row and "span_labels" not in row and "label" not in row:
            skipped += 1
            continue

        try:
            item = row_to_lettuce_example(row)
        except Exception:
            skipped += 1
            continue

        if not item["output"] or not item["contexts"]:
            skipped += 1
            continue

        prompt = PromptUtils.format_context(item["contexts"], item["query"] or None, lang)
        samples.append(
            HallucinationSample(
                prompt=prompt,
                answer=item["output"],
                labels=item["gold_spans"],
                split=row.get("split", default_split),
                task_type=item["gold_type"] or "toolhace",
                dataset="toolhace",
                language=lang,
            )
        )

    print(f"Prepared {len(samples)} samples for split='{default_split}' (skipped {skipped})")
    return samples


def split_train_dev(samples: list, dev_ratio: float, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    dev_size = max(1, int(len(shuffled) * dev_ratio)) if len(shuffled) > 1 else 0
    if dev_size == 0:
        return shuffled, []
    return shuffled[:-dev_size], shuffled[-dev_size:]


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
    )

    set_seed(args.seed)

    train_rows, dev_rows = load_train_dev_rows(args)
    train_samples = rows_to_samples(train_rows, "train", args.lang)
    dev_samples = rows_to_samples(dev_rows, "dev", args.lang) if dev_rows else []

    if not train_samples:
        raise ValueError("No labeled training samples were prepared.")

    if not dev_samples:
        train_samples, dev_samples = split_train_dev(train_samples, args.dev_ratio, args.seed)

    if not dev_samples:
        raise ValueError("No dev samples available after splitting.")

    from lettucedetect.datasets.hallucination_dataset import HallucinationDataset
    from lettucedetect.models.trainer_ import Trainer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

    train_dataset = HallucinationDataset(train_samples, tokenizer, max_length=args.max_length)
    dev_dataset = HallucinationDataset(dev_samples, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        trust_remote_code=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        test_loader=dev_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=args.output_dir,
        accumulation_steps=args.grad_accum,
    )
    trainer.train()


if __name__ == "__main__":
    main()
