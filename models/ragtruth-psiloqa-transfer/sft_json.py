"""Full-parameter, completion-only SFT for JSON hallucination verdicts."""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from pipeline_common import SYSTEM, build_prompt, target_json


def rank_zero_print(*values):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*values, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local HF snapshot directory")
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--maxlen", type=int, default=6144)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--map-procs", type=int, default=8)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--eos-id", type=int, default=-1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = args.eos_id if args.eos_id >= 0 else tokenizer.eos_token_id

    train = load_dataset("parquet", data_files=args.train_parquet, split="train")
    if args.limit:
        train = train.select(range(min(args.limit, len(train))))

    def prepare(example):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(example)},
        ]
        try:
            prompt_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if not isinstance(prompt_ids, list):
            prompt_ids = prompt_ids["input_ids"]
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
        target_ids = tokenizer(
            target_json(example), add_special_tokens=False
        )["input_ids"] + [eos_id]
        input_ids = (prompt_ids + target_ids)[: args.maxlen]
        labels = ([-100] * len(prompt_ids) + target_ids)[: args.maxlen]
        if all(value == -100 for value in labels):
            raise ValueError("maxlen truncated the entire target")
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    processed = train.map(
        prepare,
        remove_columns=train.column_names,
        num_proc=args.map_procs,
        desc="tokenizing completion-only SFT data",
    )
    rank_zero_print(f"prepared {len(processed)} training examples")

    class Collator:
        def __call__(self, features):
            length = max(len(feature["input_ids"]) for feature in features)
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for feature in features:
                padding = length - len(feature["input_ids"])
                batch["input_ids"].append(
                    feature["input_ids"] + [tokenizer.pad_token_id] * padding
                )
                batch["attention_mask"].append(
                    feature["attention_mask"] + [0] * padding
                )
                batch["labels"].append(feature["labels"] + [-100] * padding)
            return {key: torch.tensor(value) for key, value in batch.items()}

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    rank_zero_print(
        "trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.3f}B",
    )

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=1.0,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        # This Transformers build accepts a fractional value through
        # warmup_steps (the working release recipe uses 0.03) and has no
        # warmup_ratio argument.
        warmup_steps=0.03,
        weight_decay=0.01,
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        bf16=True,
        remove_unused_columns=False,
        optim=args.optim,
        gradient_checkpointing=args.grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed,
        data_collator=Collator(),
    )
    trainer.train()

    if int(os.environ.get("RANK", "0")) == 0:
        save_dir = os.path.join(args.out, "model")
        trainer.model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer.save_pretrained(save_dir)
        rank_zero_print("saved", save_dir)


if __name__ == "__main__":
    main()
