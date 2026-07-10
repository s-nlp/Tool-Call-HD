"""
train.py — Unsloth-based LoRA fine-tuning for tool-calling error detection.

After several failed attempts with raw TRL+PEFT on Qwen3.5 (the model isn't
yet in TRL's known-families list for chat-template patching, and PEFT
issue #2653 makes vocab-expansion + tied-embeddings unstable), this script
uses Unsloth, which has model-specific patches for Qwen3.5:
  - Native Qwen3.5 chat template support (no template borrowing needed)
  - Tied-embedding handling done internally
  - train_on_responses_only masks user/system tokens via known role-prefix
    strings, avoiding the {% generation %} marker requirement entirely
Reference: https://unsloth.ai/docs/models/qwen3.5/fine-tune

bf16 LoRA — Unsloth explicitly advises against 4-bit for Qwen3.5 due to
higher than normal quantization error. VRAM:
  Qwen3.5-9B dense:  22 GB
  Qwen3.5-27B dense: 56 GB
  Gemma 4 E4B:       ~10 GB

Required: transformers v5 (Unsloth installs this automatically on `pip
install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo`).

Usage:
  python train.py --model qwen   --data ./sft_data_qwen   --output ./out/qwen9b
  python train.py --model gemma  --data ./sft_data_gemma  --output ./out/gemma4
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Unsloth MUST be imported before transformers so its patches apply. Inside
# Unsloth, `FastLanguageModel` is the path for text-only dense models
# (Qwen3.5-9B/27B). `FastModel` is for MoE (Qwen3.5-35B-A3B) and unified
# multimodal flows — different code path.
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from datasets import DatasetDict, load_from_disk
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str         # short id for output paths/logs
    hf_id: str        # exact HF id (Unsloth's docs use Qwen/... directly)
    # train_on_responses_only role markers. These strings must EXACTLY match
    # the prefixes the chat template emits for user vs assistant turns. The
    # trailing "\n" is part of the marker — Unsloth searches for the literal
    # string in the tokenized input. If wrong, masks come out empty and you
    # get a divide-by-zero crash from "all labels are -100".
    instruction_part: str
    response_part: str
    # Per-row kwargs passed to apply_chat_template (already set by
    # prepare_data.py via the chat_template_kwargs column, but kept here
    # for the explicit render-and-inspect path).
    chat_template_kwargs: dict


MODELS: dict[str, ModelSpec] = {
    "qwen_2b": ModelSpec(
        name="qwen3.5-2b",
        hf_id="Qwen/Qwen3.5-2B",
        # Qwen ChatML markers — same as Qwen3 (the Unsloth FAQ documents
        # these explicitly for the Qwen3 family).
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        chat_template_kwargs={"enable_thinking": False},
    ),
    "qwen_08b": ModelSpec(
        name="qwen3.5-0.8b",
        hf_id="Qwen/Qwen3.5-0.8B",
        # Qwen ChatML markers — same as Qwen3 (the Unsloth FAQ documents
        # these explicitly for the Qwen3 family).
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        chat_template_kwargs={"enable_thinking": False},
    ),
    "gemma": ModelSpec(
        name="gemma4-e2b",
        hf_id="unsloth/gemma-4-e2b-it",
        # Gemma's role markers (also from Unsloth's FAQ). Note Gemma uses
        # "model" rather than "assistant" as the assistant role name.
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
        chat_template_kwargs={},
    ),
}


# ---------------------------------------------------------------------------
# Render messages → text via the tokenizer's chat template. We do this
# explicitly (rather than letting SFTTrainer auto-format) so we can honor
# per-row chat_template_kwargs that prepare_data.py emitted, and so we can
# verify the response_part marker is actually present before training.
# ---------------------------------------------------------------------------
def format_dataset(dataset: DatasetDict, tokenizer, spec: ModelSpec,
                   max_seq_length: int) -> DatasetDict:
    def render(ex):
        kwargs = ex.get("chat_template_kwargs") or spec.chat_template_kwargs
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
            **kwargs,
        )
        # Use .encode (text-only path) for length counting. tokenizer(text)
        # would also work for a plain tokenizer, but for a multimodal
        # processor it routes through the image branch — defensive choice.
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        return {"text": text, "_n_tokens": n_tokens}

    print("\nRendering chat templates...")
    rendered = dataset.map(render, desc="Apply chat template")

    # Length stats — long examples blow up VRAM after padding.
    for split_name in rendered:
        lens = rendered[split_name]["_n_tokens"]
        over = sum(1 for n in lens if n > max_seq_length)
        p95 = sorted(lens)[int(len(lens) * 0.95)] if lens else 0
        print(f"  {split_name}: n={len(lens)} max={max(lens)} "
              f"mean={sum(lens)/len(lens):.0f} p95={p95} "
              f"over {max_seq_length}: {over} ({100*over/len(lens):.1f}%)")

    rendered = rendered.filter(lambda ex: ex["_n_tokens"] <= max_seq_length)

    # Keep "text" (what SFTTrainer reads) and a few metadata columns useful
    # for downstream eval. Drop everything else, including 'messages' and
    # 'chat_template_kwargs' — they confuse SFTTrainer if left in.
    keep = {"text", "type", "label", "n_spans", "dialogue_id"}
    drop = [c for c in rendered["train"].column_names if c not in keep]
    rendered = rendered.remove_columns(drop)
    print("After length filter: "
          + ", ".join(f"{k}={len(v)}" for k, v in rendered.items()))
    return rendered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), required=True,
                    help="'qwen' for Qwen3.5-9B, 'gemma' for Gemma 4 E4B")
    ap.add_argument("--data", required=True,
                    help="Path to the SFT DatasetDict from prepare_data.py")
    ap.add_argument("--output", required=True,
                    help="Output directory for checkpoints + final adapter")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16,
                    help="Unsloth recommends alpha == r (NOT 2*r).")
    ap.add_argument("--lora-dropout", type=float, default=0.0,
                    help="Unsloth's fast kernels require dropout=0.0.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=3407,
                    help="Unsloth examples use 3407 — keeping it for repro.")
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="For quick smoke tests.")
    ap.add_argument("--max-eval-samples", type=int, default=500)
    args = ap.parse_args()

    spec = MODELS[args.model]
    print(f"=== Training {spec.name} ({spec.hf_id}) via Unsloth ===")

    # ---- Load model + tokenizer ----
    # FastLanguageModel is the dense text-only path. load_in_4bit=False +
    # load_in_16bit=True selects bf16. full_finetuning=False enables LoRA.
    #
    # IMPORTANT for Qwen3.5: even though we're using the language-only
    # FastLanguageModel API, the returned "tokenizer" object is actually the
    # multimodal *processor* (because Qwen3.5 is internally a unified VLM).
    # Calling processor(text) with a string routes through the image-input
    # branch and crashes with "Incorrect image source".
    #
    # We extract the actual text tokenizer via .tokenizer when it exists;
    # for purely text models (e.g. older Qwen3) the returned object IS the
    # tokenizer, so the hasattr guard handles both cases.
    print("\nLoading model in bf16 (no quantization)...")
    model, processor = FastLanguageModel.from_pretrained(
        model_name=spec.hf_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
        print(f"  Extracted text tokenizer from multimodal processor "
              f"({type(processor).__name__} → {type(tokenizer).__name__})")
    else:
        tokenizer = processor
        print(f"  Using returned object directly as tokenizer "
              f"({type(tokenizer).__name__})")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  pad_token was None; set to eos_token ({tokenizer.eos_token!r})")

    # ---- Attach LoRA ----
    # No modules_to_save: Unsloth's text-only example deliberately omits it.
    # The Qwen3.5 chat template is already in the tokenizer (no new tokens
    # added), so there's nothing for modules_to_save to handle. The vision
    # branch is the one that needs it.
    print(f"\nAttaching LoRA (r={args.lora_r}, alpha={args.lora_alpha}, "
          f"dropout={args.lora_dropout})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # "unsloth" checkpointing is their custom impl — lower VRAM than
        # standard HF gradient_checkpointing.
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    # ---- Dataset ----
    print(f"\nLoading dataset from {args.data}...")
    dataset = load_from_disk(args.data)
    if not isinstance(dataset, DatasetDict):
        print("Error: --data must be a DatasetDict with train/test splits.",
              file=sys.stderr)
        return 1
    for split in ("train", "test"):
        if split not in dataset:
            print(f"Error: missing split {split!r}.", file=sys.stderr)
            return 1
    print(f"  train={len(dataset['train'])}, test={len(dataset['test'])}")

    dataset = format_dataset(dataset, tokenizer, spec, args.max_seq_length)

    if args.max_train_samples:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(
            range(min(args.max_train_samples, len(dataset["train"])))
        )
        print(f"Subsampled train to {len(dataset['train'])}")
    if args.max_eval_samples and len(dataset["test"]) > args.max_eval_samples:
        dataset["test"] = dataset["test"].shuffle(seed=args.seed).select(
            range(args.max_eval_samples)
        )
        print(f"Capped eval at {len(dataset['test'])}")

    # ---- Sanity check the rendered text ----
    # Before training, verify the response_part marker actually appears in a
    # rendered example. If it doesn't, train_on_responses_only will mask
    # everything → ZeroDivisionError on the first loss step.
    sample = dataset["train"][0]["text"]
    print("\n--- Sample rendered text (first 1200 chars) ---")
    print(sample[:1200])
    print("--- ... ---")
    print("\n--- Last 400 chars (should end with assistant's JSON target) ---")
    print(sample[-400:])
    print("--- end ---")
    if spec.response_part not in sample:
        print(f"\nERROR: response_part {spec.response_part!r} not found in "
              f"rendered text. train_on_responses_only will produce all-zero "
              f"masks → ZeroDivisionError at train start.", file=sys.stderr)
        return 1

    # ---- SFT config ----
# ---- SFT config ----
    sft_config = SFTConfig(
        output_dir=args.output,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        # Schedule
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        # adamw_8bit per Unsloth's recipe — saves a bunch of optimizer VRAM
        # with no measurable quality cost for LoRA.
        optim="adamw_8bit",
        # Precision
        bf16=True,
        fp16=False,
        # Eval + save + log
        eval_strategy="no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        #metric_for_best_model="eval_loss",
        #greater_is_better=False,
        # Only return the scalar eval loss; don't gather logits across the
        # eval set. For Qwen3.5 with its ~152K vocab, the full logits
        # tensor is 2.5 GB in fp32 per batch, and accelerate's bf16→fp32
        # autocast hook on that tensor crashes with "CUDA driver error:
        # invalid argument" on CUDA 12.9 + Unsloth's Triton kernels.
        # We don't use those logits for anything (load_best_model_at_end
        # just needs eval_loss, which IS still computed), so skip the
        # whole gather/convert path.
        prediction_loss_only=True,
        logging_steps=args.logging_steps,
        report_to="none",
        seed=args.seed,
        # IMPORTANT: don't set assistant_only_loss=True. That requires the
        # chat template to have {% generation %} markers (which Qwen3.5
        # doesn't ship). train_on_responses_only (applied below) is the
        # Unsloth replacement and works on any template via prefix strings.
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    # Apply Unsloth's response-only masking. After this, the data collator
    # sets labels=-100 for everything outside the assistant response spans.
    print("\nApplying train_on_responses_only:")
    print(f"  instruction_part = {spec.instruction_part!r}")
    print(f"  response_part    = {spec.response_part!r}")
    trainer = train_on_responses_only(
        trainer,
        instruction_part=spec.instruction_part,
        response_part=spec.response_part,
    )

    # Trainable param count — should be small (LoRA only, no modules_to_save).
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # ---- Train ----
    print("\n=== Starting training ===")
    trainer.train()

    # ---- Save ----
    print("\n=== Saving final adapter ===")
    final_dir = Path(args.output) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved to {final_dir}")

    # ---- Final eval ----
    print("\n=== Final eval ===")
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
