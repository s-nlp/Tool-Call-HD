#!/usr/bin/env python
"""vLLM evaluation over MULTIPLE LoRA adapters without merging.

Swaps adapters in-engine (one base load, N adapter passes) so you can compare
several LoRAs cheaply. Same parser / summary / prompt-tokenization as the
merged-path script, so numbers are directly comparable.

  python vllm_eval_lora.py \
      --base-model Qwen/Qwen3.5-2B \
      --adapters v1=./out_JULY/qwen3.5-2B-lora-v1/final \
                 v2=./out_JULY/qwen3.5-2B-lora-v2/final \
      --data ./sft_data_full_V3_nothinking_qwen --split test \
      --include-base

================================  READ THIS  ================================
vLLM matches LoRA against Qwen3.5's FUSED internal module names (qkv_proj,
gate_up_proj, in_proj_ba, ...). Adapters trained with PEFT/Unsloth's UNFUSED
names (q_proj, gate_proj, in_proj_a, ...) — especially on the gated-delta
linear_attn layers — can be SILENTLY IGNORED at load with only a WARNING, not
an error. That reproduces the "model looks base-like" failure with no crash.

This script:
  1. Prints each adapter's rank + target_modules and flags risky ones.
  2. Tells you exactly which vLLM log line to grep for to confirm nothing was
     dropped.
But the only authoritative check is empirical: run ONE adapter both via this
script AND via the merged path (vllm_eval.py) on ~200 rows and confirm the
predicted types match. Do that once before trusting the comparison.
============================================================================
"""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# --- JSON parsing: identical balanced-brace scanner as the fixed eval.py ---
def _extract_first_json_object(s: str) -> str | None:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return None


def parse_prediction(text: str) -> tuple[dict | None, str | None]:
    if not text or not text.strip():
        return None, "empty_output"
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
    raw = _extract_first_json_object(stripped)
    if raw is None:
        return (None, "unbalanced_or_truncated_json") if "{" in stripped \
            else (None, "no_json_object_found")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"json_decode_error:{e.msg}"
    if not isinstance(obj, dict):
        return None, "not_a_dict"
    if "type" not in obj:
        return obj, "missing_type_field"
    return obj, None


def gold_spans_from_messages(messages: list[dict]) -> list[list[int]]:
    """[[start, end], ...] from the gold assistant turn (last message), so each
    adapter's JSONL is self-contained for span scoring in check_eval.py."""
    if not messages or messages[-1].get("role") != "assistant":
        return []
    raw = _extract_first_json_object(messages[-1].get("content", "") or "")
    if raw is None:
        return []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return []
    spans = obj.get("spans") if isinstance(obj, dict) else None
    if not isinstance(spans, list):
        return []
    out = []
    for s in spans:
        if isinstance(s, dict):
            st, en = s.get("start"), s.get("end")
            if isinstance(st, int) and isinstance(en, int) and en > st:
                out.append([st, en])
    return out


# --- Adapter preflight -----------------------------------------------------
# vLLM's known supported LoRA target modules for the Qwen3.5 GDN class (fused
# names). PEFT adapters usually list the unfused names on the left; vLLM maps
# the common ones (q/k/v->qkv_proj, gate/up->gate_up_proj) but the gated-delta
# linear_attn projections are the ones that tend to fall through.
_GDN_RISKY = {"in_proj_a", "in_proj_b", "in_proj", "conv1d", "linear_attn"}


def inspect_adapter(name: str, path: str) -> None:
    cfg_path = Path(path) / "adapter_config.json"
    if not cfg_path.exists():
        print(f"  [{name}] no adapter_config.json at {path} — cannot preflight")
        return
    cfg = json.loads(cfg_path.read_text())
    r = cfg.get("r")
    tmods = cfg.get("target_modules")
    tset = set(tmods) if isinstance(tmods, list) else {str(tmods)}
    risky = sorted(tset & _GDN_RISKY)
    print(f"  [{name}] r={r}  target_modules={sorted(tset)}")
    if risky:
        print(f"  [{name}] ⚠ targets gated-delta modules {risky} — vLLM may "
              f"silently drop these. VERIFY against the merged path.")
    if "base_model_name_or_path" in cfg:
        print(f"  [{name}] trained on base: {cfg['base_model_name_or_path']}")


def summarize(results: list[dict], n: int, elapsed: float, label: str) -> dict:
    n_parsed = sum(1 for r in results if r["parsed"] is not None)
    per_class = defaultdict(lambda: {"n": 0, "correct": 0, "preds": Counter()})
    parse_errors = Counter()
    n_correct = 0
    for r in results:
        gt = r["gold_type"]
        pc = per_class[gt]
        pc["n"] += 1
        if r["parsed"] is not None and "type" in r["parsed"]:
            pred = r["parsed"]["type"]
        else:
            pred = "<unparseable>"
            if r["parse_error"]:
                parse_errors[r["parse_error"].split(":")[0]] += 1
        pc["preds"][pred] += 1
        if pred == gt:
            pc["correct"] += 1
            n_correct += 1
    print(f"\n=== {label} ===")
    print(f"Parse rate: {n_parsed/n:.1%} ({n_parsed}/{n})   "
          f"Overall acc: {n_correct/n:.1%}   "
          f"Throughput: {n/elapsed:.1f} ex/s")
    for gt in sorted(per_class):
        pc = per_class[gt]
        print(f"  {gt:24s} n={pc['n']:5d}  acc={pc['correct']/pc['n']:.1%}  "
              f"top: {dict(pc['preds'].most_common(4))}")
    if parse_errors:
        print(f"  parse errors: {dict(parse_errors.most_common())}")
    return {"label": label, "parse_rate": n_parsed / n,
            "overall_acc": n_correct / n, "throughput": n / elapsed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True,
                    help="Base checkpoint the adapters were trained on.")
    ap.add_argument("--adapters", nargs="+", required=True,
                    help="One or more name=path (or bare path) LoRA adapters.")
    ap.add_argument("--include-base", action="store_true",
                    help="Also eval the base model with NO adapter, as a "
                         "reference row (sanity: adapters should beat it).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="vllm_lora_preds")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-model-len", type=int, default=4608)
    ap.add_argument("--max-prompt-tokens", type=int, default=4096)
    ap.add_argument("--max-lora-rank", type=int, default=64,
                    help="Must be >= the largest adapter rank r.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    # Parse name=path specs.
    adapters = []
    for spec in args.adapters:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name, path = Path(spec).parent.name or Path(spec).name, spec
        adapters.append((name, path))

    print("Adapters to evaluate:")
    for name, path in adapters:
        inspect_adapter(name, path)
    print("\n>> While vLLM loads each adapter, watch its log for:\n"
          "   'is not in the model's supported LoRA target modules' / "
          "'will be ignored'\n"
          "   Any such line means that adapter is partially applied — merge "
          "it instead.\n")

    # ---- Tokenizer + dataset ----
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    ds = load_from_disk(args.data)
    if not isinstance(ds, DatasetDict) or args.split not in ds:
        print(f"--data must be a DatasetDict with split {args.split!r}.",
              file=sys.stderr)
        return 1
    split = ds[args.split]
    if args.max_samples is not None:
        split = split.select(range(min(args.max_samples, len(split))))
    n = len(split)
    cols = split.column_names
    print(f"Evaluating {n} examples from {args.data}:{args.split}")

    # ---- Stop tokens ----
    stop_ids = []
    for s in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(s)
        if tid is not None and tid != tok.unk_token_id:
            stop_ids.append(tid)
    if tok.eos_token_id is not None and tok.eos_token_id not in stop_ids:
        stop_ids.append(tok.eos_token_id)
    print(f"  stop_token_ids = {stop_ids}")

    # ---- Render prompts as token ids once (shared across adapters) ----
    ctk = split["chat_template_kwargs"] if "chat_template_kwargs" in cols \
        else [None] * n
    default_ctk = {"enable_thinking": False}
    prompt_ids = []
    n_trunc = 0
    for i in range(n):
        kw = ctk[i] or default_ctk
        # Two-step render: to STRING, then encode to ids. apply_chat_template(
        # tokenize=True) can return a *string* for this processor-backed
        # tokenizer; vLLM then sees prompt_token_ids as a string and dies in
        # input validation ("'>' not supported between 'str' and 'int'"). The
        # __call__ path always yields list[int], and matches the HF eval
        # exactly (render with tokenize=False, encode add_special_tokens=False).
        text = tok.apply_chat_template(
            split["messages"][i], tokenize=False,
            add_generation_prompt=True, **kw,
        )
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > args.max_prompt_tokens:
            ids = ids[-args.max_prompt_tokens:]
            n_trunc += 1
        prompt_ids.append(ids)
    # Fail loud if anything isn't a clean list[int] before handing to vLLM.
    bad = next((j for j, p in enumerate(prompt_ids)
                if not isinstance(p, list) or not p
                or not isinstance(p[0], int)), None)
    if bad is not None:
        sample = prompt_ids[bad]
        kind = type(sample).__name__ if not isinstance(sample, list) \
            else (f"list[{type(sample[0]).__name__}]" if sample else "empty list")
        raise TypeError(
            f"prompt_ids[{bad}] is {kind}, expected list[int]. vLLM needs "
            f"integer token ids; check the chat-template render.")
    if n_trunc:
        print(f"  WARNING: left-truncated {n_trunc}/{n} prompts.")
    vllm_prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]

    gold_type = split["type"] if "type" in cols else [None] * n
    gold_label = split["label"] if "label" in cols else [None] * n
    gold_nspans = split["n_spans"] if "n_spans" in cols else [None] * n
    did = split["dialogue_id"] if "dialogue_id" in cols else [None] * n
    msgs_col = split["messages"]
    gold_spans = [gold_spans_from_messages(msgs_col[i]) for i in range(n)]

    # ---- Engine: one base load, LoRA enabled ----
    llm = LLM(
        model=args.base_model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        enable_lora=True,
        max_lora_rank=args.max_lora_rank,
        max_loras=1,            # one active at a time; we pass sequentially
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                              stop_token_ids=stop_ids)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the run list: optional base, then each adapter.
    runs = []
    if args.include_base:
        runs.append(("base", None))
    for idx, (name, path) in enumerate(adapters, start=1):
        runs.append((name, LoRARequest(name, idx, path)))

    comparison = []
    for label, lora_req in runs:
        t0 = time.time()
        gen = llm.generate(vllm_prompts, sampling, lora_request=lora_req)
        elapsed = time.time() - t0

        results = []
        for i, out in enumerate(gen):
            text = out.outputs[0].text
            parsed, err = parse_prediction(text)
            results.append({
                "dialogue_id": did[i], "gold_type": gold_type[i],
                "gold_label": gold_label[i], "gold_n_spans": gold_nspans[i],
                "gold_spans": gold_spans[i],
                "raw_output": text, "parsed": parsed, "parse_error": err,
            })
        with (out_dir / f"{label}.jsonl").open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        comparison.append(summarize(results, n, elapsed, label))

    # ---- Side-by-side ----
    print("\n=== Comparison ===")
    print(f"{'adapter':20s} {'parse_rate':>11s} {'overall_acc':>12s} "
          f"{'ex/s':>8s}")
    for c in comparison:
        print(f"{c['label']:20s} {c['parse_rate']:>10.1%} "
              f"{c['overall_acc']:>11.1%} {c['throughput']:>8.1f}")
    print(f"\nPer-adapter predictions written to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
