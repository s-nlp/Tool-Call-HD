"""
prepare_data.py — Convert the tool-calling error detection dataset to SFT format.

Input rows have columns:
  system, conversations, span_labels, label, type, subset,
  dialogue_id, available_tools, system_prompt_format

Each row becomes an SFT example with messages in the standard chat format:
  [system, user, assistant]
where the user message contains the full tool-calling trace to analyze and the
assistant message is the JSON target: {"type": ..., "spans": [...]}.

Output is saved as a HuggingFace dataset on disk, ready for TRL's SFTTrainer
with `assistant_only_loss=True` (loss is computed only on the JSON target).

Supports any chat model — pass --model to apply that tokenizer's chat template
for a sanity-check preview. The saved `messages` field is template-agnostic.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset


# ---------------------------------------------------------------------------
# Prompt: role description + taxonomy + output format.
# Kept in the system message so the user message is purely the case to analyze.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an error detector for tool-calling LLM responses. \
Given a tool-calling trace (the user query, the assistant's tool call, the \
tool's response, and the assistant's final answer), determine whether the \
final answer contains an error and if so, identify its type and exact spans.

Error types:
- clean: the final answer is faithful to the tool response and uses only \
available tools.
- answer_mismatch: the final answer contradicts the tool response (e.g. \
reports a value that differs from what the tool returned).
- overgeneration: the final answer adds information beyond what the tool \
response supports (commentary, inferences, or details not in the tool output).
- missing_tool: the final answer references capabilities, actions, or follow-ups \
that are not available as tools in the system prompt.
- undergeneration: the final answer omits important information that was \
present in the tool response.

Respond with a single JSON object:
{"type": "<class>", "spans": [{"start": <int>, "end": <int>, "text": "<substring>"}, ...]}

Span offsets are character positions in the final answer string. \
For 'clean' and 'undergeneration', spans must be an empty list. For other \
classes, list every erroneous span. The "text" field must equal \
answer[start:end] exactly."""


# ---------------------------------------------------------------------------
# Parsing helpers — handle both raw HF rows (already-parsed lists) and CSVs
# (string representations that came through pandas).
# ---------------------------------------------------------------------------
def _coerce_list(raw: Any) -> list:
    """Accept either a Python list (HF Dataset row) or a stringified list (CSV)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        return ast.literal_eval(s)
    # pyarrow types / numpy arrays / etc. — try to coerce
    try:
        return list(raw)
    except TypeError as e:
        raise TypeError(
            f"Cannot coerce value of type {type(raw).__name__} to list: {raw!r}"
        ) from e


def split_history_and_target(conversations: list[dict]) -> tuple[list[dict], str]:
    """Split a (possibly multi-turn) dialogue into history + target answer.

    Spans always refer to the LAST final_answer turn. Everything before that
    turn is conversation history the model needs to see for context (e.g. to
    judge 'missing_tool', the analyzer needs to know which tools were called
    earlier in the dialogue).

    Returns (history_turns, target_final_answer).
    """
    final_answer_indices = [
        i for i, t in enumerate(conversations) if t.get("turn_role") == "final_answer"
    ]
    if not final_answer_indices:
        return conversations, ""
    target_idx = final_answer_indices[-1]
    history = conversations[:target_idx]
    target = conversations[target_idx].get("value", "")
    return history, target


def format_history(history: list[dict]) -> str:
    """Render the conversation history as a readable transcript."""
    parts = []
    for turn in history:
        role = turn.get("turn_role")
        val = turn.get("value", "")
        label = {
            "user": "User",
            "tool_call": "Assistant tool call",
            "tool_response": "Tool response",
            "final_answer": "Assistant (earlier turn)",
        }.get(role, role or "?")
        parts.append(f"### {label}\n{val}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Common Unicode mismatches between span annotations and normalized answers.
# Curly quotes, non-breaking hyphen, and narrow nbsp show up in annotator
# output but get normalized to ASCII in the stored final_answer. NFKC does
# NOT cover these (Unicode treats them as semantically distinct), so we
# apply targeted substitutions.
# ---------------------------------------------------------------------------
_UNICODE_FIXES = {
    "\u2018": "'", "\u2019": "'",          # curly single quotes
    "\u201C": '"', "\u201D": '"',          # curly double quotes
    "\u2011": "-",                          # non-breaking hyphen
    "\u2013": "-", "\u2014": "-",           # en/em dash
    "\u202F": " ", "\u00A0": " ",           # narrow nbsp, nbsp
}


def _normalize_text(s: str) -> str:
    for k, v in _UNICODE_FIXES.items():
        s = s.replace(k, v)
    return s


# ---------------------------------------------------------------------------
# Validation — make sure span offsets actually index into the final answer.
# Bad offsets in training data are a silent killer: the model learns to emit
# offsets that don't correspond to any real text.
#
# Returns (ok, reason, fixed_spans). If `fix_unicode=True`, spans whose text
# matches answer[start:end] after Unicode normalization get patched and the
# row is considered valid.
# ---------------------------------------------------------------------------
def validate_row(row: dict, fix_unicode: bool = True) -> tuple[bool, str, list]:
    convs = _coerce_list(row["conversations"])
    spans = _coerce_list(row["span_labels"])
    _, answer = split_history_and_target(convs)

    if not answer:
        return False, "no final_answer turn", []

    fixed_spans = []
    for i, s in enumerate(spans):
        sub = answer[s["start"]:s["end"]]
        if sub == s["text"]:
            fixed_spans.append(dict(s))
            continue
        if fix_unicode and _normalize_text(s["text"]) == _normalize_text(sub):
            # Patch span text to match the answer exactly. Offsets stay the same.
            fixed_spans.append({"start": s["start"], "end": s["end"], "text": sub})
            continue
        # Real mismatch — either bad offsets or answer text is from a different version
        if sub == "":
            return False, (
                f"span {i} points outside answer (len={len(answer)}, "
                f"requested [{s['start']}:{s['end']}]) — likely wrong answer version"
            ), []
        return False, f"span {i} text mismatch: expected {s['text']!r}, got {sub!r}", []

    # Class-specific invariants
    cls = row["type"]
    if cls in ("clean", "undergeneration") and len(spans) > 0:
        return False, f"{cls} should have empty spans, has {len(spans)}", []
    if cls in ("answer_mismatch", "overgeneration", "missing_tool") and len(spans) == 0:
        return False, f"{cls} should have non-empty spans", []

    return True, "", fixed_spans


# ---------------------------------------------------------------------------
# Build the SFT example: messages list + meta.
# ---------------------------------------------------------------------------
def build_example(row: dict, validated_spans: list | None = None,
                  chat_template_kwargs: dict | None = None) -> dict:
    convs = _coerce_list(row["conversations"])
    spans = validated_spans if validated_spans is not None else _coerce_list(row["span_labels"])
    history, answer = split_history_and_target(convs)

    # User message: tool definitions, full conversation history up to the
    # analyzed turn, then the answer being analyzed. Clearly delimited so the
    # model can attend to each section. For single-turn dialogues, history
    # contains the one (user, tool_call, tool_response) triple. For multi-turn,
    # it contains the full prior conversation including any earlier final_answers.
    user_content = (
        "## Tool definitions (from the assistant's system prompt)\n"
        f"{row['system']}\n\n"
        "## Conversation history\n"
        f"{format_history(history)}\n\n"
        "## Assistant final answer (to analyze)\n"
        f"{answer}\n\n"
        "Analyze the final answer above for errors."
    )

    # Target: compact JSON. ensure_ascii=False so unicode passes through.
    target = {
        "type": row["type"],
        "spans": [
            {"start": int(s["start"]), "end": int(s["end"]), "text": s["text"]}
            for s in spans
        ],
    }
    target_json = json.dumps(target, ensure_ascii=False)

    out = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target_json},
        ],
        # Metadata kept separately for analysis / stratified splits.
        "dialogue_id": row.get("dialogue_id"),
        "type": row["type"],
        "label": int(row["label"]),
        "subset": row.get("subset"),
        "n_spans": len(spans),
        "n_history_turns": len(history),
    }
    # As of TRL v0.19+, SFTConfig no longer accepts a chat_template_kwargs
    # argument. Instead, TRL reads a per-row `chat_template_kwargs` column
    # and passes its contents to apply_chat_template for that row. We set the
    # same dict on every row here (e.g. {"enable_thinking": False} for Qwen3.5).
    if chat_template_kwargs:
        out["chat_template_kwargs"] = chat_template_kwargs
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_input(path: str):
    """Load a Dataset, DatasetDict, CSV, or JSON file."""
    p = Path(path)
    #if p.is_dir():
    if True:
        #loaded = load_from_disk(path)
        loaded = load_dataset(path)
        return loaded  # Could be Dataset or DatasetDict
    if p.suffix == ".csv":
        return Dataset.from_pandas(pd.read_csv(path))
    if p.suffix in (".json", ".jsonl"):
        return Dataset.from_json(path)
    raise ValueError(f"unsupported input: {path}")


def process_split(ds: Dataset, split_name: str, fix_unicode: bool, strict: bool,
                  chat_template_kwargs: dict | None = None) -> Dataset | None:
    """Validate and build SFT examples for a single split.

    Returns the processed Dataset, or None if --strict and there were failures.
    """
    print(f"\n--- Processing split '{split_name}' ({len(ds)} rows) ---")

    validated_spans_by_idx: dict[int, list] = {}
    failures: list[tuple[int, str, str]] = []
    n_unicode_fixed = 0

    for i, row in enumerate(ds):
        try:
            ok, reason, patched = validate_row(row, fix_unicode=fix_unicode)
        except Exception as e:
            ok, reason, patched = False, f"validate_row raised {type(e).__name__}: {e}", []
        if ok:
            validated_spans_by_idx[i] = patched
            orig = _coerce_list(row["span_labels"])
            if any(o["text"] != p["text"] for o, p in zip(orig, patched)):
                n_unicode_fixed += 1
        else:
            failures.append((i, str(row.get("dialogue_id", "?")), reason))

    if n_unicode_fixed:
        print(f"  Auto-fixed Unicode mismatches in {n_unicode_fixed} rows.")

    if failures:
        print(f"  {len(failures)}/{len(ds)} rows failed validation:")
        for i, did, reason in failures[:10]:
            print(f"    row {i} ({did}): {reason}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
        if strict:
            return None
        keep = sorted(validated_spans_by_idx.keys())
        ds = ds.select(keep)
        validated_spans_by_idx = {
            new_i: validated_spans_by_idx[old_i] for new_i, old_i in enumerate(keep)
        }
        print(f"  Continuing with {len(ds)} valid rows.")
    else:
        print(f"  All rows validated.")

    def _build(row, idx):
        return build_example(
            row,
            validated_spans=validated_spans_by_idx[idx],
            chat_template_kwargs=chat_template_kwargs,
        )

    formatted = ds.map(
        _build,
        with_indices=True,
        remove_columns=ds.column_names,
        desc=f"Building SFT examples ({split_name})",
    )
    return formatted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="HF DatasetDict/Dataset dir, .csv, or .jsonl")
    ap.add_argument("--output", required=True,
                    help="Output directory (HF DatasetDict/Dataset save_to_disk format)")
    ap.add_argument("--model", default=None,
                    help="Optional: tokenizer/chat-template to preview "
                         "(e.g. Qwen/Qwen3.5-9B-Instruct)")
    ap.add_argument("--strict", action="store_true",
                    help="Fail on any row that doesn't validate")
    ap.add_argument("--no-fix-unicode", action="store_true",
                    help="Disable auto-fix of curly quote / nbsp / non-breaking "
                         "hyphen mismatches between span text and answer text")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="Add chat_template_kwargs={'enable_thinking': False} to "
                         "each row. Required for Qwen3.5 training to ensure "
                         "deterministic non-thinking outputs (TRL reads this per-row "
                         "column when applying the chat template). No effect on "
                         "models without a thinking mode (Gemma 4).")
    args = ap.parse_args()

    loaded = load_input(args.input)
    fix_unicode = False #not args.no_fix_unicode
    chat_template_kwargs = {"enable_thinking": False} if args.disable_thinking else None
    if chat_template_kwargs:
        print(f"Adding chat_template_kwargs={chat_template_kwargs} to every row.")

    # Handle DatasetDict (train/test/val) vs single Dataset uniformly
    if isinstance(loaded, DatasetDict):
        print(f"Loaded DatasetDict with splits: {list(loaded.keys())}")
        out_splits: dict[str, Dataset] = {}
        for split_name, split_ds in loaded.items():
            processed = process_split(
                split_ds, split_name, fix_unicode, args.strict,
                chat_template_kwargs=chat_template_kwargs,
            )
            if processed is None:
                print(f"\nAborting due to --strict failures in split '{split_name}'.")
                return 1
            out_splits[split_name] = processed
        formatted = DatasetDict(out_splits)
    else:
        print(f"Loaded Dataset with {len(loaded)} rows")
        processed = process_split(
            loaded, "data", fix_unicode, args.strict,
            chat_template_kwargs=chat_template_kwargs,
        )
        if processed is None:
            print("\nAborting due to --strict failures.")
            return 1
        formatted = processed

    formatted.save_to_disk(args.output)
    total = (
        sum(len(s) for s in formatted.values())
        if isinstance(formatted, DatasetDict) else len(formatted)
    )
    print(f"\nWrote {total} examples to {args.output}")

    # Preview from the first split (or the single dataset)
    preview_ds = (
        next(iter(formatted.values())) if isinstance(formatted, DatasetDict) else formatted
    )
    print("\n=== Example 0 (from first split) ===")
    ex = preview_ds[0]
    print(f"meta: type={ex['type']} label={ex['label']} n_spans={ex['n_spans']} "
          f"n_history_turns={ex['n_history_turns']}")
    print(f"\nsystem ({len(ex['messages'][0]['content'])} chars): "
          f"{ex['messages'][0]['content'][:200]}...")
    print(f"\nuser ({len(ex['messages'][1]['content'])} chars):\n"
          f"{ex['messages'][1]['content'][:400]}...")
    print(f"\nassistant target:\n{ex['messages'][2]['content']}")

    if args.model:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model)
            templated = tok.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
            n_tokens = len(tok(templated)["input_ids"])
            print(f"\n=== With {args.model} chat template ===")
            print(f"Total tokens: {n_tokens}")
            print(f"First 400 chars:\n{templated[:400]}")
        except Exception as e:
            print(f"\n(Skipping chat-template preview: {e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())