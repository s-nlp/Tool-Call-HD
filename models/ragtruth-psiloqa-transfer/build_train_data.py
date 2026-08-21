"""Convert RAGTruth and PsiloQA train splits to a shared ToolHACE-style schema.

Mapping:
  RAGTruth *Conflict       -> answer_mismatch
  RAGTruth *Baseless Info -> overgeneration
  PsiloQA [HAL] spans     -> answer_mismatch
  no spans                -> clean
"""

import argparse
import json
import os
import re
from collections import Counter

from datasets import Dataset, Features, Value, load_dataset

from pipeline_common import ANSWER_MAX_CHARS

try:
    from datasets import List as DList
except ImportError:
    from datasets import Sequence as DList


FEATURES = Features(
    {
        "source": Value("string"),
        "uid": Value("string"),
        "dialogue_id": Value("string"),
        "type": Value("string"),
        "label": Value("int64"),
        "available_tools": DList(Value("string")),
        "conversations": DList(
            {
                "from": Value("string"),
                "turn_role": Value("string"),
                "value": Value("string"),
            }
        ),
        "answer": Value("string"),
        "span_labels": DList(
            {
                "end": Value("int64"),
                "reason": Value("string"),
                "span_type": Value("string"),
                "start": Value("int64"),
                "text": Value("string"),
            }
        ),
        "n_spans": Value("int64"),
        "language": Value("string"),
        "source_model": Value("string"),
    }
)

HAL_TAG = re.compile(r"\[(/?)HAL\]", re.IGNORECASE)


def all_occurrences(text, needle):
    start = 0
    while needle and (position := text.find(needle, start)) >= 0:
        yield position
        start = position + 1


def locate_text(answer, text, hint=0):
    """Locate annotation text, preferring the occurrence nearest its source offset."""
    candidates = list(all_occurrences(answer, text))
    if not candidates and text != text.strip():
        text = text.strip()
        candidates = list(all_occurrences(answer, text))
    if not candidates:
        return None
    start = min(candidates, key=lambda value: abs(value - int(hint or 0)))
    return start, start + len(text)


def deduplicate(spans):
    result = []
    seen = set()
    for span in sorted(spans, key=lambda item: (item["start"], item["end"], item["span_type"])):
        key = (span["start"], span["end"], span["span_type"])
        if key not in seen:
            result.append(span)
            seen.add(key)
    return result


def row_type(spans):
    kinds = sorted({span["span_type"] for span in spans})
    if not kinds:
        return "clean"
    return kinds[0] if len(kinds) == 1 else "mixed"


def make_row(source, uid, question, context, answer, spans, language, source_model):
    return {
        "source": source,
        "uid": uid,
        "dialogue_id": uid,
        "type": row_type(spans),
        "label": int(bool(spans)),
        "available_tools": [],
        "conversations": [
            {"from": "user", "turn_role": "user", "value": question},
            {"from": "tool", "turn_role": "tool_response", "value": context},
            {"from": "assistant", "turn_role": "final_answer", "value": answer},
        ],
        "answer": answer,
        "span_labels": spans,
        "n_spans": len(spans),
        "language": language,
        "source_model": source_model,
    }


def ragtruth_class(label_type):
    value = (label_type or "").lower()
    if "conflict" in value:
        return "answer_mismatch"
    if "baseless" in value:
        return "overgeneration"
    return None


def parse_ragtruth_annotations(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        return json.loads(raw) if raw else []
    return list(raw)


def convert_ragtruth(dataset, args, stats):
    rows = []
    for example in dataset:
        if args.ragtruth_quality != "all" and example.get("quality") != args.ragtruth_quality:
            stats["ragtruth_filtered_quality"] += 1
            continue
        answer_full = example.get("output") or ""
        question = example.get("query") or ""
        context = example.get("context") or ""
        if not answer_full.strip() or not context.strip():
            stats["ragtruth_dropped_empty"] += 1
            continue

        annotations = parse_ragtruth_annotations(example.get("hallucination_labels"))
        spans = []
        unknown = False
        for annotation in annotations:
            span_type = ragtruth_class(annotation.get("label_type"))
            if not span_type:
                stats["ragtruth_unknown_label_type"] += 1
                unknown = True
                continue
            text = annotation.get("text") or ""
            start = int(annotation.get("start", -1))
            end = int(annotation.get("end", -1))
            if not (0 <= start < end <= len(answer_full)) or answer_full[start:end] != text:
                location = locate_text(answer_full, text, start)
                if location is None:
                    stats["ragtruth_unlocatable_span"] += 1
                    continue
                start, end = location
                text = answer_full[start:end]
                stats["ragtruth_repaired_span"] += 1
            if end > ANSWER_MAX_CHARS:
                stats["ragtruth_span_after_answer_cap"] += 1
                continue
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "span_type": span_type,
                    "reason": f"RAGTruth: {annotation.get('label_type', '')}",
                }
            )

        spans = deduplicate(spans)
        if annotations and not spans:
            stats["ragtruth_dropped_positive_without_visible_span"] += 1
            continue
        if unknown:
            stats["ragtruth_rows_with_unknown_labels"] += 1
        answer = answer_full[:ANSWER_MAX_CHARS]
        uid = f"ragtruth:{example.get('id', len(rows))}"
        rows.append(
            make_row(
                "ragtruth",
                uid,
                question,
                context,
                answer,
                spans,
                "en",
                example.get("model") or "",
            )
        )
        stats[f"ragtruth_kept_{row_type(spans)}"] += 1
        if args.max_ragtruth and len(rows) >= args.max_ragtruth:
            break
    return rows


def parse_hal_markers(annotated):
    """Return unmarked text and marked intervals; malformed markup returns None."""
    parts = []
    spans = []
    cursor = 0
    plain_position = 0
    open_start = None
    for match in HAL_TAG.finditer(annotated or ""):
        segment = annotated[cursor : match.start()]
        parts.append(segment)
        plain_position += len(segment)
        closing = bool(match.group(1))
        if closing:
            if open_start is None:
                return None
            spans.append((open_start, plain_position))
            open_start = None
        else:
            if open_start is not None:
                return None
            open_start = plain_position
        cursor = match.end()
    tail = (annotated or "")[cursor:]
    parts.append(tail)
    plain_position += len(tail)
    if open_start is not None:
        return None
    return "".join(parts), spans


def valid_offset_spans(answer, raw_labels):
    result = []
    for value in raw_labels or []:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        start, end = int(value[0]), int(value[1])
        if 0 <= start < end <= len(answer):
            result.append((start, end))
    return result


def align_marker_spans(answer, plain, marker_spans):
    if plain == answer:
        return marker_spans
    aligned = []
    for start, end in marker_spans:
        location = locate_text(answer, plain[start:end], start)
        if location is None:
            return None
        aligned.append(location)
    return aligned


def convert_psiloqa(dataset, args, stats):
    rows = []
    wanted_languages = None
    if args.psiloqa_languages.lower() != "all":
        wanted_languages = {
            item.strip() for item in args.psiloqa_languages.split(",") if item.strip()
        }

    for example in dataset:
        language = example.get("lang") or ""
        if wanted_languages is not None and language not in wanted_languages:
            stats["psiloqa_filtered_language"] += 1
            continue
        answer_full = example.get("llm_answer") or ""
        question = example.get("question") or ""
        context = example.get("wiki_passage") or ""
        if not answer_full.strip() or not context.strip():
            stats["psiloqa_dropped_empty"] += 1
            continue

        annotated_span = example.get("annotated_span") or ""
        has_marker_tag = bool(HAL_TAG.search(annotated_span))
        parsed = parse_hal_markers(annotated_span)
        label_spans = valid_offset_spans(answer_full, example.get("labels"))
        marker_spans = None
        marker_count = 0
        if parsed is None:
            stats["psiloqa_malformed_markers"] += 1
        else:
            plain, raw_marker_spans = parsed
            marker_count = len(raw_marker_spans)
            marker_spans = align_marker_spans(answer_full, plain, raw_marker_spans)
            if raw_marker_spans and marker_spans is None:
                stats["psiloqa_unalignable_markers"] += 1

        if marker_spans:
            chosen = marker_spans
            stats["psiloqa_used_markers"] += 1
            if label_spans and set(label_spans) != set(marker_spans):
                stats["psiloqa_marker_label_disagreement"] += 1
        elif label_spans:
            chosen = label_spans
            stats["psiloqa_used_label_fallback"] += 1
        else:
            chosen = []

        source_positive = bool(has_marker_tag or example.get("labels"))
        spans = []
        for start, end in chosen:
            if end > ANSWER_MAX_CHARS:
                stats["psiloqa_span_after_answer_cap"] += 1
                continue
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "text": answer_full[start:end],
                    "span_type": "answer_mismatch",
                    "reason": "PsiloQA: [HAL]",
                }
            )
        spans = deduplicate(spans)
        if source_positive and not spans:
            stats["psiloqa_dropped_positive_without_visible_span"] += 1
            continue

        answer = answer_full[:ANSWER_MAX_CHARS]
        uid = str(example.get("id") or f"psiloqa:{len(rows)}")
        if not uid.startswith("psiloqa"):
            uid = "psiloqa:" + uid
        rows.append(
            make_row(
                "psiloqa",
                uid,
                question,
                context,
                answer,
                spans,
                language,
                example.get("llm_checkpoint") or "",
            )
        )
        stats[f"psiloqa_kept_{row_type(spans)}"] += 1
        if args.max_psiloqa and len(rows) >= args.max_psiloqa:
            break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output train.parquet")
    parser.add_argument("--meta", default="", help="conversion JSON; default beside --out")
    parser.add_argument(
        "--source",
        choices=["ragtruth", "psiloqa"],
        required=True,
        help="source to convert; the experiment keeps the sources separate",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--psiloqa-languages", default="en", help="comma list or all")
    parser.add_argument("--ragtruth-quality", default="good", choices=["good", "all"])
    parser.add_argument("--max-ragtruth", type=int, default=0)
    parser.add_argument("--max-psiloqa", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true", help="useful for small smoke tests")
    args = parser.parse_args()

    stats = Counter()
    ragtruth_rows = []
    psiloqa_rows = []
    if args.source == "ragtruth":
        print("loading wandb/RAGTruth-processed train", flush=True)
        ragtruth = load_dataset(
            "wandb/RAGTruth-processed",
            split="train",
            cache_dir=args.cache_dir,
            streaming=args.streaming,
        )
        ragtruth_rows = convert_ragtruth(ragtruth, args, stats)
        print(f"converted RAGTruth: {len(ragtruth_rows)} rows", flush=True)

    if args.source == "psiloqa":
        print("loading s-nlp/PsiloQA train", flush=True)
        psiloqa = load_dataset(
            "s-nlp/PsiloQA",
            split="train",
            cache_dir=args.cache_dir,
            streaming=args.streaming,
        )
        psiloqa_rows = convert_psiloqa(psiloqa, args, stats)
        print(f"converted PsiloQA: {len(psiloqa_rows)} rows", flush=True)

    rows = ragtruth_rows + psiloqa_rows
    if not rows:
        raise SystemExit("conversion produced no rows")
    dataset = Dataset.from_list(rows, features=FEATURES).shuffle(seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dataset.to_parquet(args.out)

    meta_path = args.meta or os.path.join(os.path.dirname(args.out), "convert_meta.json")
    metadata = {
        "mapping": {
            "RAGTruth *Conflict": "answer_mismatch",
            "RAGTruth *Baseless Info": "overgeneration",
            "PsiloQA [HAL]": "answer_mismatch",
            "no spans": "clean",
        },
        "source": args.source,
        "psiloqa_languages": args.psiloqa_languages,
        "ragtruth_quality": args.ragtruth_quality,
        "answer_max_chars": ANSWER_MAX_CHARS,
        "counts": {
            "ragtruth": len(ragtruth_rows),
            "psiloqa": len(psiloqa_rows),
            "written": len(rows),
        },
        "stats": dict(sorted(stats.items())),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"wrote {args.out} ({len(rows)} rows)", flush=True)
    print(f"wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
