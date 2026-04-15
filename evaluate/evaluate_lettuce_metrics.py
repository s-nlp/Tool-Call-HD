import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


LETTUCE_MODEL_NAME = "KRLabsOrg/lettucedect-large-modernbert-en-v1"


def flatten_json_to_text(json_str: str) -> str:
    """Превращает JSON tool_response в плоский текст."""
    try:
        obj = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return str(json_str)

    lines: list[str] = []

    def _walk(node: Any, depth: int = 0) -> None:
        pad = "  " * depth
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{pad}{key}:")
                    _walk(value, depth + 1)
                else:
                    lines.append(f"{pad}{key}: {value}")
            return

        if isinstance(node, list):
            for item in node:
                _walk(item, depth)
            return

        lines.append(f"{pad}{node}")

    _walk(obj)
    return "\n".join(lines)


def split_sentences(text: str) -> list[str]:
    """Простое разбиение ответа на предложения."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def normalize_label(label: str) -> str:
    """Приводит label модели к ожидаемым именам."""
    value = label.lower().strip()
    # LettuceDetect sometimes exposes generic HF labels instead of semantic names.
    # Empirically for this model:
    # - LABEL_0 behaves like supported
    # - LABEL_1 behaves like unsupported / not grounded
    if value == "label_0":
        return "supported"
    if value == "label_1":
        return "unsupported"
    if value in {"supported", "unsupported", "contradicted"}:
        return value
    return value


def evaluate_faithfulness(
    nli_pipe,
    question: str,
    answer: str,
    context: str,
    batch_size: int,
) -> dict[str, Any]:
    """Оценивает каждое предложение ответа через LettuceDetect."""
    sentences = split_sentences(answer)
    if not sentences:
        return {"sentences": [], "faithfulness_score": None, "label_counts": {}}

    # В premise добавляем вопрос, чтобы сохранить исходную задачу.
    premise = f"Question: {question}\n\nContext:\n{context}"
    pairs = [{"text": premise, "text_pair": sentence} for sentence in sentences]
    raw_predictions = nli_pipe(pairs, batch_size=batch_size)

    results = []
    for sentence, prediction in zip(sentences, raw_predictions):
        results.append(
            {
                "sentence": sentence,
                "label": normalize_label(prediction["label"]),
                "score": round(float(prediction["score"]), 4),
            }
        )

    labels = [item["label"] for item in results]
    supported = labels.count("supported")

    return {
        "sentences": results,
        "faithfulness_score": round(supported / len(labels), 4),
        "label_counts": {
            "supported": supported,
            "unsupported": labels.count("unsupported"),
            "contradicted": labels.count("contradicted"),
        },
    }


def calculate_metrics(evaluated_dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """Считает агрегированные метрики по всем строкам."""
    scores = [
        row["faithfulness_score"]
        for row in evaluated_dataset
        if row.get("faithfulness_score") is not None
    ]

    label_totals = {
        "supported": sum(row.get("label_counts", {}).get("supported", 0) for row in evaluated_dataset),
        "unsupported": sum(row.get("label_counts", {}).get("unsupported", 0) for row in evaluated_dataset),
        "contradicted": sum(row.get("label_counts", {}).get("contradicted", 0) for row in evaluated_dataset),
    }
    total_sentences = sum(label_totals.values())

    if not scores:
        return {
            "rows_total": len(evaluated_dataset),
            "rows_scored": 0,
            "average_score": None,
            "fully_faithful": 0,
            "low_score_lt_0_5": 0,
            "min_score": None,
            "max_score": None,
            "label_totals": label_totals,
            "total_sentences": total_sentences,
        }

    return {
        "rows_total": len(evaluated_dataset),
        "rows_scored": len(scores),
        "average_score": round(sum(scores) / len(scores), 4),
        "fully_faithful": sum(1 for score in scores if score == 1.0),
        "low_score_lt_0_5": sum(1 for score in scores if score < 0.5),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "label_totals": label_totals,
        "total_sentences": total_sentences,
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    print(f" FAITHFULNESS SUMMARY ({metrics['rows_scored']} rows)")
    print(f"  Average score        : {metrics['average_score']}")
    print(f"  Fully faithful (1.0) : {metrics['fully_faithful']}")
    print(f"  Low score  (< 0.50)  : {metrics['low_score_lt_0_5']}")
    print(f"  Min score            : {metrics['min_score']}")
    print(f"  Max score            : {metrics['max_score']}")

    total = metrics["total_sentences"]
    print()
    print(f"  Sentence label breakdown ({total} total sentences):")
    for label, count in metrics["label_totals"].items():
        percent = 100 * count / total if total else 0
        print(f"    {label:<12} : {count}  ({percent:.1f}%)")


def evaluate_dataset(
    input_path: Path,
    output_path: Path,
    summary_path: Path | None,
    model_name: str,
    answer_field: str,
    batch_size: int,
) -> None:
    """Основной проход по датасету."""
    import torch
    from tqdm.auto import tqdm
    from transformers import pipeline as hf_pipeline

    with input_path.open(encoding="utf-8") as file:
        eval_source = json.load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading NLI model: {model_name} ...")

    nli_pipe = hf_pipeline(
        task="text-classification",
        model=model_name,
        device=0 if device == "cuda" else -1,
        truncation=True,
        max_length=512,
    )
    print("Model loaded.")

    evaluated_dataset = []
    for row in tqdm(eval_source, desc="Evaluating", unit="row"):
        new_row = deepcopy(row)
        context = flatten_json_to_text(row["tool_response"])

        eval_result = evaluate_faithfulness(
            nli_pipe=nli_pipe,
            question=row["user_prompt"],
            answer=row[answer_field],
            context=context,
            batch_size=batch_size,
        )

        new_row["faithfulness_eval"] = eval_result["sentences"]
        new_row["faithfulness_score"] = eval_result["faithfulness_score"]
        new_row["label_counts"] = eval_result["label_counts"]
        evaluated_dataset.append(new_row)

    metrics = calculate_metrics(evaluated_dataset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(evaluated_dataset, file, ensure_ascii=False, indent=2)

    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"Evaluation complete: {len(evaluated_dataset)} rows")
    print_metrics(metrics)
    print(f"Saved evaluated dataset: {output_path}")
    if summary_path:
        print(f"Saved metrics summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate faithfulness with LettuceDetect and calculate metrics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset_v3_corrupted.json"),
        help="Path to source dataset JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_v3_evaluated.json"),
        help="Path for evaluated dataset JSON.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("dataset_v3_metrics.json"),
        help="Path for aggregate metrics JSON. Use empty string to skip.",
    )
    parser.add_argument(
        "--model",
        default=LETTUCE_MODEL_NAME,
        help="Hugging Face model name for LettuceDetect.",
    )
    parser.add_argument(
        "--answer-field",
        default="original_answer",
        help="Dataset field with answer text.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for sentence-pair classification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.summary_output if str(args.summary_output) else None
    evaluate_dataset(
        input_path=args.input,
        output_path=args.output,
        summary_path=summary_path,
        model_name=args.model,
        answer_field=args.answer_field,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
