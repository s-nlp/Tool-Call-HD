"""
Convert singlehop_synthetic_toolace_converted.json (flat single-hop rows:
user_prompt, tool_call, tool_response, original_answer, system,
_source_dialogue_id, _source_dataset) into RAGTruth-style "clean" rows:

  id, query, context, output, task_type, quality, model, temperature,
  hallucination_labels, hallucination_labels_processed, input_str,
  hallucination_type, language

Output:
  singlehop_toolace_clean_ragtruth.jsonl
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "singlehop_synthetic_toolace_converted.json"
OUT = ROOT / "output" / "singlehop_toolace_clean_ragtruth.jsonl"


def main():
    with open(SRC, encoding="utf-8") as f:
        rows = json.load(f)

    out_rows = []
    for i, row in enumerate(rows, 1):
        query = row.get("user_prompt", "")
        context = row.get("tool_response", "")
        output = row.get("original_answer", "")
        out_rows.append({
            "id": i,
            "query": query,
            "context": context,
            "output": output,
            "task_type": "clean",
            "quality": "good",
            "model": "",
            "temperature": "",
            "hallucination_labels": "[]",
            "hallucination_labels_processed": json.dumps({"evident_conflict": 0, "baseless_info": 0}),
            "input_str": f"User:\n{query}\n\nTool Response:\n{context}",
            "hallucination_type": "clean",
            "language": "en",
        })

    print(f"rows: {len(out_rows)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
