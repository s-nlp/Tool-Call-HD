"""
Predict hallucination spans using etomoscow/tool-calling-hallucination-modernbert-large-crf-best.

Input:  9k_sample_lettuce.parquet (or any parquet with query/context/output columns)
Output: predictions JSONL + updated parquet with span_labels filled in

Usage:
    python3 predict_spans.py
    python3 predict_spans.py --input my_data.parquet --output my_preds.jsonl
    python3 predict_spans.py --batch-size 8  # reduce if OOM
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lettucedetect.models.inference import HallucinationDetector

MODEL_ID = "etomoscow/tool-calling-hallucination-modernbert-large-crf-best"


def predict_all(detector, questions, contexts, answers):
    all_spans = []
    skipped = 0
    for i in tqdm(range(len(answers)), desc="Predicting"):
        try:
            preds = detector.predict(
                context=[contexts[i]],
                question=questions[i],
                answer=answers[i],
                output_format="spans",
            )
            spans = []
            for p in preds:
                spans.append({
                    "start": p.get("start", 0),
                    "end": p.get("end", 0),
                    "text": p.get("text", ""),
                    "confidence": p.get("confidence", 0.0),
                    "label_type": p.get("category", "hallucination"),
                })
            all_spans.append(spans)
        except Exception:
            # context+answer too long for model — truncate context and retry
            try:
                ctx = contexts[i]
                max_ctx = 4000
                if len(ctx) > max_ctx:
                    ctx = ctx[:max_ctx]
                preds = detector.predict(
                    context=[ctx],
                    question=questions[i][:500],
                    answer=answers[i],
                    output_format="spans",
                )
                spans = [{"start": p.get("start", 0), "end": p.get("end", 0),
                          "text": p.get("text", ""), "confidence": p.get("confidence", 0.0),
                          "label_type": p.get("category", "hallucination")} for p in preds]
                all_spans.append(spans)
            except Exception:
                all_spans.append([])
                skipped += 1
    if skipped:
        print(f"  Skipped {skipped} records (too long even after truncation)")
    return all_spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="9k_sample_lettuce.parquet")
    ap.add_argument("--output", default="9k_sample_predictions.jsonl")
    ap.add_argument("--output-parquet", default="9k_sample_with_spans.parquet")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--hf-token", default=None, help="HF token for private repos (or set HF_TOKEN env var)")
    args = ap.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)
        print("Logged in to HF Hub")

    print(f"Loading model: {args.model}")
    detector = HallucinationDetector(method="transformer", model_path=args.model)

    print(f"Loading data: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  {len(df)} records")

    questions = df['query'].tolist()
    contexts = df['context'].tolist()
    answers = df['output'].tolist()

    print(f"Running inference...")
    all_preds = predict_all(detector, questions, contexts, answers)

    # save predictions JSONL
    with open(args.output, 'w', encoding='utf-8') as f:
        for i, (_, row) in enumerate(df.iterrows()):
            f.write(json.dumps({
                'dialogue_id': row.get('dialogue_id', ''),
                'query': row['query'][:200],
                'pred_spans': all_preds[i],
                'n_spans': len(all_preds[i]),
            }, ensure_ascii=False) + '\n')
    print(f"Predictions saved → {args.output}")

    # update parquet with predicted spans
    import numpy as np
    span_labels = []
    hal_labels = []
    labels = []
    for spans in all_preds:
        span_labels.append(np.array(spans, dtype=object))
        hal_labels.append(json.dumps(spans, ensure_ascii=False))
        labels.append(1 if len(spans) > 0 else 0)

    df['span_labels'] = span_labels
    df['hallucination_labels'] = hal_labels
    df['label'] = labels

    df.to_parquet(args.output_parquet, index=False)

    n_hal = sum(1 for s in all_preds if len(s) > 0)
    n_clean = sum(1 for s in all_preds if len(s) == 0)
    total_spans = sum(len(s) for s in all_preds)
    print(f"\nResults:")
    print(f"  Hallucinated: {n_hal} ({100*n_hal/len(df):.1f}%)")
    print(f"  Clean:        {n_clean} ({100*n_clean/len(df):.1f}%)")
    print(f"  Total spans:  {total_spans}")
    print(f"  Avg spans:    {total_spans/len(df):.2f}")
    print(f"\nSaved → {args.output_parquet}")


if __name__ == "__main__":
    main()
