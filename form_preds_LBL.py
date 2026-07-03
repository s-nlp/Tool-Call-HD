'''
python form_preds_LBL.py \
    --pred ../datasets/TH_nq_span.jsonl \
    --gold ../datasets/TOOLHACE_final.jsonl \
    --output ../datasets/TH_NQ_SPAN.csv
'''
# save as create_evaluation_csv.py
import json
import csv
import argparse

def extract_spans_from_file(file_path: str, is_prediction: bool = True) -> list:
    """
    Extract spans from JSONL file in order
    """
    spans_list = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            if is_prediction:
                # For prediction files
                if 'predicted_spans' in data:
                    spans = data['predicted_spans']
                elif 'start' in data and 'end' in data:
                    spans = [data]
                else:
                    spans = []
            else:
                # For gold file - hallucination_labels is a JSON string
                hallucination_labels = data.get('hallucination_labels', "[]")
                if isinstance(hallucination_labels, str):
                    try:
                        spans = json.loads(hallucination_labels)
                    except:
                        spans = []
                else:
                    spans = hallucination_labels if isinstance(hallucination_labels, list) else []
            
            # Convert to standardized format
            span_list = []
            for span in spans:
                if isinstance(span, dict):
                    start = span.get('start') or span.get('start_char')
                    end = span.get('end') or span.get('end_char')
                    if start is not None and end is not None:
                        span_list.append({'start': int(start), 'end': int(end)})
            
            spans_list.append(span_list)
    
    return spans_list

def spans_to_json_string(spans: list) -> str:
    """Convert spans list to JSON string"""
    if not spans:
        return "[]"
    return json.dumps(spans, separators=(',', ':'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Predictions JSONL file")
    parser.add_argument("--gold", type=str, required=True, help="Gold dataset JSONL file")
    parser.add_argument("--output", type=str, default="evaluation_spans.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    print(f"Loading predictions from: {args.pred}")
    pred_spans = extract_spans_from_file(args.pred, is_prediction=True)
    print(f"  Loaded {len(pred_spans)} entries")
    
    print(f"Loading gold spans from: {args.gold}")
    gold_spans = extract_spans_from_file(args.gold, is_prediction=False)
    print(f"  Loaded {len(gold_spans)} entries")
    
    # Ensure same length
    num_samples = min(len(pred_spans), len(gold_spans))
    if len(pred_spans) != len(gold_spans):
        print(f"Warning: Files have different lengths ({len(pred_spans)} vs {len(gold_spans)}). Using {num_samples} samples.")
    
    # Create CSV
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pred', 'gold'])
        
        for i in range(num_samples):
            writer.writerow([
                spans_to_json_string(pred_spans[i]),
                spans_to_json_string(gold_spans[i])
            ])
    
    # Statistics
    pred_nonempty = sum(1 for s in pred_spans[:num_samples] if s)
    gold_nonempty = sum(1 for s in gold_spans[:num_samples] if s)
    both = sum(1 for i in range(num_samples) if pred_spans[i] and gold_spans[i])
    
    print(f"\n{'='*50}")
    print(f"CSV CREATED: {args.output}")
    print(f"{'='*50}")
    print(f"Total samples: {num_samples}")
    print(f"  - Samples with pred spans: {pred_nonempty}")
    print(f"  - Samples with gold spans: {gold_nonempty}")
    print(f"  - Samples with both: {both}")

if __name__ == "__main__":
    main()