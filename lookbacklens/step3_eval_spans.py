'''
python step3_eval_spans.py \
    --lookback_ratio_file ../datasets/lookback_ratio_TOOLHACE_FINAL.pt \
    --classifier_file classifiers/classifier_anno-cnndm-7b_predefined_span.pkl \
    --output_file ../datasets/TH_cnndm_span.jsonl \
    --tokenizer_name meta-llama/Llama-2-7b-chat-hf \
    --auth_token 'INSERT_TOKEN_NAME' \
    --max_span_length 50 \
    --merge_threshold 2
'''

# save as evaluate_predefined_span.py
import torch
import pickle
import json
import numpy as np
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer

def extract_span_features(lookback_tensor, start_token, end_token):
    """
    Extract features for a specific span from the lookback ratio tensor.
    Returns features of shape (num_layers * num_heads,) = 1024 for 7B model.
    """
    # Extract the span: shape [num_layers, num_heads, span_length]
    span_data = lookback_tensor[:, :, start_token:end_token+1]
    
    # Average over the span length (token dimension)
    # Result shape: [num_layers, num_heads]
    span_avg = span_data.mean(dim=2)
    
    # Flatten to 1D: [num_layers * num_heads]
    feature_vector = span_avg.flatten().numpy()
    
    return feature_vector.reshape(1, -1)

def generate_candidate_spans(num_tokens, max_span_length=50):
    """
    Generate candidate spans of varying lengths
    """
    candidates = []
    for start in range(num_tokens):
        # Try different span lengths
        for length in range(1, min(max_span_length, num_tokens - start) + 1):
            end = start + length - 1
            candidates.append((start, end))
    
    # Limit number of candidates for long sequences (sample evenly)
    if len(candidates) > 5000:
        step = len(candidates) // 5000
        candidates = candidates[::step]
    
    return candidates

def merge_overlapping_spans(spans, merge_threshold=2):
    """
    Merge overlapping or nearby spans
    """
    if not spans:
        return []
    
    # Sort by start token
    spans.sort(key=lambda x: x['start_token'])
    
    merged = []
    current = spans[0].copy()
    
    for span in spans[1:]:
        # If spans overlap or are very close, merge them
        if span['start_token'] <= current['end_token'] + merge_threshold:
            # Merge: take min start, max end, average probability
            current['end_token'] = max(current['end_token'], span['end_token'])
            current['probability'] = (current['probability'] + span['probability']) / 2
        else:
            merged.append(current)
            current = span.copy()
    
    merged.append(current)
    return merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback_ratio_file", type=str, required=True)
    parser.add_argument("--classifier_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="hallucination_predictions_predefined.jsonl")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max_span_length", type=int, default=50)
    parser.add_argument("--merge_threshold", type=int, default=2)
    parser.add_argument("--auth_token", type=str, required=True)
    
    args = parser.parse_args()
    
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, token=args.auth_token)
    
    print(f"Loading lookback ratios from: {args.lookback_ratio_file}")
    lookback_data = torch.load(args.lookback_ratio_file)
    
    # Extract data
    lookback_tensors = []
    model_completions = []
    model_completion_ids_list = []
    data_indices = []
    full_input_texts = []
    
    for item in lookback_data:
        lookback_tensors.append(item['lookback_ratio'])
        model_completions.append(item['model_completion'])
        model_completion_ids_list.append(item['model_completion_ids'])
        data_indices.append(item['data_index'])
        full_input_texts.append(item['full_input_text'])
    
    print(f"Loaded {len(lookback_tensors)} examples")
    
    # Load classifier
    print(f"Loading classifier from: {args.classifier_file}")
    with open(args.classifier_file, 'rb') as f:
        classifier_data = pickle.load(f)
        classifier = classifier_data['clf']

        # Use best_threshold from pickle if available, otherwise use 0.5 as fallback
        if args.threshold is None:
            # Try to get best_threshold from the pickle
            if 'best_threshold' in classifier_data:
                args.threshold = classifier_data['best_threshold']
                print(f"Using best_threshold from classifier file: {args.threshold}")
            else:
                args.threshold = 0.5
                print(f"Warning: No best_threshold found in classifier file. Using default: {args.threshold}")
        else:
            print(f"Using manually specified threshold: {args.threshold}")
    
    print(f"Classifier expects {classifier.n_features_in_} features")
    
    all_char_spans = []
    all_overall_probs = []
    
    for idx, (lookback, completion, completion_ids) in enumerate(tqdm(
        zip(lookback_tensors, model_completions, model_completion_ids_list), 
        desc="Processing examples", total=len(lookback_tensors)
    )):
        num_tokens = lookback.shape[-1]  # Number of tokens in the completion
        
        # Skip if no tokens
        if num_tokens == 0:
            all_char_spans.append([])
            all_overall_probs.append(0.0)
            continue
        
        # Generate candidate spans
        candidates = generate_candidate_spans(num_tokens, args.max_span_length)
        
        # Classify each candidate span
        span_predictions = []
        
        for start_token, end_token in candidates:
            # Extract features for this span
            features = extract_span_features(lookback, start_token, end_token)
            
            # Verify feature dimension
            if features.shape[1] != classifier.n_features_in_:
                print(f"Warning: Feature dimension mismatch. Got {features.shape[1]}, expected {classifier.n_features_in_}")
                continue
            
            # Predict hallucination probability for this span
            factual_prob = classifier.predict_proba(features)[0, 1]
            hallucination_prob = 1 - factual_prob
            
            if hallucination_prob >= args.threshold:
                span_predictions.append({
                    'start_token': start_token,
                    'end_token': end_token,
                    'probability': float(hallucination_prob)
                })
        
        # Merge overlapping spans
        merged_spans = merge_overlapping_spans(span_predictions, args.merge_threshold)
        
        # Convert to character-level spans
        char_spans = []
        if merged_spans:
            # Tokenize with offset mapping for character positions
            encoded = tokenizer(completion, return_offsets_mapping=True, add_special_tokens=False)
            token_offsets = encoded['offset_mapping']
            
            for span in merged_spans:
                start_token = span['start_token']
                end_token = span['end_token']
                
                if start_token < len(token_offsets) and end_token < len(token_offsets):
                    start_char = token_offsets[start_token][0]
                    end_char = token_offsets[end_token][1]
                    span_text = completion[start_char:end_char]
                    
                    char_spans.append({
                        'start': start_char,
                        'end': end_char,
                        'text': span_text,
                        'probability': span['probability'],
                        'start_token': start_token,
                        'end_token': end_token
                    })
        
        all_char_spans.append(char_spans)
        
        # Calculate overall hallucination probability for the example
        if span_predictions:
            all_overall_probs.append(max(p['probability'] for p in span_predictions))
        else:
            all_overall_probs.append(0.0)
        
        # Progress update for first few samples
        if idx < 3:
            print(f"Sample {idx}: {len(candidates)} candidates, {len(span_predictions)} predictions, {len(char_spans)} merged spans")
    
    # Save predictions
    print(f"Saving predictions to: {args.output_file}")
    with open(args.output_file, 'w') as f:
        for idx, (data_idx, completion, full_input, overall_prob, char_spans) in enumerate(
            zip(data_indices, model_completions, full_input_texts, all_overall_probs, all_char_spans)
        ):
            output = {
                'data_index': data_idx,
                'model_completion': completion,
                'full_input_text': full_input,
                'overall_hallucination_probability': float(overall_prob),
                'overall_is_hallucination': bool(overall_prob >= args.threshold),
                'predicted_spans': char_spans
            }
            f.write(json.dumps(output) + '\n')
    
    # Print statistics
    print("\n" + "="*50)
    print("PREDICTION STATISTICS:")
    print("="*50)
    print(f"Total examples: {len(lookback_tensors)}")
    hallucinated_examples = sum(1 for prob in all_overall_probs if prob >= args.threshold)
    print(f"Examples predicted as hallucinated: {hallucinated_examples} ({hallucinated_examples/len(all_overall_probs)*100:.2f}%)")
    print(f"Average hallucination probability: {np.mean(all_overall_probs):.4f}")
    
    total_spans = sum(len(spans) for spans in all_char_spans)
    print(f"Total predicted character spans: {total_spans}")
    print(f"Examples with at least one span: {sum(1 for spans in all_char_spans if spans)}")
    
    # Example output
    if all_char_spans and all_char_spans[0]:
        print(f"\nExample span from first prediction:")
        print(f"  Text: {all_char_spans[0][0]['text'][:100]}...")
        print(f"  Character range: {all_char_spans[0][0]['start']}-{all_char_spans[0][0]['end']}")
        print(f"  Probability: {all_char_spans[0][0]['probability']:.4f}")
    
    print(f"\nPredictions saved to: {args.output_file}")

if __name__ == "__main__":
    main()
