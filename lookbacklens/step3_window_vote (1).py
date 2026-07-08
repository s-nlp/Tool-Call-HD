"""
step3_eval_spans_voting.py - Modified version using voting approach instead of padding
FIXED: Correctly converts factual probabilities to hallucination probabilities

python step3_window_vote.py \
    --lookback_ratio_file ../datasets/lookback_ratio_TOOLHACE_FINAL.pt \
    --classifier_file classifiers/classifier_anno-nq-7b_sliding_window_8.pkl \
    --auth_token 'INSERT_THE_TOKEN_HERE' \
    --output_file ../datasets/TH_nq_window.jsonl \
    --tokenizer_name meta-llama/Llama-2-7b-chat-hf
"""

import torch
import pickle
import json
import numpy as np
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer

def convert_to_token_level_predictions_voting(lookback_tensor, classifier, threshold, sliding_window=8):
    """
    Convert lookback ratios to token-level hallucination predictions using voting.
    
    Each token gets the majority vote of all windows that contain it.
    pred=1 means factual, pred=0 means hallucinated.
    """
    all_predictions = []
    all_probabilities = []  # This will store factual probabilities internally
    
    for idx in tqdm(range(len(lookback_tensor)), desc="Processing examples"):
        example = lookback_tensor[idx]
        num_layers, num_heads, num_new_tokens = example.shape
        
        # Step 1: Classify each window
        window_predictions = []
        window_probabilities = []
        
        # We need windows starting from 0 to num_new_tokens - sliding_window
        for start in range(0, num_new_tokens - sliding_window + 1):
            end = start + sliding_window - 1
            window_features = example[:, :, start:end+1]
            window_features_flat = window_features.view(-1, sliding_window).mean(dim=1)
            feature_vector = window_features_flat.numpy().reshape(1, -1)
            
            # Get factual probability from classifier
            pred_proba = classifier.predict_proba(feature_vector)[0, 1]
            pred = 1 if pred_proba > threshold else 0
            
            window_predictions.append(pred)
            window_probabilities.append(pred_proba)
        
        # Step 2: For each token, collect votes from all windows containing it
        token_votes = [[] for _ in range(num_new_tokens)]
        token_probs = [[] for _ in range(num_new_tokens)]
        
        for start, (pred, prob) in enumerate(zip(window_predictions, window_probabilities)):
            for token_idx in range(start, start + sliding_window):
                if token_idx < num_new_tokens:
                    token_votes[token_idx].append(pred)
                    token_probs[token_idx].append(prob)
        
        # Step 3: Determine token-level predictions via majority voting
        token_predictions = []
        token_probabilities = []  # This will store factual probabilities
        
        for token_idx in range(num_new_tokens):
            if token_votes[token_idx]:
                # Majority vote (more than half vote factual = 1)
                factual_votes = sum(1 for v in token_votes[token_idx] if v == 1)
                total_votes = len(token_votes[token_idx])
                
                if factual_votes > total_votes / 2:
                    token_pred = 1  # factual
                else:
                    token_pred = 0  # hallucinated
                
                # Average factual probability
                token_prob = sum(token_probs[token_idx]) / len(token_probs[token_idx])
            else:
                # Edge case: token not in any window (shouldn't happen with proper overlap)
                token_pred = 0  # default to hallucinated (conservative)
                token_prob = 0.0  # factual probability = 0 means hallucinated confidence = 1
            
            token_predictions.append(token_pred)
            token_probabilities.append(token_prob)
        
        all_predictions.append(token_predictions)
        all_probabilities.append(token_probabilities)
    
    return all_predictions, all_probabilities

def convert_predictions_to_spans(predictions, probabilities, min_span_length=3, threshold=0.5, merge_gap=1):
    """
    Convert token-level predictions to aggregated spans.
    pred=1 means factual, pred=0 means hallucinated.
    probabilities should be hallucination probabilities (not factual).
    """
    spans = []
    i = 0
    while i < len(predictions):
        # Look for hallucinated tokens (pred=0) with hallucination probability >= threshold
        if predictions[i] == 0 and probabilities[i] >= threshold:
            start = i
            # Find end of span (allow small gaps)
            j = i
            consecutive_zeros = 0
            while j < len(predictions):
                if predictions[j] == 0 and probabilities[j] >= threshold:
                    consecutive_zeros = 0
                    j += 1
                elif consecutive_zeros < merge_gap:
                    consecutive_zeros += 1
                    j += 1
                else:
                    break
            
            end = j - 1 - consecutive_zeros
            
            # Only add if span length meets minimum
            if end - start + 1 >= min_span_length:
                spans.append({
                    'start_token': start,
                    'end_token': end,
                    'probability': float(np.mean(probabilities[start:end+1]))
                })
            i = j
        else:
            i += 1
    return spans

def convert_token_spans_to_char_spans(text, tokenizer, token_spans, model_completion_ids):
    """
    Convert token-level spans to character-level spans using tokenizer offset mapping
    """
    if not token_spans:
        return []
    
    # Tokenize with offset mapping to get character positions
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_offsets = encoded['offset_mapping']
    
    char_spans = []
    for span in token_spans:
        start_token = span['start_token']
        end_token = span['end_token']
        
        # Ensure indices are within bounds
        if start_token >= len(token_offsets) or end_token >= len(token_offsets):
            print(f"Warning: Token span {start_token}-{end_token} out of range (max {len(token_offsets)})")
            continue
        
        # Get character positions
        start_char = token_offsets[start_token][0]
        end_char = token_offsets[end_token][1]
        
        # Extract the exact text
        span_text = text[start_char:end_char]
        
        char_spans.append({
            'start': start_char,
            'end': end_char,
            'text': span_text,
            'probability': span['probability'],
            'start_token': start_token,
            'end_token': end_token
        })
    
    return char_spans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback_ratio_file", type=str, required=True)
    parser.add_argument("--classifier_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="hallucination_predictions.jsonl")
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--sliding_window", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min_span_length", type=int, default=3)
    parser.add_argument("--merge_gap", type=int, default=1)
    parser.add_argument("--auth_token", type=str, required = True)
    
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
        
        # Use best_threshold from pickle if available
        if args.threshold is None:
            if 'best_threshold' in classifier_data:
                args.threshold = classifier_data['best_threshold']
                print(f"Using best_threshold from classifier file: {args.threshold}")
            else:
                args.threshold = 0.5
                print(f"Warning: No best_threshold found. Using default: {args.threshold}")
        else:
            print(f"Using manually specified threshold: {args.threshold}")
    
    # Get token-level predictions using VOTING (not padding)
    print("Computing token-level predictions with voting...")
    token_predictions, token_probabilities_factual = convert_to_token_level_predictions_voting(
        lookback_tensors, classifier, args.threshold, args.sliding_window
    )
    
    # ✅ FIX 1: Convert factual probabilities to hallucination probabilities
    print("Converting factual probabilities to hallucination probabilities...")
    token_probabilities_hallucination = []
    for probs in token_probabilities_factual:
        token_probabilities_hallucination.append([1 - p for p in probs])
    
    # ✅ FIX 2: For overall predictions, use hallucination probabilities
    overall_probs = [max(probs) if probs else 0.0 for probs in token_probabilities_hallucination]
    overall_preds = [1 if prob >= args.threshold else 0 for prob in overall_probs]
    # Now overall_pred=1 means hallucinated, overall_pred=0 means factual
    
    # ✅ FIX 3: Convert to spans using hallucination probabilities
    print("Converting to character-level spans...")
    all_char_spans = []
    
    for idx, (completion, token_preds, token_probs_hall, completion_ids) in enumerate(
        zip(model_completions, token_predictions, token_probabilities_hallucination, model_completion_ids_list)
    ):
        # Aggregate token predictions into spans using hallucination probabilities
        token_spans = convert_predictions_to_spans(
            token_preds, token_probs_hall, 
            args.min_span_length, args.threshold, args.merge_gap
        )
        
        # Convert token spans to character spans
        char_spans = convert_token_spans_to_char_spans(
            completion, tokenizer, token_spans, completion_ids
        )
        all_char_spans.append(char_spans)
    
    # Save predictions
    print(f"Saving predictions to: {args.output_file}")
    with open(args.output_file, 'w') as f:
        for data_idx, completion, full_input, overall_prob, overall_pred, token_preds, token_probs_hall, char_spans in zip(
            data_indices, model_completions, full_input_texts, 
            overall_probs, overall_preds, token_predictions, token_probabilities_hallucination, all_char_spans
        ):
            output = {
                'data_index': data_idx,
                'model_completion': completion,
                'full_input_text': full_input,
                'overall_hallucination_probability': float(overall_prob),
                'overall_is_hallucination': bool(overall_pred),  # Now 1=hallucinated
                'token_predictions': token_preds,
                'token_probabilities': [float(p) for p in token_probs_hall],  # Now hallucination probabilities
                'predicted_spans': char_spans
            }
            f.write(json.dumps(output) + '\n')
    
    # Print statistics
    print("\n" + "="*50)
    print("PREDICTION STATISTICS:")
    print("="*50)
    print(f"Total examples: {len(lookback_tensors)}")
    print(f"Examples predicted as hallucinated: {sum(overall_preds)} ({sum(overall_preds)/len(overall_preds)*100:.2f}%)")
    print(f"Average hallucination probability: {np.mean(overall_probs):.4f}")
    
    total_spans = sum(len(spans) for spans in all_char_spans)
    print(f"Total predicted character spans: {total_spans}")
    print(f"Examples with at least one span: {sum(1 for spans in all_char_spans if spans)}")
    
    # Example output
    if len(all_char_spans) > 0 and all_char_spans[0]:
        print(f"\nExample span from first prediction:")
        print(f"  Text: {all_char_spans[0][0]['text'][:100]}...")
        print(f"  Character range: {all_char_spans[0][0]['start']}-{all_char_spans[0][0]['end']}")
        print(f"  Probability: {all_char_spans[0][0]['probability']:.4f}")
    
    print(f"\nPredictions saved to: {args.output_file}")

if __name__ == "__main__":
    main()