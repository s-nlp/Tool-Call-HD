# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
'''
python step01.py \
    --model-name meta-llama/Llama-2-7b-chat-hf \
    --data-path ../datasets/TOOLHACE_final.jsonl \
    --output-path ../datasets/lookback_ratio_TOOLHACE_FINAL.pt \
    --teacher-forcing-jsonl ../datasets/TEAFOR_TOOLHACE_final.jsonl \
    --auth-token 'INSERT_THE_TOKEN_HERE' \
    --custom-dataset \
    --num-gpus 1 \
    --max-memory 15 \
    --max-new-tokens 408 \
    --debug
'''

import os
import json
import random
import torch
import numpy as np
import transformers
from tqdm import tqdm
import argparse
import tiktoken
import gc
from models.lookbacklens.generation import LLM

transformers.logging.set_verbosity(40)

# ============================================
# GPU CONFIGURATION - SET VISIBLE DEVICES
# ============================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ============================================

data_response_names = {
    'nq': 'Answer',
    'xsum': 'Summary',
    'cnndm': 'Summary',
}

def num_tokens_from_message(message, model="davinci"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(message))
    return num_tokens

def truncate_message(prompt1, prompt2, model="davinci"):
    if num_tokens_from_message(prompt1 + prompt2, model) > 2033:
        truncation_length = 2033 - num_tokens_from_message(prompt2)
        while num_tokens_from_message(prompt1) > truncation_length:
            prompt1 = " ".join(prompt1.split(' ')[:-1])
    prompt = prompt1 + prompt2
    return prompt

def load_nq_open(file_path, parallel=False, total_shard=8, shard_id=0, debug=False, data_type='nq_open', subsample=None):
    list_data_dict = []
    is_train = 'nq_train' in file_path
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        if debug:
            data = data[:10]
        if subsample is not None:
            data = [data[i] for i in range(len(data)) if i % subsample == 0]
        if parallel:
            chunk_size = len(data) // total_shard
            data = data[shard_id * chunk_size: (shard_id + 1) * chunk_size]

        for idx in range(len(data)):
            data_index = idx
            question = data[idx]['question']
            question = question[0].upper() + question[1:]
            if question[-1] != '?':
                question += '?'
            answers = data[idx]['answers']
            if is_train:
                pos_ctxs = data[idx]['positive_ctxs']
                neg_ctxs = data[idx]['negative_ctxs']
            else:
                ctxs = data[idx]['ctxs']
                pos_ctxs = [ctx for ctx in ctxs if ctx['hasanswer']]
                neg_ctxs = [ctx for ctx in ctxs if not ctx['hasanswer']]
            assert len(pos_ctxs) > 0, "No positive context found."
            assert len(neg_ctxs) >= 2, "At least two negative contexts are required."
            context = f"#Document#: " + neg_ctxs[0]['text'] + '\n' + pos_ctxs[0]['text'] + '\n' + neg_ctxs[1]['text']
            context += f"\n#Question#: {question}"
            response = f"\n#Answer#:"
            new_item = dict(
                context=context,
                response=response,
                net_response=None,
                answer=answers[0],
                data_index=data_index
            )
            list_data_dict.append(new_item)
    return list_data_dict

def load_summarization(file_path, parallel=False, total_shard=8, shard_id=0, debug=False, data_type='cnndm', subsample=None):
    list_data_dict = []
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        data_indices = []
        data_index = 0
        for line in f:
            data.append(json.loads(line))
            data_indices.append(data_index)
            data_index += 1
        if debug:
            data = data[:10]
            data_indices = data_indices[:10]
        if subsample is not None:
            data = [data[i] for i in range(len(data)) if i % subsample == 0]
            data_indices = [data_indices[i] for i in range(len(data_indices)) if i % subsample == 0]
        if parallel:
            chunk_size = len(data) // total_shard
            data = data[shard_id * chunk_size: (shard_id + 1) * chunk_size]
            data_indices = data_indices[shard_id * chunk_size: (shard_id + 1) * chunk_size]

        for idx in range(len(data)):
            data_index = data_indices[idx]
            context = "#Document#: " if data_type == 'cnndm' else "#Article#: "
            context += data[idx]['document']
            response = f"\n#Summary#:"
            new_item = dict(
                context=context,
                response=response,
                net_response=None,
                answer=data[idx].get('summary', ''),
                data_index=data_index
            )
            list_data_dict.append(new_item)
    return list_data_dict

def load_custom_dataset(file_path, parallel=False, total_shard=8, shard_id=0, debug=False, data_type='nq', subsample=None):
    list_data_dict = []
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        if debug:
            data = data[:10]
        if subsample is not None:
            data = [data[i] for i in range(len(data)) if i % subsample == 0]
        if parallel:
            chunk_size = len(data) // total_shard
            data = data[shard_id * chunk_size: (shard_id + 1) * chunk_size]

        for idx in range(len(data)):
            data_index = idx
            query = data[idx].get('query', '')
            context = data[idx].get('context', '')
            output = data[idx].get('output', '')
            
            formatted_context = f"#Document#: {context}\n#Question#: {query}"
            response = f"\n#Answer#:"
            
            new_item = dict(
                context=formatted_context,
                response=response,
                net_response=None,
                answer=output,
                data_index=data_index
            )
            list_data_dict.append(new_item)
    return list_data_dict

def create_demo_text(pondering=None, data_type='cnndm'):
    if data_type == 'cnndm':
        return "Generate a summary based on the information in the document.\n\n"
    elif data_type == 'nq':
        return "Answer the question based on the information in the document. Explain your reasoning in the document step-by-step before providing the final answer.\n\n"
    elif data_type == 'xsum':
        return "Generate a summary comprising of 1 sentence for the given article.\n\n"
    else:
        raise ValueError("Please specify the data type.")

def build_prompt(context, response, pondering=None, data_type='cnndm'):
    demo = create_demo_text(pondering, data_type)
    prompt = demo + context
    if data_type == 'cnndm' or data_type == 'xsum':
        input_text_prompt = truncate_message(prompt, response)
    else:
        input_text_prompt = prompt + response
    return input_text_prompt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--num-gpus", type=int, default=3)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-path", type=str, default="data/cnndm-1000.jsonl")
    parser.add_argument("--output-path", type=str, default="lookback-ratio-cnndm-7b.pt")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auth-token", type=str, required = True)
    parser.add_argument("--data-type", type=str, default=None)
    parser.add_argument("--teacher-forcing-jsonl", type=str, default=None)
    parser.add_argument("--max-memory", type=int, default=10)
    parser.add_argument("--custom-dataset", action="store_true")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    
    forced_truncate = ('gpt2' in args.model_name)
    if args.data_type is None:
        if 'cnndm' in args.data_path:
            args.data_type = 'cnndm'
        elif 'nq-open' in args.data_path:
            args.data_type = 'nq'
        elif 'xsum' in args.data_path:
            args.data_type = 'xsum'
        else:
            args.data_type = 'nq'
            print(f"Data type not detected, defaulting to: {args.data_type}")
    
    fp = args.data_path
    if not os.path.exists(fp):
        raise ValueError(f"Test file {fp} does not exist.")

    if args.custom_dataset:
        print(f"Using custom dataset loader for: {fp}")
        list_data_dict = load_custom_dataset(fp, parallel=args.parallel, total_shard=args.total_shard, shard_id=args.shard_id, debug=args.debug, subsample=args.subsample, data_type=args.data_type)
    elif "nq-open" in fp:
        print(f"Using NQ dataset loader for: {fp}")
        list_data_dict = load_nq_open(fp, parallel=args.parallel, total_shard=args.total_shard, shard_id=args.shard_id, debug=args.debug, subsample=args.subsample)
    else:
        print(f"Using summarization dataset loader for: {fp}")
        list_data_dict = load_summarization(fp, parallel=args.parallel, total_shard=args.total_shard, shard_id=args.shard_id, debug=args.debug, data_type=args.data_type, subsample=args.subsample)
    
    print(f"Loaded {len(list_data_dict)} examples")

    llm = LLM(
        model_name, device, num_gpus, 
        auth_token=args.auth_token, 
        max_memory=args.max_memory)
    
    stop_word_list = ["#Document#:", "#Question#:", "#Article#:", "Q:", "\end{code}"]
    llm.set_stop_words(stop_word_list)
    
    teacher_forcing_dict = {}
    if args.teacher_forcing_jsonl is not None:
        if not os.path.exists(args.teacher_forcing_jsonl):
            raise ValueError(f"Teacher forcing file {args.teacher_forcing_jsonl} does not exist.")
        with open(args.teacher_forcing_jsonl, 'r') as f:
            for line in f:
                data = json.loads(line)
                teacher_forcing_dict[data['data_index']] = data['model_completion_ids']
        print(f"Loaded teacher forcing data for {len(teacher_forcing_dict)} examples")

    to_save_list = []
    
    response_template = f"\n#{data_response_names[args.data_type]}#:"
    extra_prompt_length = len(llm.tokenizer(response_template)['input_ids']) - 1
    if extra_prompt_length < 0:
        extra_prompt_length = 0
    print(f"Extra prompt length: {extra_prompt_length}")
    
    for idx in tqdm(range(len(list_data_dict)), desc="Processing examples"):
        sample = list_data_dict[idx]

        # Create teacher forcing tensor on the same device as the model
        teacher_forcing_ids = None
        if args.teacher_forcing_jsonl is not None:
            if sample['data_index'] not in teacher_forcing_dict:
                print(f"Warning: No teacher forcing data for index {sample['data_index']}, skipping")
                continue
            model_device = next(llm.model.parameters()).device
            token_list = teacher_forcing_dict[sample['data_index']]
            teacher_forcing_ids = torch.tensor([token_list], device=model_device, dtype=torch.long)
            if args.debug:
                print(f"Created teacher_forcing_ids on device: {teacher_forcing_ids.device}")
        
        input_text = build_prompt(sample['context'], response_template, data_type=args.data_type)

        # Clear cache before generation
        torch.cuda.empty_cache()
        gc.collect()
        
        # Tokenize input
        input_ids = llm.tokenizer(input_text, return_tensors="pt").input_ids.to(model_device)
        
        if args.debug:
            print(f"Input IDs device: {input_ids.device}")
            print(f"Input IDs shape: {input_ids.shape}")
        
        max_len = input_ids.shape[-1] + args.max_new_tokens
        
        # BYPASS LLM.generate() wrapper - call model.generate directly
        outputs = llm.model.generate(
            inputs=input_ids,
            max_length=max_len,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            stopping_criteria=llm.stopping_criteria,
            output_attentions=True,
            teacher_forcing_seq=teacher_forcing_ids
        )
        
        # ============================================
        # DEBUGGING: Comprehensive attention analysis
        # ============================================
        if args.debug:
            print(f"\n{'='*70}")
            print(f"ATTENTION ANALYSIS - Sample {idx}")
            print(f"{'='*70}")
            
            attentions = outputs.attentions
            
            # Basic structure
            print(f"\n[1] ATTENTION STRUCTURE:")
            print(f"    Type: {type(attentions)}")
            print(f"    Length (tokens): {len(attentions)}")
            
            if len(attentions) > 0:
                print(f"    First token attention type: {type(attentions[0])}")
                print(f"    First token attention layers: {len(attentions[0])}")
                
                if len(attentions[0]) > 0:
                    first_attn = attentions[0][0]
                    print(f"\n[2] FIRST LAYER, FIRST TOKEN:")
                    print(f"    Shape: {first_attn.shape}")
                    print(f"    Dtype: {first_attn.dtype}")
                    print(f"    Device: {first_attn.device}")
                    print(f"    Min: {first_attn.min().item():.8f}")
                    print(f"    Max: {first_attn.max().item():.8f}")
                    print(f"    Mean: {first_attn.mean().item():.8f}")
                    print(f"    Std: {first_attn.std().item():.8f}")
                    print(f"    Has NaN: {torch.isnan(first_attn).any().item()}")
                    print(f"    Has Inf: {torch.isinf(first_attn).any().item()}")
                    
                    # Check dimensions
                    if len(first_attn.shape) == 4:
                        print(f"    ✅ 4D attention - Expected format!")
                        print(f"       Batch: {first_attn.shape[0]}, Heads: {first_attn.shape[1]}, Query: {first_attn.shape[2]}, Key: {first_attn.shape[3]}")
                    else:
                        print(f"    ⚠️ Unexpected shape: {first_attn.shape}")
                    
                    # Check a few more tokens
                    print(f"\n[3] CHECKING MULTIPLE TOKENS:")
                    for check_idx in [0, min(5, len(attentions)-1), min(10, len(attentions)-1)]:
                        if check_idx < len(attentions):
                            attn = attentions[check_idx][0]
                            print(f"    Token {check_idx}: shape={attn.shape}, hasNaN={torch.isnan(attn).any().item()}, mean={attn.mean().item():.6f}")
            
            print(f"{'='*70}\n")
        
        # Extract results
        sequences = outputs.sequences
        attentions = outputs.attentions
        
        # ============================================
        # FIX: Use teacher forcing IDs for model_completion
        # ============================================
        if args.teacher_forcing_jsonl is not None and teacher_forcing_ids is not None:
            # Use teacher forcing IDs directly (this is the ground truth output)
            gen_sequences = teacher_forcing_ids[0]
            model_completion = llm.tokenizer.decode(gen_sequences, skip_special_tokens=True)
            model_completion_ids = gen_sequences.tolist()
            if args.debug:
                print(f"✅ Using teacher forcing for completion: {model_completion[:100]}...")
        else:
            # Fallback to generated sequences
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            model_completion = llm.tokenizer.decode(gen_sequences, skip_special_tokens=True)
            model_completion_ids = gen_sequences.tolist()
            if args.debug:
                print(f"⚠️ Using generated sequences (no teacher forcing): {model_completion[:100]}...")
        
        # ============================================
        # Original lookback ratio calculation (unchanged)
        # ============================================
        context_length = attentions[0][0].shape[-1] - extra_prompt_length
        new_token_length = len(attentions)
        num_layers = len(attentions[0])
        num_heads = attentions[0][0].shape[1]
        lookback_ratio = torch.zeros((num_layers, num_heads, new_token_length))
        
        for i in range(len(attentions)):
            for l in range(num_layers):
                attn_on_context = attentions[i][l][0, :, -1, :context_length].mean(-1)
                attn_on_new_tokens = attentions[i][l][0, :, -1, context_length:].mean(-1)
                lookback_ratio[l, :, i] = attn_on_context / (attn_on_context + attn_on_new_tokens)
        
        # Debug lookback ratio stats
        if args.debug:
            print(f"\n[4] LOOKBACK RATIO STATS:")
            print(f"    Shape: {lookback_ratio.shape}")
            print(f"    Min: {lookback_ratio.min().item():.6f}")
            print(f"    Max: {lookback_ratio.max().item():.6f}")
            print(f"    Mean: {lookback_ratio.mean().item():.6f}")
            print(f"    Std: {lookback_ratio.std().item():.6f}")
            print(f"    Has NaN: {torch.isnan(lookback_ratio).any().item()}")
            print(f"    Has Inf: {torch.isinf(lookback_ratio).any().item()}")
            print()
        
        # Remove stop words from completion
        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]

        # Save results
        to_save = {
            'data_index': sample['data_index'],
            'model_completion': model_completion,
            'model_completion_ids': model_completion_ids,
            'full_input_text': input_text,
            'lookback_ratio': lookback_ratio,
        }
        to_save_list.append(to_save)

        # Clean up
        del teacher_forcing_ids
        del input_ids
        del input_text
        del model_completion
        del attentions
        del outputs
        del lookback_ratio
        del sample
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        gc.collect()

    # Save all results
    torch.save(to_save_list, args.output_path)
    print(f"\nSaved {len(to_save_list)} examples to {args.output_path}")
    
    # Final summary
    if args.debug and to_save_list:
        print(f"\n{'='*70}")
        print(f"FINAL SUMMARY - First sample lookback_ratio:")
        first_sample = to_save_list[0]['lookback_ratio']
        print(f"  Shape: {first_sample.shape}")
        print(f"  Min: {first_sample.min().item():.6f}")
        print(f"  Max: {first_sample.max().item():.6f}")
        print(f"  Mean: {first_sample.mean().item():.6f}")
        print(f"{'='*70}")
