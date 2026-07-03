# save as prepare_teacher_forcing.py
import json
import os
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def prepare_teacher_forcing(input_jsonl, output_jsonl, model_name="meta-llama/Llama-2-7b-chat-hf"):
    """
    Create teacher forcing file with token IDs for your precomputed answers
    """

    auth_token = ''
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
    
    with open(input_jsonl, 'r') as f_in, open(output_jsonl, 'w') as f_out:
        for idx, line in enumerate(f_in):
            item = json.loads(line)
            
            # Tokenize the output
            token_ids = tokenizer.encode(item["output"], add_special_tokens=False)
            
            teacher_forcing_item = {
                "data_index": idx,
                "model_completion_ids": token_ids
            }
            
            f_out.write(json.dumps(teacher_forcing_item) + '\n')
            print(f"Processed {idx}: {len(token_ids)} tokens")

if __name__ == "__main__":
    prepare_teacher_forcing(
        "../datasets/TOOLHACE_final.jsonl",  # Your input dataset
        "../datasets/TEAFOR_TOOLHACE_final.jsonl",  # Output teacher forcing file
        "meta-llama/Llama-2-7b-chat-hf"
    )