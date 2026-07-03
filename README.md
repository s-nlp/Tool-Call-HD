# TOOLHACE-LookbackLens

## 📖 About
LookbackLens implementation for the TOOLHACE dataset (Rag-Truth like structure)

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- CUDA 11.7 (for GPU support)

### Installation
```bash
# Clone the repository
git clone https://github.com/BogdanMonogov/TOOLHACE-LookbackLens.git
cd TOOLHACE-LookbackLens

# Install dependencies
pip install -r requirements.txt
pip install -e ./transformers-4.32.0
```
### Usage
Below are the main scripts in the pipeline. Each section includes a short description and a ready-to-copy command.

#### teacher_forcing.py
Generates token IDs for model outputs. This is used for datasets that already contain precomputed answers and annotations. The specification of the dataset and output file name is required inside the script.

```bash
python teacher_forcing.py
```

#### step01.py
Computes lookback ratios for existing answers (including hallucinated segments) and saves the result as a .pt file.

```bash
python step01.py \
    --model-name meta-llama/Llama-2-7b-chat-hf \
    --data-path dataset.jsonl \
    --output-path lookback_ratios.pt \
    --teacher-forcing-jsonl teacher_forcing_ids.jsonl \
    --auth-token 'INSERT_THE_TOKEN_HERE' \
    --custom-dataset \
    --num-gpus 1 \
    --max-memory 15 \
    --max-new-tokens 408
```

#### step3_window_vote.py
Applies precomputed sliding-window classifiers to detect hallucinations in generated answers.

```bash
python step3_window_vote.py \
    --lookback_ratio_file lookback_ratios.pt \
    --classifier_file classifiers/classifier_anno-nq-7b_sliding_window_8.pkl \
    --auth_token 'INSERT_THE_TOKEN_HERE' \
    --output_file nq_window_preds.jsonl \
    --tokenizer_name meta-llama/Llama-2-7b-chat-hf
```

#### step3_eval_spans.py
Runs span-based classifiers over the lookback representations to detect hallucinations using predefined span segmentation.

```bash
python step3_eval_spans.py \
    --lookback_ratio_file lookback_ratios.pt \
    --classifier_file classifiers/classifier_anno-cnndm-7b_predefined_span.pkl \
    --output_file cnndm_span_preds.jsonl \
    --tokenizer_name meta-llama/Llama-2-7b-chat-hf \
    --auth_token 'INSERT_TOKEN_NAME' \
    --max_span_length 50 \
    --merge_threshold 2
```

#### form_preds_LBL.py 
Converts model predictions (JSONL format) into a CSV file and aligns them with gold labels from the original dataset for evaluation.

```bash
python form_preds_LBL.py \
    --pred nq_span.jsonl \
    --gold dataset.jsonl \
    --output NQ_SPAN.csv
```

