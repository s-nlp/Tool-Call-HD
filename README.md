**A Dataset for Detecting Hallucinations in Tool-Based LLM Responses**

ToolHACE is a span-level benchmark for **hallucination detection in tool-augmented LLM responses**. It targets a stage of the tool-use pipeline that existing benchmarks largely overlook: the *post-tool-call response generation* step, where a model can still go wrong even after the correct tool has been called and accurate information has been returned. This repository hosts code and resources for the this benchmark. Dataset files, data-generation pipelines, and evaluation scripts will be added here.

---

## Repository Layout

```text
generation_pipeline/
  hallucination_editing_pipeline/    span-tagged editing of gold answers (types 1.2 / 2.1 / 3.1)
  hallucination_generation_pipeline/ LLM generation + unify + judge + filter + export
evaluate/                            zero/few-shot LLM baselines + metric scorers
  verbalized/                        verbalized-baseline scorer (check_eval.py)
lettucedetect/
  train/                             ModernBERT (LettuceDetect) trainer
  inference/                         inference + rich evaluation for trained checkpoints
datasets/                            source data
toolhace_lettuce_utils.py            shared unified-row helpers
```

## Motivation

Tool-augmented language models ground their responses in external tool outputs, improving reliability beyond parametric knowledge alone. But grounding is not a guarantee. Even when the correct tool is called and returns accurate information, a model may still:

- **contradict** the tool output,
- **omit** critical details, or
- **generate unsupported claims** not backed by any tool result.

Most hallucination benchmarks for tool-augmented dialogue focus on the **tool-calling stage** — whether the right tool and arguments were selected. They say little about what happens *after* the tool returns. ToolHACE is built to fill that gap.

## What ToolHACE Provides

- **Span-level annotations.** Hallucinations are labeled at the span level within the generated response, rather than as a single document-level verdict, enabling fine-grained detection and evaluation.
- **Five fine-grained hallucination categories.** Each problematic span is assigned to one of five categories capturing distinct ways a post-tool-call response can deviate from its grounding evidence.
- **Tool-augmented context.** Each example pairs a model response with the tool calls and tool outputs it was meant to be grounded in, so detectors can reason about the response *relative to* the available evidence.

## Key Findings

Through extensive experiments with modern hallucination detectors, the ToolHACE study shows that:

1. **Post-tool-call hallucinations are widespread** and remain **challenging to detect**.
2. **Effective detection requires training on specialized tool-augmented corpora** — models trained on in-domain, tool-augmented data perform substantially better.
3. **Transfer from tool-agnostic hallucination datasets performs poorly** in this setting, underscoring that general-purpose hallucination detection does not carry over to tool-augmented responses.

Together these results highlight the need for dedicated benchmarks and models for reliable hallucination detection in tool-augmented language systems.

## Best Released ToolHACE Model

Our best released LettuceDetect-style checkpoint is:

- [`s-nlp/tool-calling-hallucination-modernbert-base-unified-final`](https://huggingface.co/s-nlp/tool-calling-hallucination-modernbert-base-unified-final)

The unified ToolHACE evaluation scripts in this repository default to the Hugging Face dataset:

- [`s-nlp/toolace-unified-hallucinations_upd_v2`](https://huggingface.co/datasets/s-nlp/toolace-unified-hallucinations_upd_v2)

## Run The Best Model

Install the core dependencies:

```bash
pip install lettucedetect>=0.1.8 datasets transformers torch tqdm pandas
```

Evaluate the best released checkpoint directly against the unified HF dataset
(loads rows, runs inference, and computes metrics in one step):

```bash
export HF_TOKEN="your_token_id"
python evaluate/evaluate_save.py \
  --model s-nlp/tool-calling-hallucination-modernbert-base-unified-final \
  --hf-dataset s-nlp/toolace-unified-hallucinations_upd_v2 \
  --hf-split test \
  --save-preds evaluate/results/toolhace_modernbert_base_test_predictions.jsonl \
  --by-type
```

To run the model over new, unlabeled data instead, use
`lettucedetect/inference/predict_spans.py` (see
`lettucedetect/inference/README.md`).

## Train On ToolHACE

Training uses the stock LettuceDetect trainer at `lettucedetect/train/train.py`,
which expects a LettuceDetect-format JSON (`prompt`, `answer`,
`labels: [{start, end, label}]`, `split`). Produce one from either pipeline:

```bash
# from the editing pipeline
python generation_pipeline/hallucination_editing_pipeline/make_lettucedetect_data.py

# or from the LLM generation pipeline
python generation_pipeline/hallucination_generation_pipeline/export_lettucedetect.py \
  --input output/final_dataset/<RUN> --out-dir lettucedetect_data
```

Then train:

```bash
CUDA_VISIBLE_DEVICES=0 python lettucedetect/train/train.py \
  --ragtruth-path lettucedetect_data/toolhace_train.json \
  --model-name answerdotai/ModernBERT-base \
  --output-dir outputs/toolhace_modernbert_base \
  --batch-size 4 --epochs 6 --learning-rate 1e-5 --grad-accum 8
```

## Evaluate A Checkpoint

To compute response-level, character-level, span-level, and by-type metrics while saving rich per-row predictions:

```bash
python evaluate/evaluate_save.py \
  --model s-nlp/tool-calling-hallucination-modernbert-base-unified-final \
  --hf-dataset s-nlp/toolace-unified-hallucinations_upd_v2 \
  --hf-split test \
  --save-preds evaluate/results/toolhace_modernbert_base_test_predictions.jsonl \
  --by-type
```

## Lookback-Lens guide

Run this script to obtain teacher forcing token ids for the lookback ratios extraction (denote the authentication token and the destination path inside the script before running it):

```bash
python lookbacklens/teacher_forcing.py
```

To compute lookback ratios on the existing model answers run:

```bash
python lookbacklens/step01.py \
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

To get predictions using the sliding window classifier:

```bash
python lookbacklens/step3_window_vote.py \
    --lookback_ratio_file lookback_ratios.pt \
    --classifier_file classifiers/classifier_anno-nq-7b_sliding_window_8.pkl \
    --auth_token 'INSERT_THE_TOKEN_HERE' \
    --output_file nq_window_preds.jsonl \
    --tokenizer_name meta-llama/Llama-2-7b-chat-hf
```
To get the predictions using the span based classifier:

```bash
python lookbacklens/step3_eval_spans.py \
    --lookback_ratio_file lookback_ratios.pt \
    --classifier_file classifiers/classifier_anno-cnndm-7b_predefined_span.pkl \
    --output_file cnndm_span_preds.jsonl \
    --tokenizer_name meta-llama/Llama-2-7b-chat-hf \
    --auth_token 'INSERT_TOKEN_NAME' \
    --max_span_length 50 \
    --merge_threshold 2
```

To tranform the jsonl predictions into the ready-to-evaluate structure with gold labels:

```bash
python lookbacklens/form_preds_LBL.py \
    --pred nq_span.jsonl \
    --gold dataset.jsonl \
    --output NQ_SPAN.csv
```
