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

Our best released checkpoint (ModernBERT-large + linear-chain CRF, uniform soup of decorrelated students — char F1 0.99 / span F1 0.885 @ IoU≥0.75 on `toolace-unified-hallucinations` test) is:

- [`s-nlp/tool-calling-hallucination-modernbert-large-crf-best`](https://huggingface.co/s-nlp/tool-calling-hallucination-modernbert-large-crf-best) *(private)*

LettuceDetect-style ModernBERT-base baseline:

- [`s-nlp/tool-calling-hallucination-modernbert-base-unified-final`](https://huggingface.co/s-nlp/tool-calling-hallucination-modernbert-base-unified-final)

Both Qwen 3.5 0.8B and 2B available at:
- [`s-nlp/tool-calling-hallucination-detection`](https://huggingface.co/collections/s-nlp/tool-calling-hallucination-detection)

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

Clone modified transformers version and original pretrained classifiers:

```bash
# 1. Clone the repo without downloading files
git clone --filter=blob:none --no-checkout https://github.com/BogdanMonogov/TOOLHACE-LookbackLens.git
cd TOOLHACE-LookbackLens

# 2. Enable sparse-checkout
git sparse-checkout init --cone

# 3. Specify the folders you need
git sparse-checkout set classifiers transformers-4.32.0

# 4. Download the files from these folders
git checkout main
```
Install the dependencies:

```bash
pip install -r requirements.txt
pip install -e ./transformers-4.32.0
```

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


---

## Model Card — `s-nlp/tool-calling-hallucination-modernbert-large-crf-best`

**Uniform soup of 6 ModernBERT-large + CRF students** for **character-level hallucination span detection** in tool-calling LLM answers. Given a multi-turn tool-calling prompt and the model's final answer, it returns the character spans of the answer that are **not grounded** (hallucinated). On the `s-nlp/toolace-unified-hallucinations` test split (5,709 rows) it reaches **Span F1 0.8854** and **CharF1 0.9915**, beating both the `s-nlp` base baseline (**0.8425 / 0.9837**) and the non-CRF large model (**0.8685 / 0.9886**).

Each student is a ModernBERT-large token classifier warm-started from `etomoscow/tool-calling-hallucination-modernbert-large`, fine-tuned with a linear-chain **CRF head** (Viterbi decoding). The 6 students are **decorrelated** (different weight decay / data oversampling / learning rate / epoch count), so averaging their weights yields a more robust model than any single one — the "model soup" recipe (Wortsman et al. 2022), which is *weight merging*, not inference-time ensembling.

### Results

All numbers below are computed with the **authoritative `evaluate.py`** from
[`s-nlp/Tool-Call-HD` (branch `upload-eval-script`)](https://github.com/s-nlp/Tool-Call-HD/blob/upload-eval-script/evaluate/evaluate.py),
over the **full 5,709-row test split** of `s-nlp/toolace-unified-hallucinations`.
**Every model is evaluated with the same `build_prompt` pipeline** (System/User/Tool
conversation format) so the comparison is apples-to-apples; our model additionally
uses CRF Viterbi decoding with calibrated `bias=-0.5`, `min_span=3`.

| model | CharF1 | Span F1 (IoU≥0.75) | Span P | Span R |
|-------|--------|--------------------|--------|--------|
| `s-nlp/…base-unified-final` (base, argmax) | 0.9837 | 0.8425 | 0.846 | 0.839 |
| `etomoscow/…modernbert-large` (large, argmax) | 0.9886 | 0.8685 | 0.886 | 0.852 |
| **ours: large + CRF soup (Viterbi)** | **0.9915** | **0.8854** | **0.915** | 0.858 |

- **Span F1 (IoU≥0.75):** +0.043 over the base baseline, +0.017 over the non-CRF large model.
- **CharF1:** +0.008 over base, +0.003 over large.
- Binary hallucination F1 ≈ 0.99 for all three (response-level detection is saturated; the differentiator is span localization).

> Metric definitions (from `evaluate.py`): **CharF1** = global micro character-overlap F1 (`Σoverlap/Σpred`, `Σoverlap/Σgold`). **Span F1 (IoU≥0.75)** = greedy pairwise span matching, a pair is TP if IoU≥0.75, summed globally to P/R/F1.
>
> Cross-check: the large model was independently measured at Span F1 0.8597 by a colleague on the same test split — within ~1pt of the 0.8685 above, confirming the pipeline.

### Soup ingredients (uniform weight average)

| student | distinct lever | single Span F1 (IoU≥0.75) |
|---------|---------------|----------------|
| e16 | weight_decay=0.1 | 0.8807 |
| e13 | oversample answer_mismatch ×3 | 0.8788 |
| e02 | lr=2e-4 | 0.8788 |
| e14 | oversample answer_mismatch ×4 | 0.8784 |
| e04 | epochs=6 | 0.8784 |
| c08 | lr=1e-4 (baseline recipe) | 0.8807 |

All trained: lr ∈ {1e-4, 2e-4}, 6–8 epochs, effective batch 16, cosine, warmup 0.1, bf16 + FlashAttention2 + `adamw_torch_fused`, CRF-NLL loss.

### ⚠️ Inference requires the CRF head + Viterbi (NOT plain argmax)

This model is a classifier **plus** a linear-chain CRF. Decode with **Viterbi** and apply the calibrated emission bias (`-0.5`) and min-span filter (`3` chars). `crf.py` (bundled) is a dependency-free `LinearCRF`.

```python
# pip install transformers torch safetensors huggingface_hub; flash-attn optional (use attn="sdpa" if unavailable)
import os, sys, torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

MODEL = "s-nlp/tool-calling-hallucination-modernbert-large-crf-best"
BIAS, MIN_SPAN, MAX_LEN = -0.5, 3, 4096

sys.path.insert(0, os.path.dirname(hf_hub_download(MODEL, "crf.py")))  # bundled LinearCRF
from crf import LinearCRF

tok = AutoTokenizer.from_pretrained(MODEL)
enc = AutoModelForTokenClassification.from_pretrained(
    MODEL, attn_implementation="flash_attention_2", dtype=torch.bfloat16).cuda().eval()
crf = LinearCRF(num_tags=2, batch_first=True); crf.float()
crf.load_state_dict(load_file(hf_hub_download(MODEL, "crf.safetensors"))); crf.cuda().eval()

def predict_spans(prompt: str, answer: str) -> list[dict]:
    """Return hallucinated char spans, relative to `answer`, as {'start','end'} dicts."""
    e = tok(prompt, answer, truncation="only_first", max_length=MAX_LEN,
            return_offsets_mapping=True, return_tensors="pt")
    offsets = e.pop("offset_mapping")[0].tolist()
    logits = enc(input_ids=e["input_ids"].cuda(),
                 attention_mask=e["attention_mask"].cuda()).logits[0].float()
    logits[..., 1] += BIAS                                   # emission calibration
    n_ans = len(tok(answer, add_special_tokens=False)["input_ids"])
    T = logits.shape[0]; a_start = T - n_ans - 1
    a_off = offsets[a_start][0]                             # char offset of answer start
    ans_logits = logits[a_start:a_start + n_ans].unsqueeze(0)
    mask = torch.ones(1, n_ans, dtype=torch.bool, device=logits.device)
    preds = crf.decode(ans_logits, mask)[0]                 # Viterbi over answer region
    spans, cur = [], None
    for k, p in enumerate(preds):
        ts, te = offsets[a_start + k]
        if ts == te: continue
        s, e_ = ts - a_off, te - a_off
        if p == 1:
            cur = [s, e_] if cur is None else [cur[0], e_]
        elif cur is not None:
            spans.append(cur); cur = None
    if cur is not None: spans.append(cur)
    return [{"start": s, "end": e} for s, e in spans if e - s >= MIN_SPAN]

# prompt = "System: ...\n\nUser: ...\n\nTool: ..." (everything except the final answer)
# answer  = the model's final_answer text
for sp in predict_spans(prompt, answer):
    print(sp, repr(answer[sp["start"]:sp["end"]]))
```

Spans are **character offsets relative to `answer`** (matching the dataset's `span_labels`). Build `prompt` from all turns except `final_answer`.

> **Private repo:** this model is private to the `s-nlp` org. Set `HF_TOKEN` (with `s-nlp` read access) before running — `hf_hub_download` / `from_pretrained` pick it up from the env.

### Usage with LettuceDetect (drop-in)

For the LettuceDetect / `Tool-Call-HD` pipeline, this repo also ships a CRF-aware **`transformer.py`** — a drop-in for `lettucedetect/detectors/transformer.py`. It auto-detects `crf.safetensors` next to the model and decodes with **Viterbi** (bias `-0.5`, `min_span 3`); models without it fall back to argmax (back-compatible, so the `s-nlp` base baseline still works unchanged).

```bash
pip install lettucedetect transformers torch safetensors huggingface_hub
export HF_TOKEN=<your s-nlp token>   # private repo
# drop the bundled transformer.py over your lettucedetect install:
LETTUCE=$(python -c "import lettucedetect,os;print(os.path.dirname(lettucedetect.__file__))")
cp "$(python -c "from huggingface_hub import hf_hub_download;print(hf_hub_download('s-nlp/tool-calling-hallucination-modernbert-large-crf-best','transformer.py'))")" \
   "$LETTUCE/detectors/transformer.py"
```

```python
from lettucedetect.models.inference import HallucinationDetector
det = HallucinationDetector(
    method="transformer",
    model_path="s-nlp/tool-calling-hallucination-modernbert-large-crf-best")
# logs: [CRF] loaded .../crf.safetensors -> Viterbi decode (bias=-0.5, min_span=3)
spans = det.predict(context=[tool_result], answer=final_answer,
                    question=user_query, output_format="spans")
```

Without the patch, `HallucinationDetector` loads the encoder and argmax-decodes (CRF head ignored) — it runs, but gives the non-CRF large-argmax quality (~0.8685 Span F1) rather than the full soup (0.8854).

### Evaluation protocol
- **Script:** `evaluate/evaluate.py` from `s-nlp/Tool-Call-HD` (branch `upload-eval-script`).
- **Data:** `s-nlp/toolace-unified-hallucinations` test split (5,709 rows).
- **Prompt format:** `build_prompt` (System / User / Tool turns), identical for all three models.
- **Metrics:** CharF1 (global micro char-overlap) and Span F1 (greedy IoU≥0.75).
- Calibration (`bias`/`min_span`) tuned on dev, never on test.

### Limitations
- `undergeneration` and `clean` examples have empty gold spans by dataset design; on Span F1 they contribute TN/FN as expected (not a detection failure).
- Spans are char-level offsets into the final answer; prompts >4096 tokens are left-truncated.
- Calibration (`bias`, `min_span`) was tuned on the s-nlp dev split; re-tune for other distributions.

### Files
- `model.safetensors` — encoder + classifier head (uniform average of 6 students, `ModernBertForTokenClassification`)
- `crf.safetensors` — averaged CRF transition parameters → `LinearCRF(num_tags=2, batch_first=True)` via `safetensors.torch.load_file`
- `crf.py` — vendored, dependency-free `LinearCRF` (NLL + Viterbi + `pack_valid`)
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `README.md`

### Citation
If you use this, please cite the LettuceDetect / s-nlp tool-calling hallucination work, the model-souping recipe (Wortsman et al. 2022), and the ModernBERT base model.
