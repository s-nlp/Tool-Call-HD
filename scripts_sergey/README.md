## 1. Extract structural multi-step dialogues from ToolACE

```bash
python toolace_multistep_pipeline.py extract \
  --hf-dataset Team-ACE/ToolACE \
  --hf-split train \
  --output ../outputs/toolace_structural_multistep.json \
  --summary-output ../outputs/toolace_structural_multistep_summary.json \
  --examples-output ../outputs/toolace_structural_multistep_examples.json
```

This keeps only dialogues with `>= 2` completed tool-use turns of the form:

`user -> assistant(tool call) -> tool -> assistant(answer)`


## 2. Clean trailing follow-up questions from the last assistant reply

LLM-based cleaning

```bash
python toolace_multistep_pipeline.py clean-last-followup \
  --input ../outputs/toolace_structural_multistep.json \
  --output ../outputs/toolace_structural_multistep_cleaned_openai.json \
  --summary-output ../outputs/toolace_structural_multistep_cleaned_openai_summary.json \
  --model gpt-4.1-mini
```

This removes only trailing assistant follow-up / CTA text such as:

- `Would you like me to ...`
- `Let me know if you need ...`
- `Feel free to ask ...`

## 3. Run LettuceDetect on the ToolHACE

### Installation

```bash
pip install lettucedetect>=0.1.8 datasets transformers torch tqdm pandas
```

### Best released ToolHACE checkpoint

- [`s-nlp/tool-calling-hallucination-modernbert-base-unified-final`](https://huggingface.co/s-nlp/tool-calling-hallucination-modernbert-base-unified-final)

### Prepare ToolHACE rows for span inference

`run_lettuce_detector.py` expects flat JSONL with `query`, `context`, and `output`. Convert the unified ToolHACE rows first:

```bash
python prepare_lettuce_detector_input.py \
  --hf-dataset s-nlp/toolace-unified-hallucinations \
  --hf-split test \
  --output ../data/toolhace_test_for_lettuce.jsonl
```

### Input format

Each line of the input JSONL must be a JSON object with:

| Field | Description |
|---|---|
| `query` | The user question |
| `context` | Tool/retrieval response the answer is grounded in |
| `output` | The generated answer to evaluate |

Example line:
```json
{"query": "What is the capital of France?", "context": "France is a country in Western Europe. Its capital city is Paris.", "output": "The capital of France is Lyon."}
```

### Usage

```bash
python run_lettuce_detector.py \
    --method lettucedetect \
    --checkpoint s-nlp/tool-calling-hallucination-modernbert-base-unified-final \
    --data ../data/toolhace_test_for_lettuce.jsonl \
    --output ../predictions/toolhace_modernbert_base_unified_final.jsonl
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--method` | required | Detector backend: `lettucedetect`, `haldetect`, `always_hal`, `always_no_hal` |
| `--checkpoint` | `""` | HuggingFace model ID or local path |
| `--data` | required | Path to input JSONL file |
| `--output` | auto | Output path (defaults to `predictions/<method>_<checkpoint>_<input_stem>.jsonl`) |
| `--batch-size` | `32` | Inference batch size |
| `--threshold` | `0.5` | Token-level hallucination probability cutoff |

### Output format

Each output line is the input object extended with a `pred` field — a list of hallucinated spans, or `[]` if none detected:

```json
{
  "query": "What is the capital of France?",
  "context": "France is a country in Western Europe. Its capital city is Paris.",
  "output": "The capital of France is Lyon.",
  "pred": [{"start": 26, "end": 30, "text": "Lyon"}]
}
```

### Examples

Run with the best ToolHACE checkpoint:
```bash
python run_lettuce_detector.py \
    --method lettucedetect \
    --checkpoint s-nlp/tool-calling-hallucination-modernbert-base-unified-final \
    --data ../data/toolhace_test_for_lettuce.jsonl
```

### Train a ToolHACE model

```bash
python train_toolhace_lettuce.py \
  --hf-dataset s-nlp/toolace-unified-hallucinations \
  --hf-train-split train \
  --hf-dev-split dev \
  --model-name answerdotai/ModernBERT-base \
  --output-dir ../outputs/toolhace_modernbert_base \
  --batch-size 4 \
  --epochs 6 \
  --learning-rate 1e-5 \
  --grad-accum 8
```

If you have local labeled files instead of the HF dataset:

```bash
python train_toolhace_lettuce.py \
  --train-input /path/to/train.json \
  --dev-input /path/to/dev.json \
  --model-name answerdotai/ModernBERT-base \
  --output-dir ../outputs/toolhace_modernbert_base
```
