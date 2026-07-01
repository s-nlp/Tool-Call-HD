# ToolHACE

**A Dataset for Detecting Hallucinations in Tool-Based LLM Responses**

ToolHACE is a span-level benchmark for **hallucination detection in tool-augmented LLM responses**. It targets a stage of the tool-use pipeline that existing benchmarks largely overlook: the *post-tool-call response generation* step, where a model can still go wrong even after the correct tool has been called and accurate information has been returned.

---

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

## Key Findings from the Paper

Through extensive experiments with modern hallucination detectors, the ToolHACE study shows that:

1. **Post-tool-call hallucinations are widespread** and remain **challenging to detect**.
2. **Effective detection requires training on specialized tool-augmented corpora** — models trained on in-domain, tool-augmented data perform substantially better.
3. **Transfer from tool-agnostic hallucination datasets performs poorly** in this setting, underscoring that general-purpose hallucination detection does not carry over to tool-augmented responses.

Together these results highlight the need for dedicated benchmarks and models for reliable hallucination detection in tool-augmented language systems.

## Repository Status

This repository hosts code and resources for the ToolHACE benchmark. Dataset files, data-generation pipelines, and evaluation scripts will be added here. Contributions and issues are welcome.

## Best Released ToolHACE Model

Our best released LettuceDetect-style checkpoint is:

- [`s-nlp/tool-calling-hallucination-modernbert-base-unified-final`](https://huggingface.co/s-nlp/tool-calling-hallucination-modernbert-base-unified-final)

The unified ToolHACE evaluation scripts in this repository default to the Hugging Face dataset:

- `s-nlp/toolace-unified-hallucinations`

## Run The Best Model

Install the core dependencies:

```bash
pip install lettucedetect>=0.1.8 datasets transformers torch tqdm pandas
```

Convert ToolHACE unified rows into the `query/context/output` JSONL format expected by `scripts/run_lettuce_detector.py`:

```bash
python scripts/prepare_lettuce_detector_input.py \
  --hf-dataset s-nlp/toolace-unified-hallucinations \
  --hf-split test \
  --output data/toolhace_test_for_lettuce.jsonl
```

Then run the best released checkpoint:

```bash
python scripts/run_lettuce_detector.py \
  --method lettucedetect \
  --checkpoint s-nlp/tool-calling-hallucination-modernbert-base-unified-final \
  --data data/toolhace_test_for_lettuce.jsonl \
  --output predictions/toolhace_modernbert_base_unified_final.jsonl
```

## Train On ToolHACE

To train a ModernBERT-based detector on labeled ToolHACE unified rows:

```bash
python scripts/train_toolhace_lettuce.py \
  --hf-dataset s-nlp/toolace-unified-hallucinations \
  --hf-train-split train \
  --hf-dev-split dev \
  --model-name answerdotai/ModernBERT-base \
  --output-dir outputs/toolhace_modernbert_base \
  --batch-size 4 \
  --epochs 6 \
  --learning-rate 1e-5 \
  --grad-accum 8
```

If you already have local train/dev files, replace the HF arguments with `--train-input ... --dev-input ...`.

## Evaluate A Checkpoint

To compute response-level, character-level, span-level, and by-type metrics while saving rich per-row predictions:

```bash
python evaluate/evaluate_save.py \
  --model s-nlp/tool-calling-hallucination-modernbert-base-unified-final \
  --hf-dataset s-nlp/toolace-unified-hallucinations \
  --hf-split test \
  --save-preds evaluate/results/toolhace_modernbert_base_test_predictions.jsonl \
  --by-type
```

## Citation

If you use ToolHACE in your work, please cite the paper:

```bibtex
@misc{toolhace,
  title  = {ToolHACE: A Dataset for Detecting Hallucinations in Tool-Based LLM Responses},
  note   = {Span-level benchmark for hallucination detection in tool-augmented LLM responses}
}
```
