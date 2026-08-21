# RAGTruth / PsiloQA transfer fine-tuning

This directory contains the focused pipeline used to run four independent
experiments on one A100:

1. Qwen3.5-0.8B trained on RAGTruth
2. Qwen3.5-0.8B trained on PsiloQA
3. Qwen3.5-2B trained on RAGTruth
4. Qwen3.5-2B trained on PsiloQA

Every run starts from the original Qwen checkpoint in the Hugging Face cache.
The language-model parameters are fully fine-tuned; LoRA, PEFT, and quantized
training are not used. Each resulting model is evaluated on ToolHACE.

## Label transfer to ToolHACE

| Source annotation | ToolHACE type |
| --- | --- |
| RAGTruth `Evident Conflict` or `Subtle Conflict` | `answer_mismatch` |
| RAGTruth `Evident Baseless Info` or `Subtle Baseless Info` | `overgeneration` |
| RAGTruth response without a hallucination annotation | `clean` |
| PsiloQA `[HAL]...[/HAL]` span | `answer_mismatch` |
| PsiloQA answer without a hallucinated span | `clean` |

RAGTruth can contain several spans with different transferred types, so those
span labels are preserved separately. PsiloQA only distinguishes hallucinated
answer text from clean text; we map its hallucinated spans to
`answer_mismatch`, because they are answer claims unsupported by the passage.
Neither source supplies direct supervision for ToolHACE `missing_tool` or
`undergeneration`.

The training input combines:

- RAGTruth: `query`, `context`, and `output`
- PsiloQA: `question`, `wiki_passage`, and `llm_answer`

The completion target is JSON:

```json
{"errors": [{"class": "answer_mismatch", "span": "exact answer text"}]}
```

A clean example uses `{"errors": []}`. Loss is computed only on the JSON
completion tokens, not on the prompt tokens.

## Run

The default configuration is one epoch, batch size 2, gradient accumulation
16 (effective batch size 32), and maximum sequence length 6144.

```bash
cd models/ragtruth-psiloqa-transfer
TEST_PARQUET=/path/to/ToolHACE_data/test-00000-of-00001.parquet ./run_all.sh
```

The runner resolves the original models from:

```text
/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B
/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B
```

Use `RUN_EXPERIMENTS=q08_ragtruth` (or a comma-separated subset) for a partial
run. The first data conversion downloads the Hugging Face datasets; by default
the PsiloQA English split and RAGTruth `quality=good` examples are used.

Outputs are written under `runs/<experiment>/`: the training checkpoint in
`sft/model/`, the standalone model in `composite/`, predictions in
`eval/verdicts.jsonl`, and metrics in `eval/metrics.{json,csv,txt}`.

## Metrics

`compute_metrics.py` is the supplied new evaluator, included unchanged.
`normalize_for_compute_metrics.py` joins generated verdicts to the ToolHACE
test rows and converts predicted span text into character offsets required by
that evaluator. Response-level and span-level precision, recall, and F1 are
reported overall and by hallucination type; span matching uses IoU greater
than 0.75 by default.
