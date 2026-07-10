# Models

Code for training and evaluating **every hallucination detector reported in the
ToolHACE paper**. Each subfolder is one modeling approach; they all target the
same task — span-level hallucination detection in *post-tool-call* LLM responses
— but differ in architecture and how the detector is trained.

No data is bundled here. All scripts consume ToolHACE data produced by the
`../generation_pipeline/` scripts or pulled from the released HF datasets (see
the top-level [`README.md`](../README.md)).

## Layout

```text
models/
  lettucedetect/     Encoder detector: ModernBERT token-classifier (LettuceDetect)
    train/           fine-tune the ModernBERT span tagger
    inference/       inference + rich evaluation for trained checkpoints
  sft-train/         Generative detector: SFT LoRA fine-tune of a chat LLM
                     (Qwen3.5 / Gemma) that emits {type, spans} as JSON
  lookbacklens/      Attention-based detector: Lookback-Lens lookback-ratio probes
  verbalized/        Zero/few-shot LLM baselines ("verbalized" prompting)
```

## The four approaches

| Folder | Family | What it does | Output |
| --- | --- | --- | --- |
| [`lettucedetect/`](lettucedetect/) | Encoder (ModernBERT) | Token-level classifier fine-tuned to tag hallucinated spans | Per-token span labels |
| [`sft-train/`](sft-train/) | Decoder (Qwen3.5 / Gemma) | LoRA-SFT a chat model to generate the error type + exact spans | JSON `{type, spans}` |
| [`lookbacklens/`](lookbacklens/) | Attention probe | Trains classifiers over lookback-ratio features of a frozen LLM | Span / window verdicts |
| [`verbalized/`](verbalized/) | Prompted LLM | No training — prompts an off-the-shelf LLM to name the error | Free-form → scored |
