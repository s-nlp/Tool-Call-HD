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