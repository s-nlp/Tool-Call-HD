# ToolACE Seed Dataset Preparation

This folder contains only the scripts used before hallucination-type generation:

1. split raw `Team-ACE/ToolACE` into singlehop and multihop seed pools;
2. synthetically complete singlehop rows that stop at an assistant tool call;
3. remove trailing assistant follow-up / CTA text.

Generation of Type 1/2/3 hallucination examples is handled elsewhere and is
intentionally not included here.

## Files

`toolace_preparation/`

- `toolace_multistep_pipeline.py` - extracts structural multihop dialogues from
  raw ToolACE and cleans trailing follow-up text.
- `toolace_singlehop_pipeline.py` - extracts rows with exactly one detected tool
  call and cleans the answer attached to the completed tool-use turn.
- `generate_synthetic_tool_responses.py` - generates missing `tool_response`
  values for singlehop rows that end at an assistant tool call.
- `generate_tool_call_answer.py` - generates the final assistant answer after a
  synthetic tool response has been added.
- `generate_model_answers_openai.py` - helper for rows that already contain
  `user_prompt`, `tool_call`, and `tool_response`.
- `clean_followup_questions_openai.py` - older flat-row cleaner for
  `original_answer` / `tagged_answer` rows.

## 1. Extract Multihop From Raw ToolACE

Run from `datasets/scripts/toolace_preparation`:

```bash
python toolace_multistep_pipeline.py extract \
  --hf-dataset Team-ACE/ToolACE \
  --hf-split train \
  --output ../../multihop/toolace_multistep_without_followups.json \
  --summary-output ../../toolace_processing_report/toolace_multistep_summary.json \
  --examples-output ../../toolace_processing_report/toolace_multistep_examples.json
```

The selection rule is at least two completed tool-use cycles:

```text
user -> assistant(tool call) -> tool -> assistant(answer)
```

Then remove trailing assistant follow-up / CTA text:

```bash
python toolace_multistep_pipeline.py clean-last-followup \
  --input ../../multihop/toolace_multistep_without_followups.json \
  --output ../../multihop/toolace_multistep_clean.json \
  --summary-output ../../toolace_processing_report/toolace_multistep_clean_summary.json \
  --model gpt-4.1-mini
```

The final published multihop seed file in this repository is:

```text
../../multihop/multihop_real_toolace.json
```

## 2. Extract Singlehop From Raw ToolACE

Run from `datasets/scripts/toolace_preparation`:

```bash
python toolace_singlehop_pipeline.py extract \
  --hf-dataset Team-ACE/ToolACE \
  --hf-split train \
  --output ../../singlehop/toolace_singlehop_exact1.json \
  --summary-output ../../toolace_processing_report/toolace_singlehop_summary.json \
  --examples-output ../../toolace_processing_report/toolace_singlehop_examples.json
```

The selection rule is exactly one detected tool-call message in the dialogue.
In our processing report this produced 7,543 rows:

- 204 already had a complete tool-use cycle and became
  `../../singlehop/singlehop_real_toolace.json`;
- 7,339 ended at the assistant tool call and were used for synthetic
  completion.

Note: the final partition from `toolace_singlehop_exact1.json` into completed
vs. ends-at-tool-call rows appears to have been done in notebook/manual
processing. The standalone extract script records enough `analysis` metadata to
recreate that split.

## 3. Generate Missing Tool Responses And Final Answers

For singlehop rows that end at an assistant tool call, first generate a
successful synthetic tool response:

```bash
python generate_synthetic_tool_responses.py \
  --input ../../singlehop/singlehop_ends_at_tool_call.json \
  --output ../../singlehop/singlehop_with_synthetic_tool_response.json \
  --model gpt-4o \
  --workers 8
```

Then generate the final assistant message grounded in that tool response:

```bash
python generate_tool_call_answer.py \
  --input ../../singlehop/singlehop_with_synthetic_tool_response.json \
  --output ../../singlehop/singlehop_synthetic_completed_raw.json \
  --model gpt-4o \
  --workers 8
```

After filtering malformed/error/unavailable rows and removing synthetic
multihop artifacts, the final published synthetic seed file is:

```text
../../singlehop/singlehop_synthetic_toolace.json
```

## 4. Clean Follow-Up Text

For extracted multihop rows:

```bash
python toolace_multistep_pipeline.py clean-last-followup ...
```

For completed singlehop rows:

```bash
python toolace_singlehop_pipeline.py clean-tool-answer-followup ...
```

For older flat rows with `original_answer` / `tagged_answer`:

```bash
python clean_followup_questions_openai.py \
  --input dataset_v3_tagged.json \
  --output dataset_v3_tagged_cleaned.json \
  --model gpt-4.1-mini
```

