"""Shared prompt, target, and parsing helpers for the transfer experiment."""

import json


ERROR_TYPES = (
    "answer_mismatch",
    "overgeneration",
    "missing_tool",
    "undergeneration",
)
ALL_TYPES = ("clean",) + ERROR_TYPES

USER_MAX_CHARS = 1200
CONTEXT_MAX_CHARS = 6000
ANSWER_MAX_CHARS = 4000

SYSTEM = (
    "You are a hallucination auditor. Compare the candidate ANSWER with the "
    "USER_REQUEST, AVAILABLE_TOOLS, and GROUNDING_CONTEXT. Return JSON only."
)

INSTRUCTION = """Audit the ANSWER above. Error classes:
- answer_mismatch: a statement or value contradicts GROUNDING_CONTEXT (span = the wrong text in ANSWER)
- overgeneration: a claim is not supported by GROUNDING_CONTEXT at all (span = the unsupported text)
- missing_tool: ANSWER offers or promises an action that AVAILABLE_TOOLS cannot perform (span = the offer)
- undergeneration: ANSWER omits items or records supplied by GROUNDING_CONTEXT and requested by the user (span = null)

Reply with JSON ONLY, with no explanation:
{"errors": []} if the answer is fully grounded, otherwise
{"errors": [{"class": "<class>", "span": "<exact text copied from ANSWER, or null for undergeneration>"}, ...]}"""


def final_answer(example):
    """Return the final assistant answer from a ToolHACE-style row."""
    for turn in reversed(example["conversations"]):
        if turn.get("turn_role") == "final_answer":
            return turn.get("value", "")
    return example.get("answer") or example["conversations"][-1].get("value", "")


def build_prompt(example):
    """Build the same model-visible prompt for source training and ToolHACE eval."""
    tools = "; ".join(example.get("available_tools") or [])
    user = ""
    contexts = []
    for turn in example["conversations"]:
        role = turn.get("turn_role")
        if not user and (role == "user" or turn.get("from") == "user"):
            user = turn.get("value", "")
        if role == "tool_response":
            contexts.append(turn.get("value", ""))
    context = "\n".join(contexts)
    answer = final_answer(example)
    return (
        f"AVAILABLE_TOOLS: {tools}\n\n"
        f"USER_REQUEST: {user[:USER_MAX_CHARS]}\n\n"
        f"GROUNDING_CONTEXT:\n{context[:CONTEXT_MAX_CHARS]}\n\n"
        f"ANSWER:\n{answer[:ANSWER_MAX_CHARS]}\n\n"
        f"{INSTRUCTION}"
    )


def target_json(example):
    """Create the completion target, preserving a class on every individual span."""
    if int(example["label"]) == 0:
        return json.dumps({"errors": []})
    if example["type"] == "undergeneration":
        return json.dumps(
            {"errors": [{"class": "undergeneration", "span": None}]}
        )

    errors = []
    for span in example.get("span_labels") or []:
        span_type = span.get("span_type") or example["type"]
        text = span.get("text") or ""
        if span_type in ERROR_TYPES and text:
            errors.append({"class": span_type, "span": text})
    if not errors:
        raise ValueError(
            f"positive row {example.get('dialogue_id')} has no usable target spans"
        )
    return json.dumps({"errors": errors}, ensure_ascii=False)


def parse_verdict(text):
    """Extract the first balanced JSON object containing an errors list."""
    text = text.split("</think>")[-1]
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            char = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            value = json.loads(text[start : end + 1])
                        except Exception:
                            break
                        if isinstance(value, dict) and isinstance(
                            value.get("errors"), list
                        ):
                            return value
                        break
    return None
