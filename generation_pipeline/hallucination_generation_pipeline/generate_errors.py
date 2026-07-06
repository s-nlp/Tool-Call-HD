"""
Synthetic error generation with SPAN-LEVEL annotations.
Supports multiple model providers (OpenAI-compatible, OpenRouter).

Pipeline order:
  1. "correct" is ALWAYS generated first — it produces data-span-annotated
     answers that other classes (especially hallucination) depend on.
  2. Other classes run after, using correct results where needed.

Classes:
  - correct: faithfully generated answer with <DATA> span annotations
  - hallucination: reuses correct answer + corrupted tool_response;
                   spans derived by cross-referencing DATA tags with JSON diff
  - overgeneration: spans mark unsupported additions
  - missing_tool: spans mark references to non-existent capabilities
  - undergeneration: lists omitted fields/items (no text spans)

Input formats supported:
  Datasets are HuggingFace Dataset objects (from disk, Hub, or files).
  Two layouts are accepted:

  Format A (flat, single tool call per sample):
    user_prompt, tool_call, tool_response, original_answer

  Format B (conversation, multi-turn with possibly multiple tool calls):
    conversations: [{"from": "user"|"assistant"|"tool", "value": str}, ...]
    -- The script targets the LAST tool turn for generation/modification.
    -- All prior messages become conversation history context.

Output:
  HuggingFace dataset saved to disk (load with datasets.load_from_disk)
  or pushed to the Hub with --push-to-hub.

Usage:
  python generate_errors.py --input ./toolace_filtered --output ./generated
  python generate_errors.py --input team-ace/ToolACE --output ./generated
  python generate_errors.py --input ./multi_tool_data --output ./generated --format conversation
  python generate_errors.py --input ./data --output my-org/errors --push-to-hub
"""

import json
import re
import argparse
import asyncio
import random
import time
from pathlib import Path
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from datasets import Dataset, load_dataset, load_from_disk

# ---------------------------------------------------------------------------
# Model provider configurations
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    base_url: str
    api_key: str
    model_id: str


PROVIDERS = {
    "gpt_oss": ModelConfig(
        name="openai/gpt-oss-120b",
        base_url="http://localhost:8000/v1",
        api_key="YOUR_API_KEY",
        model_id="openai/gpt-oss-120b",
    )
}

MAX_RETRIES = 3
RETRY_DELAY = 2


# ---------------------------------------------------------------------------
# Internal sample representation
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """
    Unified representation for both flat and conversation-format inputs.

    `system_prompt` is the original system message describing available
    tools — preserved so generation prompts can reference the actual tool
    inventory (critical for the missing_tool class).

    `history` is the trimmed history passed to the LLM as context.
    `full_history` is the untrimmed version, preserved for output records
    so the final 'full' dialogue field reflects the entire conversation.
    The fields below describe the LAST tool turn, which is the target
    of generation/modification.

    `used_fields` / `unused_fields` come from validate_coverage.py — they
    list JSON paths the original answer surfaced vs. dropped. Used to
    target hallucination corruption at fields visible in answers.
    """
    # System prompt from the original dialogue (tool definitions, etc.)
    system_prompt: str = ""

    # Trimmed history sent to the LLM (controlled by --history-turns)
    history: list[dict] = field(default_factory=list)
    # Full untrimmed history, kept for output
    full_history: list[dict] = field(default_factory=list)

    # Last tool turn — the target
    last_user_query: str = ""
    tool_call: str = ""
    tool_response: str = ""
    original_answer: str = ""

    # Optional coverage analysis from validate_coverage.py
    used_fields: list[str] = field(default_factory=list)
    unused_fields: list[str] = field(default_factory=list)

    # Metadata
    sample_id: str = ""


# ---------------------------------------------------------------------------
# Sample loaders for different input formats
# ---------------------------------------------------------------------------

FLAT_REQUIRED_COLUMNS = {
    "user_prompt", "tool_call", "tool_response", "original_answer"
}


def extract_coverage_fields(row: dict) -> tuple[list[str], list[str]]:
    """
    Pull used_fields / unused_fields from a coverage-report row if present.
    These are JSON-encoded as strings in the saved dataset.
    Returns ([], []) if not present or unparseable.
    """
    def _decode(key: str) -> list[str]:
        val = row.get(key)
        if val is None:
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []

    return _decode("used_fields"), _decode("unused_fields")


def parse_flat_sample(row: dict, idx: int) -> Sample | None:
    """Parse a flat-format row (single tool call, no history)."""
    used, unused = extract_coverage_fields(row)
    # For coverage-report rows, tool_response/answer fields are named differently
    tool_response = row.get("tool_response", "")
    answer = row.get("original_answer") or row.get("answer", "")
    user_prompt = row.get("user_prompt") or row.get("user_query", "")
    tool_call = row.get("tool_call", "")

    if not (user_prompt and tool_call and tool_response and answer):
        return None

    return Sample(
        system_prompt=row.get("system", ""),
        history=[],
        full_history=[],
        last_user_query=user_prompt,
        tool_call=tool_call,
        tool_response=tool_response,
        original_answer=answer,
        used_fields=used,
        unused_fields=unused,
        sample_id=str(row.get("sample_id", row.get("id", idx))),
    )


def trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """
    Keep only the most recent `max_turns` user-led turns from history.

    A "turn" starts at a user message and includes everything until
    the next user message (so it captures the assistant reply, plus
    any tool call + tool result + final assistant answer).

    max_turns=0 → empty history
    max_turns=N → last N user-led turns

    Examples:
      [user, assistant, user, assistant, user, assistant], max_turns=2
        → keeps the last 2 user-led blocks
        → [user, assistant, user, assistant]

      [user, assistant, tool, assistant, user, assistant], max_turns=1
        → [user, assistant]  (the tool turn dropped)
    """
    if max_turns <= 0:
        return []

    # Find indices of user messages
    user_indices = [i for i, m in enumerate(history) if m["role"] == "user"]

    if len(user_indices) <= max_turns:
        return history

    # Keep from the (len - max_turns)-th user message onwards
    keep_from = user_indices[-max_turns]
    return history[keep_from:]


def parse_conversation_sample(
    row: dict, idx: int, history_turns: int = 2,
) -> Sample | None:
    """
    Parse a conversation-format row.
    Walks the conversation, finds the LAST tool turn, and keeps the most
    recent `history_turns` user-led turns as context.

    Expected: row has a 'conversations' field (list of {"from", "value"})
    or 'messages' (list of {"role", "content"}).
    """
    # Normalize message format
    if "conversations" in row:
        msgs = []
        for m in row["conversations"]:
            role_map = {"user": "user", "assistant": "assistant", "tool": "tool"}
            role = role_map.get(m["from"])
            if role:
                msgs.append({"role": role, "content": m["value"]})
    elif "messages" in row:
        msgs = list(row["messages"])
    else:
        return None

    # Find the LAST tool turn: tool message followed by assistant answer
    last_tool_idx = None
    for i, m in enumerate(msgs):
        if m["role"] == "tool" and i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
            last_tool_idx = i

    if last_tool_idx is None:
        return None

    # Find the assistant tool_call that preceded this tool message
    tool_call_idx = None
    for i in range(last_tool_idx - 1, -1, -1):
        if msgs[i]["role"] == "assistant":
            tool_call_idx = i
            break
    if tool_call_idx is None:
        return None

    # Find the user query that preceded the tool call
    user_query_idx = None
    for i in range(tool_call_idx - 1, -1, -1):
        if msgs[i]["role"] == "user":
            user_query_idx = i
            break
    if user_query_idx is None:
        return None

    # Raw history = everything before the user_query that triggered the last tool turn
    raw_history = msgs[:user_query_idx]
    # Trim to last N user-led turns (for LLM context)
    history = trim_history(raw_history, history_turns)

    return Sample(
        system_prompt=row.get("system", ""),
        history=history,
        full_history=raw_history,  # untrimmed, for the 'full' output column
        last_user_query=msgs[user_query_idx]["content"],
        tool_call=msgs[tool_call_idx]["content"],
        tool_response=msgs[last_tool_idx]["content"],
        original_answer=msgs[last_tool_idx + 1]["content"],
        sample_id=str(row.get("id", row.get("dialogue_id", idx))),
    )


# ---------------------------------------------------------------------------
# Span parsing
# ---------------------------------------------------------------------------

def parse_spans(text: str, tag: str) -> dict:
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    spans = []
    clean_parts = []
    last_end = 0
    offset = 0

    for match in pattern.finditer(text):
        clean_parts.append(text[last_end:match.start()])
        span_text = match.group(1)
        start = match.start() - offset
        end = start + len(span_text)
        spans.append({"start": start, "end": end, "text": span_text})
        clean_parts.append(span_text)
        offset += len(f"<{tag}>") + len(f"</{tag}>")
        last_end = match.end()

    clean_parts.append(text[last_end:])
    return {"text": "".join(clean_parts), "spans": spans}


def parse_data_spans(text: str) -> dict:
    """Extract <DATA val="...">...</DATA> spans with source value mapping."""
    pattern = re.compile(r'<DATA\s+val="([^"]*)">(.*?)</DATA>', re.DOTALL)
    spans = []
    clean_parts = []
    last_end = 0
    offset = 0

    for match in pattern.finditer(text):
        clean_parts.append(text[last_end:match.start()])
        source_value = match.group(1)
        display_text = match.group(2)
        start = match.start() - offset
        end = start + len(display_text)

        spans.append({
            "start": start, "end": end,
            "text": display_text, "source_value": source_value,
        })

        clean_parts.append(display_text)
        offset += len(match.group(0)) - len(display_text)
        last_end = match.end()

    clean_parts.append(text[last_end:])
    return {"text": "".join(clean_parts), "data_spans": spans}


# ---------------------------------------------------------------------------
# Value anchoring & targeted corruption
# (replaces the old corrupt-then-diff flow — see generate_hallucination)
# ---------------------------------------------------------------------------

def parse_tool_response(raw: str):
    """Tolerant parse: strict JSON first, then Python-literal style (ToolACE
    tool turns frequently use single quotes / True / None). Non-JSON types
    from literal_eval (Ellipsis from '...', tuples, sets) are sanitized so
    the result round-trips through json.dumps."""
    import ast

    def sanitize(node):
        if isinstance(node, dict):
            return {str(k): sanitize(v) for k, v in node.items()}
        if isinstance(node, (list, tuple, set)):
            return [sanitize(v) for v in node]
        if node is Ellipsis:
            return "..."
        if isinstance(node, (str, int, float, bool)) or node is None:
            return node
        return str(node)

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return sanitize(ast.literal_eval(raw))
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def leaf_render(leaf) -> str:
    """Canonical JSON-style rendering of a scalar leaf (what a model reading
    the JSON would see): true/false/null, ints without .0, bare strings."""
    if leaf is None:
        return "null"
    if isinstance(leaf, bool):
        return "true" if leaf else "false"
    if isinstance(leaf, float) and leaf.is_integer():
        return str(int(leaf))
    if isinstance(leaf, (int, float)):
        return repr(leaf)
    return str(leaf)


def norm_val(s: str) -> str:
    """Normalization bridge between model-written val="..." attributes and
    leaf renderings: whitespace, surrounding quotes, true/false/null casing,
    thousands separators, trailing .0."""
    s = str(s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    low = s.lower()
    if low in ("true", "false", "null", "none"):
        return "null" if low == "none" else low
    num = s.replace(",", "") if re.fullmatch(r"-?[\d,]+(\.\d+)?", s) else s
    try:
        f = float(num)
        return str(int(f)) if f.is_integer() else repr(f)
    except (ValueError, OverflowError):
        pass
    return " ".join(s.split())



def _boundary_find(haystack: str, needle: str) -> bool:
    """Substring match guarded by alphanumeric word boundaries — '18' must
    NOT match inside '1840' (see editing pipeline known-fixed bugs)."""
    if not needle:
        return False
    pat = r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])"
    return re.search(pat, haystack) is not None


def _boundary_replace(haystack: str, needle: str, repl: str) -> tuple[str, int]:
    pat = r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])"
    new, n = re.subn(pat, lambda m: repl, haystack)
    return new, n


def value_occurs(obj, raw: str) -> bool:
    """True if `raw` matches a scalar leaf (normalized) or appears as a
    substring of a string leaf anywhere in obj."""
    target = norm_val(raw)
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
        else:
            if norm_val(leaf_render(node)) == target:
                return True
            if isinstance(node, str) and _boundary_find(node, raw):
                return True
    return False


def _cast_like(new_raw: str, old_leaf):
    """Cast the model's replacement string to the type of the leaf it
    replaces, falling back to string."""
    s = str(new_raw).strip()
    if isinstance(old_leaf, bool):
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        return old_leaf  # refuse nonsense bool replacement
    if isinstance(old_leaf, int) and not isinstance(old_leaf, bool):
        try:
            return int(float(s.replace(",", "")))
        except ValueError:
            return s
    if isinstance(old_leaf, float):
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return s
    return s


def apply_replacement(obj, old_raw: str, new_raw: str) -> int:
    """Replace every leaf matching old_raw (normalized scalar match, or
    substring within string leaves) with new_raw, type-preserved.
    Mutates obj in place; returns the number of leaves changed."""
    target = norm_val(old_raw)
    count = 0

    def visit(node):
        nonlocal count
        if isinstance(node, dict):
            for k in list(node.keys()):
                node[k] = descend(node[k])
        elif isinstance(node, list):
            for i in range(len(node)):
                node[i] = descend(node[i])
        return node

    def descend(leaf):
        nonlocal count
        if isinstance(leaf, (dict, list)):
            return visit(leaf)
        if norm_val(leaf_render(leaf)) == target:
            count += 1
            return _cast_like(new_raw, leaf)
        if isinstance(leaf, str):
            new_leaf, n = _boundary_replace(leaf, old_raw, str(new_raw))
            if n:
                count += n
                return new_leaf
        return leaf

    if isinstance(obj, (dict, list)):
        visit(obj)
    return count


def extract_json_object(text: str) -> dict | None:
    """Balanced-brace scan for the first top-level JSON object. Greedy regex
    breaks on braces inside string values — do not reintroduce it."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------
# Prompt building (with conversation history)
# ---------------------------------------------------------------------------

def format_history_block(history: list[dict]) -> str:
    """Render conversation history as a labeled text block for prompts."""
    if not history:
        return ""

    lines = ["CONVERSATION HISTORY (prior turns, for context only):"]
    role_labels = {
        "user": "USER", "assistant": "ASSISTANT", "tool": "TOOL_RESULT",
    }
    for m in history:
        label = role_labels.get(m["role"], m["role"].upper())
        lines.append(f"[{label}]: {m['content']}")
    return "\n".join(lines) + "\n\n"


def build_user_message(
    sample: Sample,
    task_instruction: str,
    include_coverage: bool = False,
) -> str:
    """
    Build a structured user message for the generation LLM.
    Includes the original system prompt (tool definitions), history if any,
    the last tool turn, and a task instruction.

    If include_coverage=True and the sample has coverage info, include
    USED_FIELDS / UNUSED_FIELDS hints so the model can be selective
    realistically (used for correct, overgen, undergen prompts).
    """
    parts = []

    if sample.system_prompt:
        parts.append(
            "ORIGINAL SYSTEM PROMPT (defines available tools and rules — "
            "for context only, do not act as that assistant):\n"
            f"{sample.system_prompt}\n"
        )

    if sample.history:
        parts.append(format_history_block(sample.history))

    parts.append(f"USER QUESTION:\n{sample.last_user_query}\n")
    parts.append(f"TOOL CALL:\n{sample.tool_call}\n")
    parts.append(f"TOOL RESPONSE:\n{sample.tool_response}\n")

    if include_coverage and (sample.used_fields or sample.unused_fields):
        coverage_block = "FIELD GUIDANCE (from prior coverage analysis):\n"
        if sample.used_fields:
            coverage_block += (
                f"  Fields typically surfaced in answers to this kind of "
                f"question: {sample.used_fields}\n"
            )
        if sample.unused_fields:
            coverage_block += (
                f"  Fields typically NOT surfaced (metadata, IDs, etc.): "
                f"{sample.unused_fields}\n"
            )
        parts.append(coverage_block)

    parts.append(task_instruction)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

HALLUCINATION_STEP1_SYSTEM = """\
You are a data-generation assistant for an LLM evaluation benchmark.

You are given a tool's JSON response and a list of CANDIDATE VALUES — the
values from that response which the assistant's answer actually surfaced.

Your task: pick 1 to 3 of the candidate values and invent a plausible
replacement for each, so the data becomes factually different.

Rules:
1. "old" must be copied VERBATIM from the candidate list — do not
   reformat, round, or re-quote it.
2. "new" must be the same kind of value (number for number, date for
   date, name for name) and plausible for the field — subtle, not absurd.
3. "new" must be genuinely different from "old".
4. Pick between 1 and 3 candidates. Never invent values not in the list.

Respond with ONLY this JSON object, no explanations or markdown fences:
{"replacements": [{"old": "<verbatim candidate>", "new": "<replacement>"}]}"""

CORRECT_SYSTEM = """\
You are a helpful AI assistant. The user asked a question and a tool was
called to get the answer. Based on the tool response provided, write a
clear, helpful answer to the LAST user question.

If conversation history is provided, treat it as background context — do not
re-answer earlier questions, only respond to the most recent user query
about the tool result you have just received.

KEY PRINCIPLE — SELECTIVITY:
A good answer surfaces what's RELEVANT to the user's question. Tool responses
often contain metadata (timestamps, IDs, coordinates, internal status flags,
debugging info) that a real assistant would NOT mention. Mention only the
fields that genuinely answer the user's question or add meaningful context.

Examples of when to OMIT fields:
- User asked "what's the temperature?" → mention temperature, skip humidity,
  pressure, station_id, coordinates unless directly asked.
- User asked "find me events" → mention event names + dates, skip internal
  event_ids and end_time UTC offsets unless useful.
- User asked "details about zip 10001" → mention city/state; coordinates
  are usually overkill unless they explicitly asked for location data.

If a field provides MEANINGFUL context for the user's intent (not just
"data is data"), include it. When in doubt about a non-essential field,
prefer leaving it out.

Rules:
1. Use a natural, helpful tone. Match the depth of the user's question.
2. Faithfully convey the relevant fields — do NOT change their values.
3. Do NOT add information beyond what the tool returned (no tips,
   historical context, comparisons, or elaborations).
4. Do NOT offer to use any other tools or capabilities.

CRITICAL FORMATTING RULE:
Wrap every value you DO surface from the tool response in
<DATA val="EXACT_VALUE">display text</DATA> tags, where:
- EXACT_VALUE is the raw value copied verbatim from the tool JSON
- display text is your natural-language rendering of that value

Apply this to all data values you include: names, numbers, dates, IDs,
statuses, descriptions. Connecting words and sentence structure should NOT
be wrapped. Fields you choose to OMIT need no tags (they're not in the text).

Example:
  User asked: "What's the current S&P 500?"
  Tool response: {"name": "S&P 500", "value": "4172.80", "change": "+0.68%", "timestamp": "2025-01-15T10:00:00Z", "exchange_id": "NYS001"}
  Your output: The <DATA val="S&P 500">S&P 500</DATA> is currently at \
<DATA val="4172.80">4,172.80</DATA>, up <DATA val="+0.68%">+0.68%</DATA>.

Note how `timestamp` and `exchange_id` are omitted — the user didn't ask
about them and they'd clutter the answer.

Respond with ONLY the annotated assistant message."""

OVERGENERATION_SYSTEM = """\
You are a data-generation assistant for an LLM evaluation benchmark.

Your task: given a user question, a tool call, and the tool's JSON response,
generate an assistant reply to the LAST user question that correctly states
the tool results but also adds extra information NOT present in the tool
response.

If conversation history is provided, treat it as background only.

Rules:
1. First, faithfully convey the key data from the tool response.
2. Then add plausible-sounding but UNSUPPORTED details: historical context,
   general knowledge, tips, comparisons, or elaborations.
3. Do NOT contradict any values in the tool response.
4. Do NOT reference or offer to use any other tools or capabilities.
5. Keep the added content to 1-3 extra sentences.

CRITICAL FORMATTING RULE:
Wrap ALL added/unsupported content in <OVER> tags. Content that faithfully
represents the tool response should NOT be wrapped.

Example:
  Tool response: {"weather": "sunny", "temp": "25°C"}
  Your output: The weather is sunny with a temperature of 25°C. \
<OVER>This is typical for this region during summer, making it a \
great time to visit outdoor attractions.</OVER>

Respond with ONLY the annotated assistant message."""

MISSING_TOOL_SYSTEM = """\
You are a data-generation assistant for an LLM evaluation benchmark.

Your task: given a user question, a tool call, and the tool's JSON response,
generate an assistant reply to the LAST user question that correctly states
the tool results but then offers to perform an action using a tool that
does NOT exist.

If conversation history is provided, treat it as background only.

CRITICAL: If an "ORIGINAL SYSTEM PROMPT" is provided, it lists the tools
that DO exist in this system. Your fabricated capability MUST be a tool
or service NOT listed there. Do NOT invent capabilities that match any
existing tool's purpose — pick something genuinely outside the available
toolset (booking, scheduling, sending messages, executing transactions,
generating media, etc., as long as no listed tool covers it).

Rules:
1. First, faithfully convey the key data from the tool response.
2. Then suggest an additional action requiring a DIFFERENT, non-existent
   tool or service. Make the offer sound natural and related to context.
3. Do NOT contradict any values from the tool response.
4. Do NOT add unsupported factual claims about the tool's data.

CRITICAL FORMATTING RULE:
Wrap ALL references to non-existent capabilities in <MISSING> tags.

Example:
  Available tools: Weather_API
  Tool response: {"weather": "sunny"}
  Your output: The weather is sunny today. <MISSING>Would you like me to \
book a restaurant with outdoor seating for you?</MISSING>

Respond with ONLY the annotated assistant message."""

UNDERGENERATION_SYSTEM = """\
You are a data-generation assistant for an LLM evaluation benchmark.

Your task: given a user question, a tool call, and the tool's JSON response,
generate an assistant reply that omits significant RELEVANT information
from the tool response while appearing complete.

KEY DISTINCTION:
A good answer is naturally SELECTIVE — it skips metadata and irrelevant
fields. That is NOT undergeneration. Undergeneration means dropping
information the user actually wanted or that's central to answering their
question. Examples:

  Good selectivity (NOT undergeneration):
    User: "what's the temperature?" → answer skips coordinates, timestamp.
  Undergeneration:
    User: "list all events" → answer skips half the events.
    User: "weather forecast for tomorrow" → answer gives temperature
      but skips precipitation and wind which the forecast includes.
    User: "give me details on this product" → answer skips price.

Rules:
1. Identify what fields are RELEVANT to the user's question.
2. Drop at least one of those relevant fields or items, without
   acknowledging the omission.
3. If the tool returns a list of items the user asked to enumerate,
   drop at least one entire item.
4. Do NOT contradict any values you DO include.
5. Do NOT add information beyond what the tool returned.
6. Do NOT reference or offer to use any other tools.
7. The response should read as if it's complete — don't say "here are some"
   or hint that information was left out.

Additionally, output a JSON block at the END of your response listing
exactly what was omitted that should have been included, in this format:

<OMITTED>
{"omitted_items": ["description of each relevant piece of information that was dropped"]}
</OMITTED>

Only list items that a complete answer to the user's question SHOULD
contain. Do not list metadata that would naturally be omitted.

Example:
  User: "Find me upcoming AI events"
  Tool response: {"events": [{"id": 1, "name": "A", "date": "Jan 1"}, \
{"id": 2, "name": "B", "date": "Feb 2"}]}
  Your output: Here is the upcoming AI event:

  1. **A** — January 1st

  <OMITTED>
  {"omitted_items": ["Event B (id: 2, date: Feb 2) was completely dropped — user asked for events plural"]}
  </OMITTED>

Respond with ONLY the assistant message followed by the OMITTED block."""


# ---------------------------------------------------------------------------
# Async LLM calls
# ---------------------------------------------------------------------------

async def call_llm(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    config: ModelConfig,
    system: str,
    user: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str | None:
    """Single LLM call with concurrency control and retries.

    Reasoning models (gpt-oss, Qwen 3+) can return None content when they
    exhaust their token budget on reasoning. We treat that as a retryable
    error and bump max_tokens by default to give them headroom.
    """
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=config.model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                choice = response.choices[0]
                content = choice.message.content
                finish = choice.finish_reason

                # Reasoning models may produce reasoning tokens but no
                # final content if they hit the token limit
                if content is None:
                    print(
                        f"  [attempt {attempt+1}/{MAX_RETRIES}] "
                        f"Empty content (finish_reason={finish}). "
                        f"Model may have exhausted tokens on reasoning."
                    )
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    continue

                return content.strip()

            except Exception as e:
                print(f"  [attempt {attempt+1}/{MAX_RETRIES}] Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Per-class generators
# ---------------------------------------------------------------------------

def base_record(sample: Sample, error_class: str, idx: int) -> dict:
    """Common fields for output records."""
    return {
        "id": f"{error_class}_{idx}",
        "class": error_class,
        "sample_id": sample.sample_id,
        "system_prompt": sample.system_prompt,
        "history": sample.history,
        "full_history": sample.full_history,
        "last_user_query": sample.last_user_query,
        "tool_call": sample.tool_call,
        "tool_response": sample.tool_response,
        "original_answer": sample.original_answer,
        "used_fields": sample.used_fields,
        "unused_fields": sample.unused_fields,
    }


async def generate_correct(
    sem: asyncio.Semaphore, cli: AsyncOpenAI, cfg: ModelConfig,
    sample: Sample, idx: int,
) -> dict | None:
    raw = await call_llm(
        sem, cli, cfg, CORRECT_SYSTEM,
        build_user_message(
            sample,
            "Write a helpful, data-annotated answer to the last user question.",
            include_coverage=True,
        ),
        temperature=0.3,
    )
    if not raw:
        return None

    parsed = parse_data_spans(raw)
    if not parsed["data_spans"]:
        print(f"  [WARN] correct_{idx}: no DATA spans found, skipping")
        return None

    rec = base_record(sample, "correct", idx)
    rec["generated_answer"] = parsed["text"]
    rec["data_spans"] = parsed["data_spans"]
    rec["spans"] = []
    return rec


async def generate_hallucination(
    sem: asyncio.Semaphore, cli: AsyncOpenAI, cfg: ModelConfig,
    sample: Sample, idx: int,
    correct_result: dict | None = None,
) -> dict | None:
    """Span-targeted corruption (same lesson as the editing pipeline's
    1.1 -> 1.2 move). Flow:
      1. anchor: keep data_spans whose source_value is actually locatable
         in the parsed original tool response (normalized match)
      2. LLM picks 1-3 candidates and invents replacements (tiny JSON)
      3. replacements are applied PROGRAMMATICALLY -> corrupted response is
         valid JSON by construction
      4. hal_spans = data_spans whose source_value was replaced -> the span
         match is guaranteed by construction
    Distinct WARN codes per failure mode so yield problems stay diagnosable.
    """
    if not correct_result or not correct_result.get("data_spans"):
        print(f"  [WARN] hallucination_{idx}: no-correct-result, skipping")
        return None

    orig_json = parse_tool_response(sample.tool_response)
    if orig_json is None or not isinstance(orig_json, (dict, list)):
        print(f"  [WARN] hallucination_{idx}: unparseable-tool-response, skipping")
        return None

    data_spans = correct_result["data_spans"]

    # Step 1: anchor — candidates are source_values locatable in the response.
    # Fallback: if val="..." doesn't anchor, try the span's display text
    # (models occasionally swap the two).
    candidates, seen, unanchored = [], set(), []
    for s in data_spans:
        for sv in (s.get("source_value", ""), s.get("text", "")):
            key = norm_val(sv)
            if not key or key in seen:
                continue
            if value_occurs(orig_json, sv):
                seen.add(key)
                candidates.append(sv)
                break
        else:
            if s.get("source_value"):
                unanchored.append(s["source_value"])
    if not candidates:
        examples = ", ".join(repr(u)[:40] for u in unanchored[:3])
        print(f"  [WARN] hallucination_{idx}: no-anchorable-values "
              f"(e.g. {examples}), skipping")
        return None

    # Step 2: LLM proposes replacements among the candidates
    candidate_block = "\n".join(f"  - {c}" for c in candidates)
    user_prompt = (
        f"TOOL RESPONSE (JSON):\n{json.dumps(orig_json, ensure_ascii=False)}\n\n"
        f"CANDIDATE VALUES (copy 'old' verbatim from this list):\n"
        f"{candidate_block}\n\n"
        f"Return the replacements JSON object."
    )
    raw = await call_llm(
        sem, cli, cfg, HALLUCINATION_STEP1_SYSTEM, user_prompt,
        temperature=0.7,
    )
    if not raw:
        print(f"  [WARN] hallucination_{idx}: llm-no-output, skipping")
        return None

    parsed = extract_json_object(raw)
    replacements = (parsed or {}).get("replacements")
    if not isinstance(replacements, list) or not replacements:
        print(f"  [WARN] hallucination_{idx}: llm-bad-replacements, skipping")
        return None

    # Step 3: apply programmatically, keeping only real, anchored changes
    candidate_norms = {norm_val(c) for c in candidates}
    corrupted_json = json.loads(json.dumps(orig_json))  # deep copy
    changed_norms: set[str] = set()
    for r in replacements[:3]:
        if not isinstance(r, dict):
            continue
        old, new = str(r.get("old", "")), str(r.get("new", ""))
        if not old or not new:
            continue
        key = norm_val(old)
        if key not in candidate_norms or norm_val(new) == key:
            continue  # invented value or non-change
        if apply_replacement(corrupted_json, old, new) > 0:
            changed_norms.add(key)
    if not changed_norms:
        print(f"  [WARN] hallucination_{idx}: no-applied-replacements, skipping")
        return None

    # Step 4: spans follow from the applied replacements by construction.
    # Check both norms — the anchor may have come from the display-text
    # fallback in Step 1.
    answer = correct_result["generated_answer"]
    hal_spans = [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in data_spans
        if norm_val(s.get("source_value", "")) in changed_norms
        or norm_val(s.get("text", "")) in changed_norms
    ]
    if not hal_spans:
        print(f"  [WARN] hallucination_{idx}: no-spans-matched, skipping")
        return None

    rec = base_record(sample, "hallucination", idx)
    # Override tool_response with the corrupted version (the "context")
    rec["tool_response"] = json.dumps(corrupted_json, ensure_ascii=False)
    rec["original_tool_response"] = sample.tool_response
    rec["generated_answer"] = answer
    rec["spans"] = hal_spans
    return rec


async def generate_overgeneration(
    sem: asyncio.Semaphore, cli: AsyncOpenAI, cfg: ModelConfig,
    sample: Sample, idx: int,
) -> dict | None:
    raw = await call_llm(
        sem, cli, cfg, OVERGENERATION_SYSTEM,
        build_user_message(
            sample, "Generate the annotated assistant reply.",
            include_coverage=True,
        ),
    )
    if not raw:
        return None

    parsed = parse_spans(raw, "OVER")
    if not parsed["spans"]:
        print(f"  [WARN] overgeneration_{idx}: no spans found, skipping")
        return None

    rec = base_record(sample, "overgeneration", idx)
    rec["generated_answer"] = parsed["text"]
    rec["spans"] = parsed["spans"]
    return rec


async def generate_missing_tool(
    sem: asyncio.Semaphore, cli: AsyncOpenAI, cfg: ModelConfig,
    sample: Sample, idx: int,
) -> dict | None:
    raw = await call_llm(
        sem, cli, cfg, MISSING_TOOL_SYSTEM,
        build_user_message(
            sample, "Generate the annotated assistant reply.",
            include_coverage=True,
        ),
    )
    if not raw:
        return None

    parsed = parse_spans(raw, "MISSING")
    if not parsed["spans"]:
        print(f"  [WARN] missing_tool_{idx}: no spans found, skipping")
        return None

    rec = base_record(sample, "missing_tool", idx)
    rec["generated_answer"] = parsed["text"]
    rec["spans"] = parsed["spans"]
    return rec


async def generate_undergeneration(
    sem: asyncio.Semaphore, cli: AsyncOpenAI, cfg: ModelConfig,
    sample: Sample, idx: int,
) -> dict | None:
    raw = await call_llm(
        sem, cli, cfg, UNDERGENERATION_SYSTEM,
        build_user_message(
            sample, "Generate the assistant reply with the OMITTED block.",
            include_coverage=True,
        ),
    )
    if not raw:
        return None

    omitted_match = re.search(r"<OMITTED>\s*(.*?)\s*</OMITTED>", raw, re.DOTALL)
    omitted_items = []
    if omitted_match:
        try:
            omitted_data = json.loads(omitted_match.group(1))
            omitted_items = omitted_data.get("omitted_items", [])
        except json.JSONDecodeError:
            print(f"  [WARN] undergeneration_{idx}: couldn't parse OMITTED")

    clean_answer = re.sub(
        r"\s*<OMITTED>.*?</OMITTED>\s*", "", raw, flags=re.DOTALL
    ).strip()

    if not omitted_items:
        print(f"  [WARN] undergeneration_{idx}: no omitted items, skipping")
        return None

    rec = base_record(sample, "undergeneration", idx)
    rec["generated_answer"] = clean_answer
    rec["omitted_items"] = omitted_items
    return rec


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self, total: int, label: str):
        self.total = total
        self.done = 0
        self.failed = 0
        self.label = label
        self.start_time = time.time()

    def tick(self, success: bool):
        self.done += 1
        if not success:
            self.failed += 1
        elapsed = time.time() - self.start_time
        rate = self.done / elapsed if elapsed > 0 else 0
        eta = (self.total - self.done) / rate if rate > 0 else 0
        print(
            f"\r  [{self.label}] {self.done}/{self.total} "
            f"({self.failed} failed) | {rate:.1f} samples/s | "
            f"ETA: {eta:.0f}s",
            end="", flush=True,
        )

    def finish(self):
        elapsed = time.time() - self.start_time
        print(
            f"\n  [{self.label}] Done: {self.done - self.failed}/{self.total} "
            f"succeeded in {elapsed:.1f}s"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

INDEPENDENT_GENERATORS = {
    "overgeneration": generate_overgeneration,
    "missing_tool": generate_missing_tool,
    "undergeneration": generate_undergeneration,
}


async def generate_independent_class(
    sem, cli, cfg, samples, error_class, samples_per_class,
) -> list[dict]:
    subset = samples
    if samples_per_class and samples_per_class < len(samples):
        subset = random.sample(samples, samples_per_class)

    print(f"\n{'='*60}")
    print(f"Generating class: {error_class} ({len(subset)} samples)")
    print(f"{'='*60}")

    generator = INDEPENDENT_GENERATORS[error_class]
    progress = ProgressTracker(len(subset), error_class)

    async def wrapped(sample, idx):
        result = await generator(sem, cli, cfg, sample, idx)
        progress.tick(success=result is not None)
        return result

    tasks = [wrapped(s, i) for i, s in enumerate(subset)]
    raw = await asyncio.gather(*tasks)
    progress.finish()
    return [r for r in raw if r is not None]


async def generate_correct_class(
    sem, cli, cfg, samples, samples_per_class,
) -> list[dict]:
    # Keep GLOBAL indices when subsampling — record ids ("correct_<idx>")
    # are the pairing key for the hallucination class, so they must index
    # into the full `samples` list, never into the random subset.
    indexed = list(enumerate(samples))
    if samples_per_class and samples_per_class < len(indexed):
        indexed = random.sample(indexed, samples_per_class)

    print(f"\n{'='*60}")
    print(f"Generating class: correct ({len(indexed)} samples)")
    print(f"{'='*60}")

    progress = ProgressTracker(len(indexed), "correct")

    async def wrapped(sample, idx):
        result = await generate_correct(sem, cli, cfg, sample, idx)
        progress.tick(success=result is not None)
        return result

    tasks = [wrapped(s, i) for i, s in indexed]
    raw = await asyncio.gather(*tasks)
    progress.finish()
    return [r for r in raw if r is not None]


async def generate_hallucination_class(
    sem, cli, cfg, samples, correct_results, samples_per_class,
) -> list[dict]:
    correct_by_id = {}
    for cr in correct_results:
        idx = int(cr["id"].split("_")[1])
        correct_by_id[idx] = cr

    # Pair by global index AND verify identity via sample_id — a silent
    # cross-dialogue pairing produces mislabeled rows, which is worse than
    # a skip. (This is the bug that made --samples-per-class runs fail
    # with 'no-anchorable-values' on nearly every row.)
    candidates, mismatched = [], 0
    for i, s in enumerate(samples):
        cr = correct_by_id.get(i)
        if cr is None:
            continue
        if str(cr.get("sample_id")) != str(s.sample_id):
            mismatched += 1
            continue
        candidates.append((s, i, cr))
    if mismatched:
        print(f"  [WARN] hallucination pairing: {mismatched} correct results "
              f"had sample_id mismatch and were skipped — "
              f"stale correct_results from a different run/subset?")

    if samples_per_class and samples_per_class < len(candidates):
        candidates = random.sample(candidates, samples_per_class)

    print(f"\n{'='*60}")
    print(f"Generating class: hallucination ({len(candidates)} samples)")
    print(f"{'='*60}")

    progress = ProgressTracker(len(candidates), "hallucination")

    async def wrapped(sample, idx, correct_result):
        result = await generate_hallucination(
            sem, cli, cfg, sample, idx, correct_result
        )
        progress.tick(success=result is not None)
        return result

    tasks = [wrapped(s, i, cr) for s, i, cr in candidates]
    raw = await asyncio.gather(*tasks)
    progress.finish()
    return [r for r in raw if r is not None]


async def process_dataset(
    samples: list[Sample],
    classes: list[str],
    config: ModelConfig,
    concurrency: int,
    samples_per_class: int | None = None,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
    results = []

    correct_results = []
    if "correct" in classes or "hallucination" in classes:
        correct_results = await generate_correct_class(
            semaphore, client, config, samples, samples_per_class
        )
        if "correct" in classes:
            results.extend(correct_results)

    if "hallucination" in classes:
        hal_results = await generate_hallucination_class(
            semaphore, client, config, samples, correct_results,
            samples_per_class,
        )
        results.extend(hal_results)

    independent = [c for c in classes if c in INDEPENDENT_GENERATORS]
    for error_class in independent:
        class_results = await generate_independent_class(
            semaphore, client, config, samples, error_class, samples_per_class
        )
        results.extend(class_results)

    return results


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

def load_input_dataset(
    path: str, fmt: str, history_turns: int = 2,
) -> list[Sample]:
    """Load and parse input dataset into Sample objects."""
    p = Path(path)

    if p.is_dir() and (p / "dataset_info.json").exists():
        print(f"Loading HF dataset from disk: {path}")
        ds = load_from_disk(path)
    elif p.is_file():
        ext = p.suffix.lower()
        fmt_map = {
            ".json": "json", ".jsonl": "json",
            ".csv": "csv", ".parquet": "parquet",
        }
        loader_fmt = fmt_map.get(ext)
        if not loader_fmt:
            raise ValueError(f"Unsupported file format: {ext}")
        print(f"Loading {loader_fmt} file: {path}")
        ds = load_dataset(loader_fmt, data_files=str(p), split="train")
    else:
        print(f"Loading from HuggingFace Hub: {path}")
        ds = load_dataset(path, split="train")

    print(f"Raw dataset: {len(ds)} rows, columns: {ds.column_names}")

    # Auto-detect format if not specified
    if fmt == "auto":
        cols = set(ds.column_names)
        if FLAT_REQUIRED_COLUMNS.issubset(cols):
            fmt = "flat"
        elif "user_query" in cols and "answer" in cols and "tool_response" in cols:
            # Coverage-report shape from validate_coverage.py
            fmt = "flat"
            print("(Detected coverage-report format)")
        elif "conversations" in cols or "messages" in cols:
            fmt = "conversation"
        else:
            raise ValueError(
                f"Cannot auto-detect format. Columns: {ds.column_names}. "
                f"Expected either {FLAT_REQUIRED_COLUMNS}, coverage-report "
                f"columns, or 'conversations'/'messages'."
            )
        print(f"Auto-detected format: {fmt}")

    if fmt == "flat":
        parse = lambda row, i: parse_flat_sample(row, i)
    else:
        parse = lambda row, i: parse_conversation_sample(row, i, history_turns)

    samples = []
    skipped = 0
    for i, row in enumerate(ds):
        sample = parse(dict(row), i)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    print(f"Parsed {len(samples)} valid samples ({skipped} skipped)")
    if fmt == "conversation":
        print(f"History trimmed to last {history_turns} user-led turns")
    return samples


def build_full_dialogue(record: dict) -> list[dict]:
    """
    Reconstruct the complete dialogue for a result record:
      [system] + full_history... + last_user_query + tool_call
              + tool_response + generated_answer

    Uses the UNTRIMMED full_history so the resulting dialogue contains the
    entire conversation, not just the trimmed slice that was sent to the LLM.

    The generated_answer is the modified/erroneous content for non-correct
    classes, so the dialogue represents the dataset point as it would be
    consumed by a classifier.
    """
    full = []

    # Original system prompt (tool definitions, etc.)
    system = record.get("system_prompt") or ""
    if system:
        full.append({"role": "system", "content": system})

    full.extend(record.get("full_history") or record.get("history", []))

    full.append({
        "role": "user",
        "content": record.get("last_user_query", ""),
    })
    full.append({
        "role": "assistant",
        "content": record.get("tool_call", ""),
    })
    full.append({
        "role": "tool",
        "content": record.get("tool_response", ""),
    })
    full.append({
        "role": "assistant",
        "content": record.get("generated_answer", ""),
    })
    return full


def save_output_dataset(
    results: list[dict], output_path: str, push_to_hub: bool = False,
):
    """Save results as HF Dataset, JSON-encoding nested fields."""
    NESTED_FIELDS = {
        "spans", "data_spans", "omitted_items",
        "history", "full_history", "full",
        "used_fields", "unused_fields",
    }

    # Attach the reconstructed full dialogue to each record
    for r in results:
        r["full"] = build_full_dialogue(r)

    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    normalized = []
    for r in results:
        row = {}
        for key in all_keys:
            val = r.get(key)
            if key in NESTED_FIELDS and isinstance(val, (list, dict)):
                row[key] = json.dumps(val, ensure_ascii=False)
            else:
                row[key] = val
        normalized.append(row)

    ds = Dataset.from_list(normalized)

    if push_to_hub:
        print(f"Pushing to HuggingFace Hub: {output_path}")
        ds.push_to_hub(output_path)
    else:
        print(f"Saving to disk: {output_path}")
        ds.save_to_disk(output_path)

    return ds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate span-annotated synthetic error data"
    )
    parser.add_argument(
        "--input", required=True,
        help="HF Hub ID, local HF dataset dir, or file (.json/.jsonl/.csv/.parquet)",
    )
    parser.add_argument("--output", default="./generated_errors")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--format", default="auto",
        choices=["auto", "flat", "conversation"],
        help="Input format. 'flat' = single tool call columns, "
             "'conversation' = multi-turn with 'conversations' field",
    )
    parser.add_argument(
        "--history-turns", type=int, default=2,
        help="For conversation format: number of recent user-led turns to "
             "keep as context. 0 = no history. Default: 2.",
    )
    parser.add_argument("--classes", default="all")
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=10)

    parser.add_argument(
        "--model", default="gpt-oss",
        help=f"Pre-defined provider: {list(PROVIDERS.keys())}, or 'custom'",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--api-key", default=None)

    args = parser.parse_args()

    # Resolve model config
    if args.model == "custom":
        if not args.base_url or not args.model_name:
            parser.error("--model custom requires --base-url and --model-name")
        config = ModelConfig(
            name=f"Custom ({args.model_name})",
            base_url=args.base_url,
            api_key=args.api_key or "no-key",
            model_id=args.model_name,
        )
    elif args.model in PROVIDERS:
        config = PROVIDERS[args.model]
        if args.api_key:
            config.api_key = args.api_key
        if args.base_url:
            config.base_url = args.base_url
        if args.model_name:
            config.model_id = args.model_name
    else:
        parser.error(f"Unknown model '{args.model}'")

    all_classes = [
        "correct", "hallucination", "overgeneration",
        "missing_tool", "undergeneration",
    ]
    if args.classes == "all":
        classes = all_classes
    else:
        classes = [c.strip() for c in args.classes.split(",")]
        invalid = set(classes) - set(all_classes)
        if invalid:
            parser.error(f"Unknown classes: {invalid}")

    samples = load_input_dataset(args.input, args.format, args.history_turns)

    # Show stats about samples
    with_history = sum(1 for s in samples if s.history)
    print(f"Samples with conversation history: {with_history} / {len(samples)}")
    print(f"Model: {config.name} ({config.model_id})")
    print(f"Concurrency: {args.concurrency}")

    results = asyncio.run(
        process_dataset(
            samples, classes, config, args.concurrency, args.samples_per_class
        )
    )

    ds = save_output_dataset(results, args.output, args.push_to_hub)

    from collections import Counter
    counts = Counter(r["class"] for r in results)
    print(f"\n{'='*60}")
    print(f"Saved {len(results)} samples → {args.output}")
    print(f"Dataset: {ds}")
    print(f"Distribution: {dict(counts)}")

    for cls in classes:
        cls_results = [r for r in results if r["class"] == cls]
        if not cls_results:
            print(f"  {cls}: 0 samples")
        elif cls == "undergeneration":
            avg = sum(len(r.get("omitted_items", [])) for r in cls_results)
            avg = avg / len(cls_results)
            print(f"  {cls}: avg {avg:.1f} omitted items/sample")
        elif cls == "correct":
            avg = sum(len(r.get("data_spans", [])) for r in cls_results)
            avg = avg / len(cls_results)
            print(f"  {cls}: avg {avg:.1f} data spans/sample")
        else:
            total = sum(len(r.get("spans", [])) for r in cls_results)
            avg = total / len(cls_results)
            print(f"  {cls}: avg {avg:.1f} spans/sample")


if __name__ == "__main__":
    main()
