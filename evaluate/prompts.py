"""
Prompt templates for hallucination detection evaluation.

Task: Given a tool-calling dialogue, identify:
  - the hallucination TYPE: clean | answer_mismatch | missing_tool | overgeneration | undergeneration
  - the hallucinated SPAN(s) in the final assistant answer (start/end character offsets + text)

Hallucination type definitions:
  - clean:            No hallucination. The final answer is fully grounded in the tool output.
  - answer_mismatch:  The answer directly contradicts or substitutes values from the tool response
                      (wrong name, wrong number, wrong entity — a misread/swap from the tool output).
  - missing_tool:     The answer contains fabricated facts NOT present in the tool response
                      (Baseless Info sub-type), OR appends an unsolicited proactive offer/question
                      not requested by the user (Proactive Follow-up sub-type).
  - overgeneration:   The answer adds unsupported elaboration, commentary, or claims beyond what
                      any tool returned — without directly contradicting the tool output.
  - undergeneration:  The answer OMITS information that the tool returned. This is an absence, not
                      a fabrication, so span_labels will always be EMPTY [] for this type.
"""

# ---------------------------------------------------------------------------
# Zero-shot system prompt
# ---------------------------------------------------------------------------

ZERO_SHOT_SYSTEM = """\
You are an expert evaluator of tool-calling dialogues for hallucination detection.

You will be given a dialogue that includes:
- A system prompt describing available tools
- A conversation between a user, an assistant (which may call tools), and tool responses
- The final assistant answer to evaluate

Your task is to detect hallucinations in the FINAL assistant answer and classify the row.

## Hallucination type definitions

- **clean**: The final answer is fully grounded in the tool output. No hallucinated content.

- **answer_mismatch**: The answer directly CONTRADICTS or SUBSTITUTES values from the tool response.
  For example: the tool returned "Gonzaga" but the answer says "Miami"; the tool returned year 2022
  but the answer says 2023. The tool response IS present — the assistant misread or swapped its data.
  SPAN RULE: mark ONLY the wrong value itself (e.g. "Miami", "2023", "7.5") — NOT the words before
  it (not "the team is", not " is", not "SD: "). The span must be the minimal substituted token.

- **missing_tool**: The answer contains fabricated concrete facts (IDs, names, amounts, dates) that
  are NOT present in or derivable from the tool response (Baseless Info); OR appends an unsolicited
  proactive offer / question not grounded in any tool result or user request (Proactive Follow-up,
  e.g. "I can help you with more actions if you'd like", "I'll go ahead and...", "Need a ... ?").
  A tool response IS always present.
  SPAN RULE for Proactive Follow-up: the hallucinated span is almost always the FINAL sentence(s)
  of the answer — the unsolicited offer or question appended at the end. Look at the very end of
  final_answer first. Do NOT mark content from the middle of the answer unless it contains
  fabricated facts (Baseless Info) that are absent from the tool response.

- **overgeneration**: The answer adds unsupported elaboration, commentary, interpretation, or claims
  that go beyond what the tool returned, without directly contradicting it.

- **undergeneration**: The answer OMITS information that the tool returned. It is an absence, not a
  fabrication. For this type, span_labels MUST be empty [].

## Output format

Respond with valid JSON and NOTHING else. Schema:

{
  "type": "<clean|answer_mismatch|missing_tool|overgeneration|undergeneration>",
  "span_labels": [
    {
      "start": <int, char offset in final_answer>,
      "end": <int, char offset in final_answer, exclusive>,
      "text": "<exact substring from final_answer>"
    }
  ],
  "reasoning": "<one or two sentence explanation>"
}

Rules for span_labels:
- Provide spans ONLY for types: answer_mismatch, missing_tool, overgeneration.
- For `clean` and `undergeneration` the list MUST be empty [].
- `start` and `end` are character offsets into `final_answer` (0-indexed, end is exclusive).
- `text` must be the exact substring final_answer[start:end].
- Mark only the MINIMAL hallucinated text — do not over-span:
  * answer_mismatch → mark only the wrong token/value, not surrounding words
    BAD:  {"start":24,"end":27,"text":" is"}   for "The area is 7.5"
    GOOD: {"start":28,"end":31,"text":"7.5"}
  * missing_tool (Proactive Follow-up) → mark the final unsolicited sentence(s) at the END
    BAD:  a span from the middle of the answer
    GOOD: the last sentence like "I can also do X if you'd like." or "Need a summary?"
- You may return multiple spans if there are multiple hallucinated regions.
"""

# ---------------------------------------------------------------------------
# Few-shot examples (one per class)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    # Example 1: clean
    {
        "role": "user",
        "content": """\
Evaluate this dialogue for hallucinations.

System prompt: You are an expert in composing functions.

Conversation:
[user]: I need to convert a web page to PDF. The URL is 'https://www.example.com'. Also provide the favicon.
[assistant]: [Convert Web Page to PDF(url="https://www.example.com"), Get Favicon(url="https://www.example.com")]
[tool]: [{"name": "Convert Web Page to PDF", "results": {"pdf_url": "https://files.example.com/presentation-example.pdf", "file_size": "1.2 MB"}}, {"name": "Get Favicon", "results": {"favicon_url": "https://files.example.com/favicon.ico", "size": "16x16"}}]
[assistant]: I've converted the web page to PDF. You can download it here: [Download PDF](https://files.example.com/presentation-example.pdf). File size: 1.2 MB. Favicon: [Download Favicon](https://files.example.com/favicon.ico), size 16x16 pixels.

final_answer: "I've converted the web page to PDF. You can download it here: [Download PDF](https://files.example.com/presentation-example.pdf). File size: 1.2 MB. Favicon: [Download Favicon](https://files.example.com/favicon.ico), size 16x16 pixels."
""",
    },
    {
        "role": "assistant",
        "content": """\
{"type": "clean", "span_labels": [], "reasoning": "All values in the final answer (pdf_url, file_size, favicon_url, size) are directly present in the tool response. No hallucinated content."}""",
    },
    # Example 2: answer_mismatch
    {
        "role": "user",
        "content": """\
Evaluate this dialogue for hallucinations.

System prompt: You are an expert in composing functions.

Conversation:
[user]: Which teams made it to the Final Four in 2023?
[assistant]: [Final Four(year="2023")]
[tool]: {"name": "Final Four", "results": {"year": 2022, "teams": ["UConn", "Gonzaga", "San Diego State", "Florida Atlantic"]}}
[assistant]: In 2023, the teams that made it to the Final Four were UConn, Miami, San Diego State, and Florida Atlantic.

final_answer: "In 2023, the teams that made it to the Final Four were UConn, Miami, San Diego State, and Florida Atlantic."
""",
    },
    {
        "role": "assistant",
        "content": """\
{"type": "answer_mismatch", "span_labels": [{"start": 3, "end": 7, "text": "2023"}, {"start": 62, "end": 67, "text": "Miami"}], "reasoning": "Tool returned year=2022 and team Gonzaga, but the answer states 2023 and substitutes Miami for Gonzaga — direct contradictions of the tool output."}""",
    },
    # Example 3: missing_tool
    {
        "role": "user",
        "content": """\
Evaluate this dialogue for hallucinations.

System prompt: You are an expert in composing functions.

Conversation:
[user]: Check token balances for my wallet 0x1234 on Ethereum, BSC, and Polygon.
[assistant]: [Get Token Balance(chainId=1), Get Token Balance(chainId=56), Get Token Balance(chainId=137)]
[tool]: [{"name": "Get Token Balance", "results": {"chainId": 1, "balance": "2.5 ETH"}}, {"name": "Get Token Balance", "results": {"chainId": 56, "balance": "13.7 BNB"}}, {"name": "Get Token Balance", "results": {"chainId": 137, "balance": "150 MATIC"}}]
[assistant]: Your balances: Ethereum: 2.5 ETH, BSC: 13.7 BNB, Polygon: 150 MATIC. I can generate a portfolio performance report if you'd like.

final_answer: "Your balances: Ethereum: 2.5 ETH, BSC: 13.7 BNB, Polygon: 150 MATIC. I can generate a portfolio performance report if you'd like."
""",
    },
    {
        "role": "assistant",
        "content": """\
{"type": "missing_tool", "span_labels": [{"start": 64, "end": 123, "text": "I can generate a portfolio performance report if you'd like."}], "reasoning": "The balance values are grounded in the tool output. The final sentence is an unsolicited proactive offer not requested by the user and not returned by any tool."}""",
    },
    # Example 4: overgeneration
    {
        "role": "user",
        "content": """\
Evaluate this dialogue for hallucinations.

System prompt: You are an expert in composing functions.

Conversation:
[user]: Calculate the trajectory of a projectile: velocity=30 m/s, angle=45°, gravity=9.81, no air resistance.
[assistant]: [simulateProjectileMotion(initialVelocity=30.0, angle=45.0, environment={"gravity": 9.81, "airResistance": 0.0})]
[tool]: {"name": "simulateProjectileMotion", "results": {"maximumHeight": 22.94, "totalFlightTime": 4.33, "horizontalDistance": 91.84}}
[assistant]: The projectile will reach a maximum height of 22.94 meters, have a total flight time of 4.33 seconds, and cover a horizontal distance of 91.84 meters. The projectile's trajectory can be precisely calculated under these conditions, and similar analyses can be extended to include wind resistance if required.

final_answer: "The projectile will reach a maximum height of 22.94 meters, have a total flight time of 4.33 seconds, and cover a horizontal distance of 91.84 meters. The projectile's trajectory can be precisely calculated under these conditions, and similar analyses can be extended to include wind resistance if required."
""",
    },
    {
        "role": "assistant",
        "content": """\
{"type": "overgeneration", "span_labels": [{"start": 152, "end": 308, "text": "The projectile's trajectory can be precisely calculated under these conditions, and similar analyses can be extended to include wind resistance if required."}], "reasoning": "The numeric results are faithfully reported. The second sentence adds unsupported editorial commentary about extendability to wind resistance — overgeneration beyond the tool output."}""",
    },
    #    Example 5: undergeneration
    {
        "role": "user",
        "content": """\
Evaluate this dialogue for hallucinations.

System prompt: You are an expert in composing functions.

Conversation:
[user]: How many live football events are happening this week?
[assistant]: [Sports Number live events(sport="football", string_range="this week")]
[tool]: {"name": "Sports", "results": "There are 15 live football events happening this week."}
[assistant]: There are live football events happening this week.

final_answer: "There are live football events happening this week."
""",
    },
    {
        "role": "assistant",
        "content": """\
{"type": "undergeneration", "span_labels": [], "reasoning": "The tool returned '15 live football events' but the answer drops the specific count '15'. This is undergeneration — information was omitted, not fabricated. Span labels are empty by design."}""",
    },
]

# ---------------------------------------------------------------------------
# Few-shot system prompt (same as zero-shot, few-shots go in messages)
# ---------------------------------------------------------------------------

FEW_SHOT_SYSTEM = ZERO_SHOT_SYSTEM  # identical system; examples added to messages

# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def build_user_message(row: dict) -> str:
    """Format a dataset row as the user message for the judge."""
    conversations = row.get("conversations", [])
    system_prompt = row.get("system", "")

    # Find final assistant answer
    final_answer = ""
    for msg in reversed(conversations):
        if msg.get("from") == "assistant":
            final_answer = msg.get("value", "")
            break

    # Format conversation turns
    conv_lines = []
    for msg in conversations:
        role = msg.get("from", "unknown")
        value = str(msg.get("value", ""))
        conv_lines.append(f"[{role}]: {value}")
    conv_str = "\n".join(conv_lines)

    return (
        f"Evaluate this dialogue for hallucinations.\n\n"
        f"System prompt: {system_prompt}\n\n"
        f"Conversation:\n{conv_str}\n\n"
        f'final_answer: {json.dumps(final_answer, ensure_ascii=False)}\n'
    )


import json  # noqa: E402 (used in build_user_message above)
