"""
export.py — Convert generated hallucination data to RAGTruth / JSON format.

Defines ExportMixin.  HallucinationAuto inherits from it.

Usage:
    ha = HallucinationAuto()
    ha.to_dataset("type1_output.json", output_format="ragtruth")   # → CSV
    ha.to_dataset("type1_output.json", output_format="json")        # → JSON

RAGTruth schema fields:
    id, query, context, output, task_type, quality, model, temperature,
    hallucination_labels, hallucination_labels_processed, input_str

The key challenge is finding the hallucinated text in the output string
(offsets vary due to number formatting, case differences, etc.).
_find_in_output tries 7 progressively fuzzier matching strategies.
"""

import json
import re
from pathlib import Path
from typing import Literal, Optional


class ExportMixin:

    _RAGTRUTH_FIELDS = [
        "id", "query", "context", "output", "task_type", "quality",
        "model", "temperature", "hallucination_labels",
        "hallucination_labels_processed", "input_str",
    ]

    # Matches [tag] and [/tag] annotation markers
    _TAG_RE = re.compile(r"\[/?[A-Za-z_][A-Za-z0-9_]*\]")

    # ── Text helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _strip_tags(text: str) -> str:
        """Remove [tag]/[/tag] annotation markers from text."""
        return ExportMixin._TAG_RE.sub("", text)

    @staticmethod
    def _extract_hall_spans(text: str) -> tuple:
        """Strip <hall>...</hall> tags and return (clean_text, spans).

        spans = list of (start, end, content) where start/end are char offsets
        into the cleaned text and content is the hallucinated substring.
        """
        spans, clean, pos = [], "", 0
        for m in re.finditer(r'<hall>(.*?)</hall>', text, re.DOTALL):
            clean += text[pos:m.start()]
            s = len(clean)
            content = m.group(1)
            clean += content
            spans.append((s, len(clean), content))
            pos = m.end()
        clean += text[pos:]
        return clean, spans

    @staticmethod
    def _find_in_output(text: str, clean_output: str) -> tuple:
        """Locate *text* inside *clean_output* using 7 fallback strategies.

        Returns (start, end, matched_text).
        Returns (-1, -1, text) if nothing matched.

        Strategies tried in order:
          1. Exact substring match
          2. Integer with thousands comma (300000 → 300,000)
          3. Float with thousands comma (12345.67 → 12,345.67)
          4. Case-insensitive match
          5. ISO date prefix (2025-01-15T10:00:00Z → 2025-01-15)
          6. Short alphanumeric word-boundary regex
          7. Short digit word-boundary regex
        """
        if not text:
            return -1, -1, text

        # 1. Exact
        idx = clean_output.find(text)
        if idx != -1:
            return idx, idx + len(text), text

        # 2. Integer with commas
        try:
            as_int = int(text)
            formatted = f"{as_int:,}"
            idx = clean_output.find(formatted)
            if idx != -1:
                return idx, idx + len(formatted), formatted
            for prefix in ("$", "€", "£", "¥", "₹"):
                pf = prefix + formatted
                idx = clean_output.find(pf)
                if idx != -1:
                    return idx, idx + len(pf), pf
        except (ValueError, OverflowError):
            pass

        # 3. Float with commas in integer part
        if "." in text:
            try:
                float(text)
                int_part, dec_part = text.split(".", 1)
                formatted = f"{int(int_part):,}.{dec_part}"
                idx = clean_output.find(formatted)
                if idx != -1:
                    return idx, idx + len(formatted), formatted
            except (ValueError, OverflowError):
                pass

        # 4. Case-insensitive
        idx = clean_output.lower().find(text.lower())
        if idx != -1:
            return idx, idx + len(text), clean_output[idx:idx + len(text)]

        # 5. ISO date prefix
        if re.match(r'^\d{4}-\d{2}-\d{2}[T ]', text):
            date_part = text[:10]
            idx = clean_output.find(date_part)
            if idx != -1:
                return idx, idx + len(date_part), date_part

        # 6. Short alphanumeric word-boundary regex
        if 3 <= len(text) <= 40 and text.isalnum():
            m = re.search(re.escape(text), clean_output, re.IGNORECASE)
            if m:
                return m.start(), m.end(), m.group()

        # 7. Short digit word-boundary regex
        if text.isdigit() and len(text) <= 6:
            m = re.search(r'(?<!\d)' + re.escape(text) + r'(?!\d)', clean_output)
            if m:
                return m.start(), m.end(), m.group()

        return -1, -1, text

    # ── Label builders ────────────────────────────────────────────────────────

    @staticmethod
    def _ragtruth_label(text: str, clean_output: str, label_type: str, meta: str,
                        implicit_true: bool = False, due_to_null: bool = False) -> dict:
        """Build a single RAGTruth-compatible label dict with start/end char offsets."""
        start, end, matched_text = ExportMixin._find_in_output(text, clean_output)
        return {
            "start": start,
            "end": end,
            "text": matched_text,
            "meta": meta,
            "label_type": label_type,
            "implicit_true": implicit_true,
            "due_to_null": due_to_null,
        }

    def _ragtruth_labels_type1(self, item: dict, clean_output: str) -> tuple:
        """Extract RAGTruth labels from Type 1 eval_spans (schema corruption).

        If no explicit (start>=0) span can be found by direct text search,
        falls back to detect_type1_spans which marks the whole answer as an
        implicit hallucination (model used corrupted data without quoting it).
        Implicit cases are kept rather than dropped — they're valid signal.
        """
        labels = []
        for span in item.get("eval_spans", []):
            if span.get("implicit_true") or span.get("due_to_null"):
                continue
            meta = span.get("meta", "")
            original_val = next(
                (line[len("Original: "):] for line in meta.split("\n")
                 if line.startswith("Original: ")),
                "",
            )
            if not original_val:
                continue
            labels.append(self._ragtruth_label(
                text=original_val,
                clean_output=clean_output,
                label_type=span.get("label_type", "Evident Conflict"),
                meta=meta,
                implicit_true=span.get("implicit_true", False),
                due_to_null=span.get("due_to_null", False),
            ))

        # Fall back to detect_type1_spans (4-tier with implicit fallback) when
        # no label has a valid offset.  The fallback returns its own `plain`
        # text whose offsets the new labels are aligned to — return it so the
        # caller can use it as `output` (avoids _strip_tags vs _span_parse_tags drift).
        plain_override = None
        if not any(L["start"] >= 0 for L in labels):
            try:
                spans, plain = self.detect_type1_spans(item)
                if spans:
                    plain_override = plain
                    labels = [
                        {
                            "start": s["start"],
                            "end":   s["end"],
                            "text":  s["text"],
                            "meta":  s.get("meta", ""),
                            "label_type":    s.get("label_type", "Evident Conflict"),
                            "implicit_true": s.get("implicit_true", False),
                            "due_to_null":   s.get("due_to_null", False),
                        }
                        for s in spans
                    ]
            except Exception:
                pass
        summary = item.get("eval_summary", {"evident_conflict": 0, "baseless_info": 0})
        return labels, summary, plain_override

    def _ragtruth_labels_type2(self, item: dict, clean_output: str) -> tuple:
        """Extract RAGTruth labels from Type 2 hallucination_spans (deletion).

        If hallucination_spans is empty or no offsets land in clean_output,
        falls back to detect_type2_spans (4-tier with implicit fallback) so
        rows where the answer over-generates without quoting deleted data are
        still kept.
        """
        labels = []
        evident_conflict = baseless_info = 0
        for span in item.get("hallucination_spans", []):
            text = span.get("original_value") or self._strip_tags(span.get("span_text", ""))
            label_type = span.get("label_type", "")
            labels.append(self._ragtruth_label(
                text=text,
                clean_output=clean_output,
                label_type=label_type,
                meta=f"{label_type.upper().replace(' ', '_')}\nSpan: {text}",
            ))
            if label_type == "Evident Conflict":
                evident_conflict += 1
            elif label_type == "Baseless Info":
                baseless_info += 1

        # Fall back to detect_type2_spans when no valid offset was found
        plain_override = None
        if not any(L["start"] >= 0 for L in labels):
            try:
                spans, plain = self.detect_type2_spans(item)
                if spans:
                    plain_override = plain
                    labels = [
                        {
                            "start": s["start"],
                            "end":   s["end"],
                            "text":  s["text"],
                            "meta":  s.get("meta", ""),
                            "label_type":    s.get("label_type", "Evident Baseless Info"),
                            "implicit_true": s.get("implicit_true", False),
                            "due_to_null":   s.get("due_to_null", True),
                        }
                        for s in spans
                    ]
                    baseless_info = sum(1 for s in spans
                                        if s.get("label_type") == "Evident Baseless Info")
            except Exception:
                pass
        return (labels,
                {"evident_conflict": evident_conflict, "baseless_info": baseless_info},
                plain_override)

    def _ragtruth_labels_type3(self, item: dict, clean_output: str) -> tuple:
        """Extract RAGTruth labels from Type 3 (tool overgeneration).

        Returns (labels, summary, plain_override). plain_override is always None
        for type3 — overgenerated_answer is already the canonical output text.
        """
        comment = item.get("overgeneration_comment", "")
        labels = []
        if comment:
            labels.append(self._ragtruth_label(
                text=comment,
                clean_output=clean_output,
                label_type="Overgeneration",
                meta=f"OVERGENERATION\n{comment}",
            ))
        return labels, {
            "evident_conflict": 0,
            "baseless_info": 0,
            "overgeneration": 1 if comment else 0,
        }, None

    # ── Row converter ─────────────────────────────────────────────────────────

    def _multistep_to_ragtruth_row(self, item: dict, row_id: int) -> dict:
        """Convert a multistep dialogue (with <hall> tags) into a RAGTruth row.

        Multistep input format (from generate_multistep_type* / pruned output):
          - 'system'        — tool list / instructions
          - 'conversations' — list of {from, value} turns; the last assistant
                              answer carries <hall>...</hall> spans
          - 'hallucination_type' — "type1" / "type2" / "type3"

        For RAGTruth:
          - query    = last user prompt before the hallucinated answer
          - context  = last tool turn value (corrupted for type1/2, original for type3)
          - output   = last assistant answer with <hall> tags stripped
          - labels   = char offsets of stripped <hall> spans
          - input_str = full dialogue history up to (but not including) the answer
        """
        convs = item.get("conversations", [])
        h_type = item.get("hallucination_type", "")

        # Find last assistant turn (with <hall> tags if present)
        last_ass_idx = None
        for i in range(len(convs) - 1, -1, -1):
            if convs[i].get("from") == "assistant" and "<hall>" in convs[i].get("value", ""):
                last_ass_idx = i
                break
        if last_ass_idx is None:  # fallback: last assistant of any kind
            for i in range(len(convs) - 1, -1, -1):
                if convs[i].get("from") == "assistant":
                    last_ass_idx = i
                    break
        if last_ass_idx is None:
            return None

        # Find tool turn before the assistant answer
        last_tool_idx = next(
            (i for i in range(last_ass_idx - 1, -1, -1) if convs[i].get("from") == "tool"),
            None,
        )
        # Find user turn before the tool turn
        last_user_idx = None
        if last_tool_idx is not None:
            last_user_idx = next(
                (i for i in range(last_tool_idx - 1, -1, -1) if convs[i].get("from") == "user"),
                None,
            )

        # Strip <hall> tags and capture span offsets in the cleaned output
        raw_answer = convs[last_ass_idx].get("value", "")
        clean_output, hall_spans = self._extract_hall_spans(raw_answer)

        # Build RAGTruth labels
        label_type_map = {
            "type1": "Evident Conflict",
            "type2": "Evident Baseless Info",
            "type3": "Overgeneration",
        }
        label_type = label_type_map.get(h_type, "Evident Conflict")
        labels = [
            {
                "start": s,
                "end": e,
                "text": txt,
                "meta": f"{label_type.upper()}\nMultistep dialogue ({h_type})",
                "label_type": label_type,
                "implicit_true": False,
                "due_to_null": h_type == "type2",
            }
            for s, e, txt in hall_spans
        ]
        summary = {"evident_conflict": 0, "baseless_info": 0}
        if h_type == "type1":
            summary["evident_conflict"] = len(labels)
        elif h_type == "type2":
            summary["baseless_info"] = len(labels)
        elif h_type == "type3":
            summary["overgeneration"] = len(labels)

        query = convs[last_user_idx].get("value", "") if last_user_idx is not None else ""
        context = convs[last_tool_idx].get("value", "") if last_tool_idx is not None else ""

        # input_str: full conversation history up to (but not including) the hallucinated answer
        parts = []
        if item.get("system"):
            parts.append(f"System:\n{item['system']}")
        for i in range(last_ass_idx):
            c = convs[i]
            role = c.get("from", "?").capitalize()
            parts.append(f"{role}:\n{c.get('value', '')}")

        return {
            "id": row_id,
            "query": query,
            "context": context,
            "output": clean_output,
            "task_type": f"multistep_{h_type}",
            "quality": "good",
            "model": "",
            "temperature": "",
            "hallucination_labels": json.dumps(labels),
            "hallucination_labels_processed": json.dumps(summary),
            "input_str": "\n\n".join(parts),
        }

    def _item_to_ragtruth_row(self, item: dict, row_id: int) -> dict:
        """Convert one generated item into a RAGTruth-compatible row dict.

        Auto-detects format:
          - multistep dialogue (has 'conversations' field) → uses _multistep_to_ragtruth_row
          - singlehop row (flat) → uses the type-specific label extractors
        """
        # Multistep dispatch
        if "conversations" in item:
            return self._multistep_to_ragtruth_row(item, row_id)

        h_type = item.get("hallucination_type", "")
        is_overgen = h_type in ("type2_overgen", "type3_tool_overgen")
        is_type1 = h_type.startswith("type1_")

        if is_overgen:
            clean_output = item.get("overgenerated_answer", item.get("original_answer", ""))
        else:
            clean_output = self._strip_tags(item.get("tagged_answer", item.get("original_answer", "")))

        if is_type1:
            context = item.get("hallucinated_tool_response", item.get("tool_response", ""))
            labels, summary, plain_override = self._ragtruth_labels_type1(item, clean_output)
        elif h_type == "type2_deletion":
            # Use reduced_tool_response (after deletion) so the answer's
            # references to deleted fields are unsupported by the context
            context = item.get("reduced_tool_response", item.get("tool_response", ""))
            labels, summary, plain_override = self._ragtruth_labels_type2(item, clean_output)
        elif is_overgen:
            context = item.get("tool_response", "")
            labels, summary, plain_override = self._ragtruth_labels_type3(item, clean_output)
        else:
            context = item.get("tool_response", "")
            labels, summary, plain_override = [], {"evident_conflict": 0, "baseless_info": 0}, None

        # If the fallback used a different plain text (e.g. _span_parse_tags
        # vs _strip_tags differs because of literal [BRACKET] strings in the
        # answer), use that text so label offsets remain valid.
        if plain_override is not None:
            clean_output = plain_override

        user_prompt = item.get("user_prompt", "")
        parts = []
        if item.get("system"):
            parts.append(f"System:\n{item['system']}")
        parts.append(f"User:\n{user_prompt}")
        parts.append(f"Tool Response:\n{item.get('tool_response', '')}")

        return {
            "id": row_id,
            "query": user_prompt,
            "context": context,
            "output": clean_output,
            "task_type": h_type,
            "quality": "good",
            "model": "",
            "temperature": "",
            "hallucination_labels": json.dumps(labels),
            "hallucination_labels_processed": json.dumps(summary),
            "input_str": "\n\n".join(parts),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def to_dataset(self, input_data,
                   output_format: Literal["jsonl", "ragtruth", "json"] = "jsonl",
                   output_path: Optional[str] = None):
        """Convert generated hallucination data to a structured dataset file.

        All three formats use the SAME RAGTruth-style row schema:
          id, query, context, output, task_type, quality, model, temperature,
          hallucination_labels, hallucination_labels_processed, input_str

        input_data:
          Path to a JSON file from generate_type*_dataset, OR a list of dicts.

        output_format:
          "jsonl"     → JSON Lines (one row per line) — DEFAULT
          "ragtruth"  → CSV matching the wandb/RAGTruth-processed schema
          "json"      → JSON array of row dicts (saved or returned in-memory)

        output_path:
          Override the default output path.
          For "jsonl":    defaults to <input_stem>.jsonl
          For "ragtruth": defaults to <input_stem>_ragtruth.csv
          For "json":     if None, returns the list in memory

        Returns:
          str (file path) or list (for in-memory "json" mode)
        """
        import csv as _csv

        if isinstance(input_data, (str, Path)):
            input_path = Path(input_data)
            with open(input_path, encoding="utf-8") as f:
                items = json.load(f)
        else:
            items, input_path = input_data, None

        rows = [self._item_to_ragtruth_row(item, i + 1) for i, item in enumerate(items)]

        # ── JSONL (default) ──────────────────────────────────────────────────
        if output_format == "jsonl":
            if output_path is not None:
                out = Path(output_path)
            elif input_path is not None:
                out = input_path.with_suffix(".jsonl")
            else:
                raise ValueError("output_path is required when input_data is a list")
            with open(out, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return str(out)

        # ── JSON array ───────────────────────────────────────────────────────
        if output_format == "json":
            if output_path is None:
                return rows
            out = Path(output_path)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            return str(out)

        # ── RAGTruth CSV ─────────────────────────────────────────────────────
        if output_path is not None:
            out = Path(output_path)
        elif input_path is not None:
            out = input_path.with_name(input_path.stem + "_ragtruth.csv")
        else:
            raise ValueError("output_path is required when input_data is a list")

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=self._RAGTRUTH_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        return str(out)
