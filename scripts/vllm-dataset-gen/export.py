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
        """Extract RAGTruth labels from Type 1 eval_spans (schema corruption)."""
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
        summary = item.get("eval_summary", {"evident_conflict": 0, "baseless_info": 0})
        return labels, summary

    def _ragtruth_labels_type2(self, item: dict, clean_output: str) -> tuple:
        """Extract RAGTruth labels from Type 2 hallucination_spans (deletion)."""
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
        return labels, {"evident_conflict": evident_conflict, "baseless_info": baseless_info}

    def _ragtruth_labels_type3(self, item: dict, clean_output: str) -> tuple:
        """Extract RAGTruth labels from Type 3 (tool overgeneration)."""
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
        }

    # ── Row converter ─────────────────────────────────────────────────────────

    def _item_to_ragtruth_row(self, item: dict, row_id: int) -> dict:
        """Convert one generated item into a RAGTruth-compatible row dict.

        Detects hallucination_type automatically and routes to the right
        label extractor.
        """
        h_type = item.get("hallucination_type", "")
        is_overgen = h_type in ("type2_overgen", "type3_tool_overgen")
        is_type1 = h_type.startswith("type1_")

        if is_overgen:
            clean_output = item.get("overgenerated_answer", item.get("original_answer", ""))
        else:
            clean_output = self._strip_tags(item.get("tagged_answer", item.get("original_answer", "")))

        if is_type1:
            context = item.get("hallucinated_tool_response", item.get("tool_response", ""))
            labels, summary = self._ragtruth_labels_type1(item, clean_output)
        elif h_type == "type2_deletion":
            context = item.get("tool_response", "")
            labels, summary = self._ragtruth_labels_type2(item, clean_output)
        elif is_overgen:
            context = item.get("tool_response", "")
            labels, summary = self._ragtruth_labels_type3(item, clean_output)
        else:
            context = item.get("tool_response", "")
            labels, summary = [], {"evident_conflict": 0, "baseless_info": 0}

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

    def to_dataset(self, input_data, output_format: Literal["json", "ragtruth"] = "ragtruth",
                   output_path: Optional[str] = None):
        """Convert generated hallucination data to a structured dataset file.

        input_data:
          Path to a JSON file from generate_type*_dataset, OR a list of dicts.

        output_format:
          "ragtruth" → CSV matching the wandb/RAGTruth-processed schema
          "json"     → list of row dicts (saved or returned in-memory)

        output_path:
          Override the default output path.
          For "ragtruth": defaults to <input_stem>_ragtruth.csv
          For "json": if None, returns the list in memory

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

        if output_format == "json":
            if output_path is None:
                return rows
            out = Path(output_path)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            return str(out)

        # ragtruth → CSV
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
