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

    @staticmethod
    def _span_leaves(obj, prefix=""):
        """Yield (path, str_value) for every leaf in a nested object."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from ExportMixin._span_leaves(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from ExportMixin._span_leaves(v, f"{prefix}.{i}")
        else:
            yield prefix, str(obj)

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

        Post-processing applied to each label:
        - Expand partial-word spans to the full word
        - Drop FPs: value unchanged in corrupted context (leaf-value exact match)
        - Add all other case-insensitive occurrences of the same value

        Falls back to detect_type1_spans when no valid label remains.
        """
        out_lower = clean_output.lower()

        # Build corrupted-context leaf set for FP filtering
        corrupted_ctx = item.get("hallucinated_tool_response", "")
        ctx_lower = corrupted_ctx.lower()   # full text for window checks
        hall_leaves: set = set()
        if corrupted_ctx:
            try:
                hall_leaves = {v.lower() for _, v in
                               self._span_leaves(json.loads(corrupted_ctx))
                               if len(v) > 2}
            except Exception:
                pass

        def _full_word(s, e):
            # Expand alphabetic boundaries
            while s > 0 and clean_output[s - 1].isalpha():
                s -= 1
            while e < len(clean_output) and clean_output[e].isalpha():
                e += 1
            # For short numeric spans, also absorb trailing % or adjacent digits
            if e - s <= 3:
                while e < len(clean_output) and clean_output[e] in '%.,':
                    e += 1
                while s > 0 and clean_output[s - 1].isdigit():
                    s -= 1
            return s, e, clean_output[s:e]

        seen: list = []   # (start, end) already added
        labels = []
        evident_conflict = baseless_info = 0

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

            # Extract the generated (corrupted) value from meta
            generated_val = next(
                (line[len("Generated: "):] for line in meta.split("\n")
                 if line.startswith("Generated: ")),
                None,
            )

            # FP filter: only drop if the field wasn't actually changed
            # (generated == original means only casing changed, no real conflict).
            # If generated differs, the field was changed → real conflict → keep
            # even if original value appears elsewhere in the corrupted context.
            if generated_val is not None:
                if generated_val.lower() == original_val.lower():
                    continue  # no semantic change
            else:
                # Fallback: use hall_leaves check when meta has no Generated line
                if original_val.lower() in hall_leaves:
                    continue

            L = self._ragtruth_label(
                text=original_val,
                clean_output=clean_output,
                label_type=span.get("label_type", "Evident Conflict"),
                meta=meta,
            )
            if L["start"] < 0:
                continue

            # For short values, reject matches embedded inside a longer token
            # (e.g. "2" matched inside "ORD1234567" instead of "- 2 units").
            if len(original_val) <= 3:
                s0, e0 = L["start"], L["end"]
                before = clean_output[s0 - 1] if s0 > 0 else " "
                after  = clean_output[e0]     if e0 < len(clean_output) else " "
                if before.isalnum() or after.isalnum():
                    continue

            label_type = span.get("label_type", "Evident Conflict")
            val_in_unchanged = original_val.lower() in hall_leaves

            if val_in_unchanged:
                # Value also appears in an unchanged field.  Find the first
                # occurrence whose surrounding window does NOT appear in the
                # corrupted context — that's the hallucinated one.
                val_re = re.escape(original_val.lower())
                chosen_idx = -1
                for m in re.finditer(val_re, out_lower):
                    window = clean_output[max(0, m.start()-30):
                                          m.end()+30].lower()
                    if window not in ctx_lower:
                        chosen_idx = m.start()
                        break
                if chosen_idx < 0:
                    # Fallback: last occurrence
                    chosen_idx = out_lower.rfind(original_val.lower())
                if chosen_idx < 0:
                    continue
                fs, fe, ftxt = _full_word(chosen_idx,
                                          chosen_idx + len(original_val))
                actual = clean_output[fs:fe]
                if not any(s < fe and fs < e for s, e in seen):
                    labels.append({**L, "start": fs, "end": fe, "text": actual})
                    seen.append((fs, fe))
                    evident_conflict += 1
            else:
                # Value only in changed fields: expand and mark all occurrences.
                fs, fe, ftxt = _full_word(L["start"], L["end"])

                # For long prose originals (tool result is a whole sentence) that
                # cover most of the output, try to mark only the specific output
                # numbers/values that came from the tool (not from the query itself).
                if len(original_val) > 50 and (fe - fs) > len(clean_output) * 0.6:
                    query_lower = item.get("user_prompt", "").lower()
                    specific = []
                    for m in re.finditer(r'\d+\.?\d+|\d+', original_val):
                        tok = m.group()
                        if tok not in query_lower:
                            idx = out_lower.find(tok)
                            if idx >= 0:
                                ns, ne = _full_word(idx, idx + len(tok))[:2]
                                actual = clean_output[ns:ne]
                                if not any(s < ne and ns < e for s, e in seen):
                                    specific.append((ns, ne, actual))
                                    seen.append((ns, ne))
                    if specific:
                        for ns, ne, actual in specific:
                            labels.append({**L, "start": ns, "end": ne, "text": actual})
                            evident_conflict += 1
                        continue  # skip the full-string label

                if not any(s < fe and fs < e for s, e in seen):
                    labels.append({**L, "start": fs, "end": fe, "text": ftxt})
                    seen.append((fs, fe))
                    evident_conflict += 1
                if len(ftxt) <= 2:
                    continue
                ftxt_lower = ftxt.lower()
                start = 0
                while True:
                    idx = out_lower.find(ftxt_lower, start)
                    if idx < 0:
                        break
                    ne = idx + len(ftxt)
                    actual = clean_output[idx:ne]
                    if not any(s < ne and idx < e for s, e in seen):
                        labels.append({
                            "start": idx, "end": ne, "text": actual,
                            "meta": meta, "label_type": label_type,
                            "implicit_true": False, "due_to_null": False,
                        })
                        seen.append((idx, ne))
                        evident_conflict += 1
                    start = idx + 1

        # Fall back to detect_type1_spans when nothing found.
        # Use span positions directly — they are already in `plain` coordinates.
        # Do NOT pass through _full_word (which closes over clean_output, a
        # different text) as that would corrupt the offsets.
        plain_override = None
        if not labels:
            try:
                spans, plain = self.detect_type1_spans(item)
                if spans:
                    plain_override = plain
                    for s in spans:
                        labels.append({
                            "start": s["start"], "end": s["end"],
                            "text":  plain[s["start"]:s["end"]],
                            "meta":  s.get("meta", ""),
                            "label_type":    s.get("label_type", "Evident Conflict"),
                            "implicit_true": s.get("implicit_true", False),
                            "due_to_null":   s.get("due_to_null", False),
                        })
                    evident_conflict = sum(1 for s in spans
                                          if not s.get("implicit_true"))
            except Exception:
                pass

        labels.sort(key=lambda L: L["start"])
        summary = {"evident_conflict": evident_conflict, "baseless_info": baseless_info}
        return labels, summary, plain_override

    def _ragtruth_labels_type2(self, item: dict, clean_output: str) -> tuple:
        """Extract RAGTruth labels from Type 2 hallucination_spans (deletion).

        After finding the primary span for each deleted field, also searches for
        all other occurrences of the same value in the output (same logic as
        the multistep type2 extension).  Skips values that appear in the
        remaining (reduced) context — those occurrences are grounded, not
        hallucinated.

        Falls back to detect_type2_spans when no primary span is found.
        """
        labels = []
        evident_conflict = baseless_info = 0
        out_lower    = clean_output.lower()
        reduced_ctx  = item.get("reduced_tool_response", "")
        ctx_lower    = reduced_ctx.lower() if reduced_ctx else ""

        raw_labels = []
        for span in item.get("hallucination_spans", []):
            text = span.get("original_value") or self._strip_tags(span.get("span_text", ""))
            label_type = span.get("label_type", "")
            L = self._ragtruth_label(
                text=text,
                clean_output=clean_output,
                label_type=label_type,
                meta=f"{label_type.upper().replace(' ', '_')}\nSpan: {text}",
            )
            raw_labels.append((L, text, label_type))
            if label_type == "Evident Conflict":
                evident_conflict += 1
            elif label_type == "Baseless Info":
                baseless_info += 1

        # Fall back to detect_type2_spans when no valid offset was found
        plain_override = None
        if not any(L["start"] >= 0 for L, _, _ in raw_labels):
            try:
                spans, plain = self.detect_type2_spans(item)
                if spans:
                    plain_override = plain
                    labels = [
                        {
                            "start": s["start"], "end": s["end"], "text": s["text"],
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

        def _full_word_t2(s, e):
            while s > 0 and clean_output[s - 1].isalpha():
                s -= 1
            while e < len(clean_output) and clean_output[e].isalpha():
                e += 1
            return s, e, clean_output[s:e]

        # Leaf values of reduced context — used for FP filtering.
        # Use exact leaf matching (not substring) so values that appear only as
        # part of a longer description are not incorrectly filtered.
        try:
            ctx_leaf_values = {v.lower() for _, v in
                               self._span_leaves(json.loads(reduced_ctx))
                               if len(v) > 2} if reduced_ctx else set()
        except Exception:
            ctx_leaf_values = set()

        # Build label list and extend with all occurrences of each value
        seen: list = []  # (start, end) of already-added spans
        path_to_label: dict = {}  # span path → index in labels list
        for (L, text, label_type), span in zip(raw_labels, item.get("hallucination_spans", [])):
            if L["start"] < 0:
                continue
            # Skip if value is a standalone leaf in the remaining context
            if len(text) > 2 and text.lower() in ctx_leaf_values:
                continue
            # Expand partial-word spans to full word
            fs, fe, ftxt = _full_word_t2(L["start"], L["end"])
            L = {**L, "start": fs, "end": fe, "text": ftxt}
            if not any(s < fe and fs < e for s, e in seen):
                path_to_label[span.get("path", "")] = len(labels)
                labels.append(L)
                seen.append((fs, fe))
            # Find all other occurrences using the full-word expanded text
            txt_lower = ftxt.lower()
            if len(ftxt) <= 2:
                continue
            start = 0
            while True:
                idx = out_lower.find(txt_lower, start)
                if idx < 0:
                    break
                ne = idx + len(ftxt)
                actual = clean_output[idx:ne]
                if not any(s < ne and idx < e for s, e in seen):
                    labels.append({
                        "start": idx, "end": ne, "text": actual,
                        "meta":  f"{label_type.upper().replace(' ', '_')}\nSpan: {ftxt}",
                        "label_type": label_type,
                        "implicit_true": False, "due_to_null": True,
                    })
                    seen.append((idx, ne))
                    baseless_info += 1
                start = idx + 1

        # Also scan deleted_paths for values missed by hallucination_spans
        # (e.g. numeric fields like profit:45000 that appear as $45,000 in output
        # and weren't tagged by _tag_row, so no hallucination_span was created).
        orig_tr = item.get("tool_response", "")
        del_paths = item.get("deleted_paths", [])
        tagged_paths = {sp.get("path", "") for sp in item.get("hallucination_spans", [])}
        if orig_tr and del_paths:
            try:
                orig_data = json.loads(orig_tr)
                for path in del_paths:
                    if path in tagged_paths:
                        continue
                    try:
                        # Resolve dot-notation path into nested dict/list
                        obj = orig_data
                        for part in re.split(r'[\.\[\]]', path):
                            if not part:
                                continue
                            obj = obj[int(part)] if part.isdigit() else obj[part]
                        val = str(obj)
                    except Exception:
                        continue
                    if len(val) <= 1 or val.lower() in ctx_lower:
                        continue
                    # Try formatted numeric variants — currency-prefixed first
                    # so "$45,000" is preferred over bare "45,000"
                    candidates = []
                    try:
                        n = int(val)
                        fmt = f"{n:,}"
                        for pfx in ("$", "€", "£", "¥", "₹"):
                            candidates.append(pfx + fmt)
                        if fmt != val:
                            candidates.append(fmt)
                    except (ValueError, OverflowError):
                        pass
                    if "." in val:
                        try:
                            ip, dp = val.split(".", 1)
                            fmt = f"{int(ip):,}.{dp}"
                            for pfx in ("$", "€", "£", "¥", "₹"):
                                candidates.append(pfx + fmt)
                            if fmt != val:
                                candidates.append(fmt)
                        except (ValueError, OverflowError):
                            pass
                    candidates.append(val)  # raw fallback
                    found_this = False
                    for candidate in candidates:
                        idx = out_lower.find(candidate.lower())
                        if idx < 0:
                            continue
                        ne = idx + len(candidate)
                        actual = clean_output[idx:ne]
                        if not any(s < ne and idx < e for s, e in seen):
                            labels.append({
                                "start": idx, "end": ne, "text": actual,
                                "meta":  f"BASELESS_INFO\nPath (deleted): {path}",
                                "label_type": "Baseless Info",
                                "implicit_true": False, "due_to_null": True,
                            })
                            seen.append((idx, ne))
                            baseless_info += 1
                        found_this = True
                        break  # first match per path

                    if not found_this and val:
                        # Value not found verbatim (e.g. paraphrased summary).
                        # If a sibling path from the same parent item WAS labeled,
                        # extend that label to the end of its paragraph block so
                        # the paraphrased content is also covered.
                        parent = re.sub(r'\.[^.]+$', '', path)  # e.g. results.0
                        sibling_idx = next(
                            (idx for p, idx in path_to_label.items()
                             if p.startswith(parent + ".") or p == parent),
                            None,
                        )
                        if sibling_idx is not None:
                            lbl = labels[sibling_idx]
                            block_end = clean_output.find('\n\n', lbl["end"])
                            if block_end < 0:
                                block_end = len(clean_output)
                            if block_end > lbl["end"]:
                                lbl["end"]  = block_end
                                lbl["text"] = clean_output[lbl["start"]:block_end]
                                seen = [(s, max(e, block_end) if s == lbl["start"] else e)
                                        for s, e in seen]
            except Exception:
                pass

        # Detect fabricated structural blocks: the model often invents whole
        # extra list items / sections that don't correspond to any value in the
        # ORIGINAL tool response.  Split the output into blocks delimited by
        # blank lines or by markdown list markers and flag any block that has
        # zero overlap with the original tool response leaf values.
        orig_tr = item.get("tool_response", "")
        if orig_tr and len(clean_output) > 50:
            try:
                orig_leaves = {v.lower() for _, v in
                               self._span_leaves(json.loads(orig_tr))
                               if len(v) > 2}
            except Exception:
                orig_leaves = set()
            if orig_leaves:
                # Split into structural blocks. We treat a block as a section
                # starting after a blank line or after a new top-level list
                # marker (numbered or bulleted).
                block_re = re.compile(r'\n\s*\n+|(?<=\n)(?=(?:[-*]\s+|\d+\.\s+)\*\*)')
                cursor = 0
                blocks = []
                for m in block_re.finditer(clean_output):
                    end = m.start()
                    if end > cursor:
                        blocks.append((cursor, end))
                    cursor = m.end()
                if cursor < len(clean_output):
                    blocks.append((cursor, len(clean_output)))

                for bs, be in blocks:
                    btext = clean_output[bs:be]
                    btext_lower = btext.lower()
                    # Require the block to look like a structural item (bullet
                    # or numbered entry) AND be reasonably long
                    if len(btext.strip()) < 30:
                        continue
                    if not re.search(r'^\s*(?:[-*]\s+|\d+\.\s+)\*\*', btext):
                        continue
                    # Count overlap with original leaf values
                    matches = sum(1 for leaf in orig_leaves
                                  if leaf in btext_lower)
                    if matches == 0:
                        # Fully fabricated block — mark whole block
                        actual_start = bs + len(btext) - len(btext.lstrip())
                        actual_end   = be - (len(btext) - len(btext.rstrip()))
                        actual_text  = clean_output[actual_start:actual_end]
                        if (actual_end > actual_start and
                            not any(s < actual_end and actual_start < e
                                    for s, e in seen)):
                            labels.append({
                                "start": actual_start, "end": actual_end,
                                "text":  actual_text,
                                "meta":  "BASELESS_INFO\nFabricated block — no value from original tool response",
                                "label_type": "Baseless Info",
                                "implicit_true": False, "due_to_null": True,
                            })
                            seen.append((actual_start, actual_end))
                            baseless_info += 1

        labels.sort(key=lambda L: L["start"])
        return (labels,
                {"evident_conflict": evident_conflict, "baseless_info": baseless_info},
                plain_override)

    def _ragtruth_labels_type3(self, item: dict, clean_output: str) -> tuple:
        """Extract RAGTruth labels from Type 3 (tool overgeneration).

        The structure is always: clean_output = original_answer + "\\n\\n" + comment
        so the span is computed from the suffix — no text search needed.

        Returns (labels, summary, plain_override). plain_override is always None
        for type3 — overgenerated_answer is already the canonical output text.
        """
        comment = item.get("overgeneration_comment", "")
        labels = []
        if comment:
            end = len(clean_output)
            start = end - len(comment)
            if start < 0 or clean_output[start:] != comment:
                # Fallback: last occurrence (handles minor whitespace drift)
                idx = clean_output.rfind(comment)
                if idx >= 0:
                    start, end = idx, idx + len(comment)
                else:
                    start, end = -1, -1
            if start >= 0:
                labels.append({
                    "start": start,
                    "end": end,
                    "text": comment,
                    "meta": f"OVERGENERATION\n{comment}",
                    "label_type": "Overgeneration",
                    "implicit_true": False,
                    "due_to_null": False,
                })
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

        # For type2: extend hall_spans with all other occurrences of each span
        # text (case-insensitive), so missed occurrences are also labelled.
        h_type_base = h_type.split("_")[0] if h_type else ""
        if h_type_base == "type2" and hall_spans:
            out_lower = clean_output.lower()
            ctx_text  = convs[last_tool_idx].get("value", "") if last_tool_idx is not None else ""
            ctx_lower = ctx_text.lower()
            hall_target = (convs[last_tool_idx].get("hallucination_target", [])
                           if last_tool_idx is not None else [])
            extended: list = list(hall_spans)

            # If the single span covers the whole answer it's an implicit fallback
            # from generation (e.g. numeric values like 50000 appeared as ¥50,000).
            is_implicit = (len(hall_spans) == 1
                           and hall_spans[0][0] == 0
                           and hall_spans[0][1] >= len(clean_output) - 2)
            if is_implicit and last_tool_idx is not None:
                def _field_keywords(targets):
                    """Split field paths into meaningful keyword tokens."""
                    kws = []
                    for t in targets:
                        parts = re.split(r'[._]|\d+', t)
                        for p in parts:
                            kws += re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', p) or [p]
                    return [k.lower() for k in kws if len(k) > 3]

                keywords = _field_keywords(hall_target)

                # If no keyword appears in the answer at all the model didn't
                # reference the deleted field — drop entirely.
                has_ref = not keywords or any(kw in out_lower for kw in keywords)
                if not has_ref:
                    extended = []
                else:
                    # 1. Numeric recovery: formatted numbers not present in context
                    ctx_nums = set(re.findall(r'[¥$€£₹]?[\d,]+(?:\.\d+)?', ctx_text))
                    ctx_nums_norm = {re.sub(r'[,¥$€£₹]', '', n) for n in ctx_nums}
                    extended = []
                    for m in re.finditer(r'[¥$€£₹]?[\d,]+(?:\.\d+)?', clean_output):
                        tok  = m.group()
                        norm = re.sub(r'[,¥$€£₹]', '', tok)
                        if norm and norm not in ctx_nums_norm and len(norm) > 2:
                            ns, ne = m.start(), m.end()
                            if not any(es < ne and ns < ee for es, ee, _ in extended):
                                extended.append((ns, ne, tok))

                    if not extended:
                        # 2. Line/sentence recovery: find the narrowest span that
                        #    contains the keyword and has content not in context.
                        def _candidate_spans(text, kw, ctx_lo):
                            """Yield (start, end) for sentences/lines with kw and unique content."""
                            for lm in re.finditer(r'(?m)^[^\n]*' + re.escape(kw) + r'[^\n]*$',
                                                  text, re.IGNORECASE):
                                chunk     = lm.group()
                                chunk_off = lm.start()
                                if len(chunk) > 120:
                                    # Sentence split: only break at ". "/! "/? " followed
                                    # by a capital — avoids splitting on "file.md" etc.
                                    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z\"\'])',
                                                     chunk)
                                    pos = 0
                                    for sent in sents:
                                        if kw.lower() in sent.lower():
                                            unique = [w for w in re.findall(
                                                r'\b[a-z]{5,}\b', sent.lower())
                                                      if w not in ctx_lo]
                                            if unique:
                                                ns = chunk_off + pos
                                                ne = chunk_off + pos + len(sent)
                                                yield ns, ne
                                                break  # first matching sentence only
                                        pos += len(sent) + 1  # +1 for the split space
                                else:
                                    unique = [w for w in re.findall(r'\b[a-z]{5,}\b',
                                                                     chunk.lower())
                                              if w not in ctx_lo]
                                    if unique:
                                        yield lm.start(), lm.end()

                        for kw in keywords:
                            # First matching span per keyword — avoids duplicate labels
                            # when the same field is referenced in summary + detail lines.
                            for ns, ne in _candidate_spans(clean_output, kw, ctx_lower):
                                raw = clean_output[ns:ne]
                                ns += len(raw) - len(raw.lstrip())
                                ne -= len(raw) - len(raw.rstrip())
                                if ns < ne and not any(es < ne and ns < ee
                                                       for es, ee, _ in extended):
                                    extended.append((ns, ne, clean_output[ns:ne]))
                                break  # one span per keyword
                    # No whole-answer fallback: if nothing found the model either
                    # didn't hallucinate or phrased it in a way we can't detect —
                    # in both cases the row is dropped (see return None below).
            else:
                # Non-implicit path: clean up original <hall> spans.
                def _in_ctx_as_word(val, ctx_lo):
                    return bool(re.search(r'\b' + re.escape(val.lower()) + r'\b',
                                          ctx_lo))

                def _full_word(text, s, e):
                    """Expand [s:e] to word boundaries if it's a partial word."""
                    while s > 0 and text[s - 1].isalpha():
                        s -= 1
                    while e < len(text) and text[e].isalpha():
                        e += 1
                    return s, e, text[s:e]

                cleaned_originals = []
                for s, e, txt in hall_spans:
                    in_ctx = _in_ctx_as_word(txt, ctx_lower)
                    is_short_word = (len(txt) <= 8
                                     and re.fullmatch(r'[a-zA-Z\s]+', txt))
                    if in_ctx and is_short_word:
                        # Common short word also in context — wrong occurrence, drop.
                        pass
                    elif in_ctx:
                        # Longer proper-noun value in context: generation tagged the
                        # first (context-grounded) occurrence; use the LAST occurrence
                        # instead, which is likely in the deleted section.
                        last_idx = out_lower.rfind(txt.lower())
                        if last_idx >= 0 and last_idx != s:
                            cleaned_originals.append(
                                (last_idx, last_idx + len(txt),
                                 clean_output[last_idx:last_idx + len(txt)]))
                        else:
                            cleaned_originals.append((s, e, txt))
                    else:
                        # Expand partial-word spans to the full word (e.g. "success"
                        # inside "successfully" → "successfully").
                        fs, fe, ftxt = _full_word(clean_output, s, e)
                        cleaned_originals.append((fs, fe, ftxt))

                extended = cleaned_originals or list(hall_spans)
                for _s, _e, txt in hall_spans:
                    # Skip very short values — too noisy (e.g. rank "2", code "IN")
                    if len(txt) <= 2:
                        continue
                    # Skip if the value also appears in the remaining context —
                    # can't distinguish deleted vs non-deleted occurrences
                    if txt.lower() in ctx_lower:
                        continue
                    txt_lower = txt.lower()
                    # Use word-boundary matching for short alphabetic values
                    # to avoid "live" matching inside "deliverable" etc.
                    if len(txt) <= 8 and re.fullmatch(r'[a-zA-Z ]+', txt):
                        for m in re.finditer(r'\b' + re.escape(txt_lower) + r'\b',
                                             out_lower):
                            ns, ne = m.start(), m.end()
                            if not any(es < ne and ns < ee for es, ee, _ in extended):
                                extended.append((ns, ne, clean_output[ns:ne]))
                    else:
                        start = 0
                        while True:
                            idx = out_lower.find(txt_lower, start)
                            if idx < 0:
                                break
                            ns, ne = idx, idx + len(txt)
                            if not any(es < ne and ns < ee for es, ee, _ in extended):
                                extended.append((ns, ne, clean_output[ns:ne]))
                            start = idx + 1

            hall_spans = sorted(extended, key=lambda t: t[0])

        # For type1: false-positive filter + partial-word expansion + all occurrences.
        # The context here is the CORRUPTED tool response.  If the original value
        # (span text) appears in the corrupted context as a whole word, the field
        # wasn't actually changed → the model didn't hallucinate → drop the span.
        elif h_type_base == "type1" and hall_spans:
            out_lower = clean_output.lower()
            ctx_text  = (convs[last_tool_idx].get("value", "")
                         if last_tool_idx is not None else "")
            ctx_lower = ctx_text.lower()

            def _fp_in_ctx(val):
                """True if val appears as a whole word in the corrupted context."""
                return bool(re.search(
                    r'\b' + re.escape(val.lower()) + r'\b', ctx_lower))

            def _full_word_t1(text, s, e):
                while s > 0 and text[s - 1].isalpha():
                    s -= 1
                while e < len(text) and text[e].isalpha():
                    e += 1
                return s, e, text[s:e]

            is_implicit_t1 = (len(hall_spans) == 1
                              and hall_spans[0][0] == 0
                              and hall_spans[0][1] >= len(clean_output) - 2)

            if is_implicit_t1:
                # Implicit whole-answer span: the generation couldn't find a
                # specific conflicting value.  Try numeric conflict recovery:
                # find formatted numbers in the answer that do NOT appear in the
                # corrupted context (those are the original values that conflict).
                ctx_nums = set(re.findall(r'[\d,]+(?:\.\d+)?', ctx_text))
                ctx_nums_norm = {re.sub(r'[,¥$€£₹]', '', n) for n in ctx_nums}
                extended = []
                for m in re.finditer(r'[¥$€£₹]?[\d,]+(?:\.\d+)?', clean_output):
                    tok  = m.group()
                    norm = re.sub(r'[,¥$€£₹]', '', tok)
                    if norm and norm not in ctx_nums_norm and len(norm) > 2:
                        ns, ne = m.start(), m.end()
                        if not any(es < ne and ns < ee for es, ee, _ in extended):
                            extended.append((ns, ne, tok))
                # No fallback to whole-answer — if nothing found the model's
                # answer was consistent with the corrupted context → FP → drop.
            else:
                extended = []
                for s, e, txt in hall_spans:
                    # Drop false positives: original value still in corrupted context
                    if len(txt) > 2 and _fp_in_ctx(txt):
                        continue
                    # Expand partial-word matches to the full word
                    fs, fe, ftxt = _full_word_t1(clean_output, s, e)
                    if not any(es < fe and fs < ee for es, ee, _ in extended):
                        extended.append((fs, fe, ftxt))
                    # Find all other occurrences in the answer (case-insensitive)
                    if len(ftxt) <= 2:
                        continue
                    start = 0
                    ftxt_lower = ftxt.lower()
                    while True:
                        idx = out_lower.find(ftxt_lower, start)
                        if idx < 0:
                            break
                        ns, ne = idx, idx + len(ftxt)
                        if not any(es < ne and ns < ee for es, ee, _ in extended):
                            extended.append((ns, ne, clean_output[ns:ne]))
                        start = idx + 1

            hall_spans = sorted(extended, key=lambda t: t[0])

        # Build RAGTruth labels
        label_type_map = {
            "type1": "Evident Conflict",
            "type2": "Evident Baseless Info",
            "type3": "Overgeneration",
        }
        # Normalize: "type3_tool_overgen" → "type3", "type1_smart" → "type1", etc.
        label_type = label_type_map.get(h_type_base, label_type_map.get(h_type, "Evident Conflict"))
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
        # Drop rows where post-processing removed all spans — keeping a 0-label
        # row in the hallucinated dataset would be wrong (type3 always has spans
        # from its overgeneration structure so it never reaches here with 0).
        if h_type_base in ("type1", "type2") and not labels:
            return None

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
        is_overgen = h_type in ("type2_overgen", "type3_tool_overgen", "type3_1")
        is_type1 = h_type.startswith("type1_")

        if is_overgen:
            clean_output = item.get("overgenerated_answer", item.get("original_answer", ""))
        elif "tagged_answer" in item:
            # Only [field]/[/field] markup needs stripping. original_answer is
            # plain text and may legitimately contain "[here](url)"-style
            # markdown links — running _strip_tags on it would corrupt the
            # char offsets of any precomputed spans.
            clean_output = self._strip_tags(item["tagged_answer"])
        else:
            clean_output = item.get("original_answer", "")

        if is_type1:
            context = item.get("hallucinated_tool_response", item.get("tool_response", ""))
            # type1_1 / type1_2 store pre-computed labels from _t21_find_spans.
            # Use them directly — re-running _ragtruth_labels_type1 falls back to
            # detect_type1_spans which produces implicit whole-answer spans.
            precomputed = item.get("hallucination_labels")
            if precomputed and isinstance(precomputed, list):
                plain_override = item.get("plain_answer") or None
                out_text = plain_override if plain_override is not None else clean_output
                out_len = len(out_text)
                labels = [
                    {
                        "start":        l.get("start", -1),
                        "end":          l.get("end", -1),
                        "text":         l.get("text", ""),
                        "meta":         l.get("meta", ""),
                        "label_type":   l.get("label_type") or l.get("label", "Evident Conflict"),
                        "implicit_true": l.get("implicit_true", False),
                        "due_to_null":  l.get("due_to_null", False),
                    }
                    for l in precomputed
                    if l.get("start", -1) >= 0
                    # drop spans that cover >50% of the answer
                    and (out_len == 0 or (l.get("end", 0) - l.get("start", 0)) / out_len <= 0.5)
                ]
                summary = {
                    "evident_conflict": sum(1 for l in labels if not l.get("implicit_true")),
                    "baseless_info": 0,
                }
            else:
                labels, summary, plain_override = self._ragtruth_labels_type1(item, clean_output)
        elif h_type == "type2_deletion":
            # Use reduced_tool_response (after deletion) so the answer's
            # references to deleted fields are unsupported by the context
            context = item.get("reduced_tool_response", item.get("tool_response", ""))
            labels, summary, plain_override = self._ragtruth_labels_type2(item, clean_output)
        elif h_type == "type2_1_deletion":
            # hallucination_spans already contain valid start/end/text offsets
            # into original_answer (from _t21_find_spans) — use directly.
            context = item.get("reduced_tool_response", item.get("tool_response", ""))
            labels = [
                {
                    "start": sp["start"], "end": sp["end"], "text": sp["text"],
                    "meta": f"BASELESS_INFO\nSpan: {sp['text']}",
                    "label_type": "Baseless Info",
                    "implicit_true": False,
                    "due_to_null": True,
                }
                for sp in item.get("hallucination_spans", [])
            ]
            summary = {"evident_conflict": 0, "baseless_info": len(labels)}
            plain_override = None
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
