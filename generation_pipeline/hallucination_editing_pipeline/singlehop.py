"""
singlehop.py — Type 1 / 2 / 3 hallucination logic for single-turn QA.

Defines SinglehopMixin.  HallucinationAuto inherits from it.

The three hallucination types
------------------------------
  Type 1 — Schema-based corruption (needs LLM + vLLM guided JSON decoding)
            The tool response is regenerated with some fields hallucinated.
            Input required: tool_response, user_prompt

  Type 2 — Deletion-based overgeneration (NO LLM)
            Fields are deleted from the tool response so the original answer
            now references data that no longer exists.
            Input required: tool_response, tagged_answer

  Type 3 — Tool-constrained overgeneration (needs LLM)
            One extra sentence is appended to the answer that subtly
            references a capability of a different (unused) tool.
            Input required: user_prompt, original_answer, system (tool list), tool_call

generate_type*_dataset functions loop over an entire dataset,
call the API per row, and save incrementally (resumable).
Async variants (_async suffix) use asyncio.gather for concurrency.
"""

import json
import re
import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Literal

from tqdm.auto import tqdm


class SinglehopMixin:

    @staticmethod
    def _is_whole_answer_span(spans: list, plain: str) -> bool:
        """True if any span covers >=50% of the answer (whole-answer noise).

        Catches both explicit 0→len spans and implicit whole-answer fallbacks
        produced by detect_answer_mismatch_spans when values can't be located.
        """
        if not spans or not plain:
            return False
        n = len(plain)
        return any(
            (s.get("end", 0) - s.get("start", 0)) >= n * 0.5
            for s in spans
        )

    def _save_ragtruth_alongside(self, json_path: str) -> str:
        """After saving the rich JSON to *json_path*, also save a RAGTruth-format
        JSONL next to it (same stem, .jsonl extension).

        Returns the JSONL path. Auto-detects singlehop vs multistep format.
        """
        try:
            jsonl_path = str(Path(json_path).with_suffix(".jsonl"))
            self.to_dataset(json_path, output_format="jsonl", output_path=jsonl_path)
            return jsonl_path
        except Exception as e:
            print(f"  [warn] RAGTruth JSONL write failed: {e}")
            return ""


    # ── System prompts (override by passing system_prompt= to the API methods) ──

    ANSWER_MISMATCH_SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer the user's question "
        "by calling the appropriate tool and returning the result as JSON."
    )

    TESTING_TYPE_SYSTEM_PROMPT = (
        "You will see user query and original answer. "
        "Your task is to suggest something with only one sentence "
        "that will be relevant to an answer - do some kind of recommendation:"
    )

    TYPE3_SYSTEM_PROMPT = (
        "You will see a user query and the original answer produced by a tool-calling assistant. "
        "The assistant had access ONLY to the following tools:\n\n{tools}\n\n"
        "Your task: output EXACTLY ONE short, natural-sounding sentence that subtly references "
        "a capability or data belonging to one of the OTHER tools listed above (not the tool "
        "already used).\n\n"
        "STRICT RULES:\n"
        "- Output ONLY the new sentence. Do NOT repeat or include the original answer.\n"
        "- Do NOT name the tool.\n"
        "- Do NOT tell the user to 'use' or 'try' anything.\n"
        "- Do NOT write filler like 'Let me know if you need further assistance' or "
        "'feel free to ask' or 'I am here to help' or 'If you have any questions'.\n"
        "- Present it as a natural continuation — e.g. 'Historical price trends are also available "
        "for these indexes' or 'Detailed nutritional breakdowns can be retrieved for each item'.\n"
        "- The sentence should sound like a helpful factual remark, not an explicit recommendation.\n"
        "- Keep it brief (one sentence). Do NOT invent capabilities not covered by the listed tools.\n"
        "- Your entire response must be this one sentence and nothing else."
    )

    # Filler pattern for stripping boilerplate from the end of original answers
    # and for rejecting generated comments that are just pleasantries.
    _T3_FILLER_RE = re.compile(
        r"feel free|let me know|if you need|anything else|further detail|"
        r"further assist|is there anything|happy to help|don'?t hesitate|"
        r"please let me|reach out|i(?:'?m| am) here|ask me|can i help|"
        r"any (other|more) (questions?|help|assist)",
        re.IGNORECASE,
    )

    @property
    def _t3_filler_re(self):
        return self._T3_FILLER_RE

    # ══════════════════════════════════════════════════════════════════════════
    # TYPE 2 — Deletion-based overgeneration (no LLM)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_data_leaves(obj, path=""):
        """Collect all (dot-path, value) pairs from a nested data structure."""
        leaves = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else k
                leaves.extend(SinglehopMixin._get_data_leaves(v, p))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                p = f"{path}.{i}" if path else str(i)
                leaves.extend(SinglehopMixin._get_data_leaves(v, p))
        else:
            leaves.append((path, obj))
        return leaves

    @staticmethod
    def _get_data_depth(obj, depth=0):
        """Return the maximum nesting depth of a data structure."""
        if isinstance(obj, dict):
            return max((SinglehopMixin._get_data_depth(v, depth + 1) for v in obj.values()),
                       default=depth)
        elif isinstance(obj, list):
            return max((SinglehopMixin._get_data_depth(v, depth + 1) for v in obj),
                       default=depth)
        return depth

    @staticmethod
    def _delete_at_path(obj, path_parts):
        """Delete a key/index from nested data. Returns (modified_obj, deleted_value)."""
        if len(path_parts) == 1:
            key = path_parts[0]
            if isinstance(obj, dict):
                return obj, obj.pop(key, None)
            else:
                return obj, obj.pop(int(key))
        key = path_parts[0]
        if isinstance(obj, dict):
            child, val = SinglehopMixin._delete_at_path(obj[key], path_parts[1:])
            obj[key] = child
        else:
            idx = int(key)
            child, val = SinglehopMixin._delete_at_path(obj[idx], path_parts[1:])
            obj[idx] = child
        return obj, val

    @staticmethod
    def _find_array_element_paths(results):
        """Find (path_to_element, path_to_array) pairs for cascade deletion."""
        paths = []
        def _walk(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _walk(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    paths.append((f"{path}.{i}" if path else str(i), path))
        _walk(results)
        return paths

    def type2_delete(self, tool_response: str, tagged_answer: str):
        """Delete data from tool_response so the original answer over-generates.

        Strategy depends on data depth:
          depth <= 1 (flat)  → delete 1-2 random leaf fields
          depth >= 2 (nested) → cascade-delete a whole array element or subtree

        Returns:
          (reduced_tool_response, deleted_paths, hallucination_spans, deletion_target)
          - reduced_tool_response  : JSON string with data removed
          - deleted_paths          : list of dot-paths that were deleted
          - hallucination_spans    : spans in the answer referencing deleted data
          - deletion_target        : the path(s) chosen for deletion
        """
        data = json.loads(tool_response)
        results = data.get("results", data)
        results_key = "results" if "results" in data else None
        prefix = "results." if results_key else ""

        depth = self._get_data_depth(results)
        all_leaves = [(p, v) for p, v in self._get_data_leaves(results)
                      if results_key or p != "name"]
        reduced = deepcopy(data)

        if depth <= 1:
            candidates = [p for p, _ in all_leaves]
            n_delete = min(random.randint(1, 2), len(candidates))
            to_delete = random.sample(candidates, n_delete)
            deleted_paths, deletion_target = [], to_delete
            for path in to_delete:
                full_path = f"{prefix}{path}"
                self._delete_at_path(reduced, full_path.split("."))
                deleted_paths.append(full_path)
        else:
            array_elements = self._find_array_element_paths(results)
            if array_elements:
                elem_path, _ = random.choice(array_elements)
                full_elem_path = f"{prefix}{elem_path}"
                # Collect leaf paths before deletion
                target_obj = results
                for part in elem_path.split("."):
                    target_obj = target_obj[part] if isinstance(target_obj, dict) else target_obj[int(part)]
                elem_leaves = self._get_data_leaves(target_obj, full_elem_path)
                deleted_paths = [p for p, _ in elem_leaves]
                deletion_target = full_elem_path
                self._delete_at_path(reduced, full_elem_path.split("."))
            else:
                candidates = [p for p, _ in all_leaves]
                n_delete = min(random.randint(1, 2), len(candidates))
                to_delete = random.sample(candidates, n_delete)
                deleted_paths, deletion_target = [], to_delete
                for path in to_delete:
                    full_path = f"{prefix}{path}"
                    self._delete_at_path(reduced, full_path.split("."))
                    deleted_paths.append(full_path)

        hallucination_spans = self._find_overgen_spans(tagged_answer, deleted_paths, tool_response)
        return (json.dumps(reduced, ensure_ascii=False), deleted_paths,
                hallucination_spans, deletion_target)

    @staticmethod
    def _find_overgen_spans(tagged_answer: str, deleted_paths: list, original_tool_response: str):
        """Find spans in tagged_answer that reference deleted data.

        tagged_answer uses [field]value[/field] markup.  We match tag names
        against the deleted dot-paths and locate the corresponding spans.
        """
        original_data = json.loads(original_tool_response)
        spans = []
        for path in deleted_paths:
            parts = path.split(".")
            tag_name = parts[-1]
            if tag_name.isdigit():
                for p in reversed(parts[:-1]):
                    if not p.isdigit() and p != "results":
                        tag_name = p
                        break
                else:
                    continue
            # Inline extract_value to avoid cross-import
            try:
                obj = original_data
                for part in path.split("."):
                    obj = obj[int(part)] if isinstance(obj, list) else obj[part]
                orig_val = obj
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            orig_str = str(orig_val)
            pattern = re.escape(f"[{tag_name}]") + r"(.*?)" + re.escape(f"[/{tag_name}]")
            for m in re.finditer(pattern, tagged_answer):
                matched_text = m.group(1)
                if orig_str in matched_text or matched_text in orig_str or orig_str == matched_text:
                    spans.append({
                        "path": path,
                        "tag": tag_name,
                        "span_start": m.start(),
                        "span_end": m.end(),
                        "span_text": m.group(0),
                        "original_value": orig_str,
                        "label_type": "Baseless Info",
                    })
        return spans

    # ══════════════════════════════════════════════════════════════════════════
    # TYPE 2.1 — Cascade deletion + direct span matching (no tagged_answer)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Tree helpers (used only by Type 2.1) ─────────────────────────────────

    @staticmethod
    def _t21_get_leaves(obj, path=""):
        """Return all (dot_path, scalar_value) pairs in a nested dict/list."""
        leaves = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else str(k)
                leaves.extend(SinglehopMixin._t21_get_leaves(v, p))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                p = f"{path}.{i}" if path else str(i)
                leaves.extend(SinglehopMixin._t21_get_leaves(v, p))
        else:
            leaves.append((path, obj))
        return leaves

    @staticmethod
    def _t21_get_by_path(obj, path):
        for key in path.split("."):
            obj = obj[int(key)] if isinstance(obj, list) else obj[key]
        return obj

    @staticmethod
    def _t21_delete_by_path(obj, path):
        keys = path.split(".")
        parent = obj
        try:
            for key in keys[:-1]:
                parent = parent[int(key)] if isinstance(parent, list) else parent[key]
            last = keys[-1]
            if isinstance(parent, list):
                parent.pop(int(last))
            else:
                del parent[last]
            return True
        except (KeyError, IndexError, TypeError):
            return False

    @staticmethod
    def _t21_cascade_target(results, leaf_path):
        """Return (delete_path, deleted_value) for cascade masking.

        Depth-1 (flat) → delete the leaf itself.
        Depth-2+       → delete the leaf's parent subtree.
        Returns (None, None) when the path is unresolvable (dotted key names).
        """
        depth = SinglehopMixin._get_data_depth(results)
        parts = leaf_path.split(".")
        if depth <= 1 or len(parts) == 1:
            try:
                return leaf_path, SinglehopMixin._t21_get_by_path(results, leaf_path)
            except Exception:
                return None, None
        parent_path = ".".join(parts[:-1])
        try:
            return parent_path, SinglehopMixin._t21_get_by_path(results, parent_path)
        except Exception:
            pass
        try:
            return leaf_path, SinglehopMixin._t21_get_by_path(results, leaf_path)
        except Exception:
            return None, None

    @staticmethod
    def _t21_find_spans(answer, deleted_value):
        """Simple case-insensitive substring span finder.

        For each scalar leaf value in deleted_value:
          - Count non-overlapping occurrences in answer (case-insensitive)
          - If count > 1  → return None  (ambiguous: caller skips this candidate)
          - If count == 1 → collect the span
          - If count == 0 → skip this leaf

        Values shorter than 3 characters are ignored to avoid false positives.

        Returns
        -------
        list of span dicts  — zero or more unambiguous spans
        None                — at least one value appears more than once; caller
                              should skip this deletion candidate entirely
        """
        leaf_values = (
            [v for _, v in SinglehopMixin._t21_get_leaves(deleted_value)]
            if isinstance(deleted_value, (dict, list))
            else [deleted_value]
        )

        ans_lower = answer.lower()
        spans = []

        def _find_occurrences(needle):
            """Return list of (start, end) for non-overlapping occurrences."""
            nl = needle.lower()
            occs, pos = [], 0
            while True:
                idx = ans_lower.find(nl, pos)
                if idx == -1:
                    break
                occs.append({"start": idx, "end": idx + len(needle),
                             "text": answer[idx:idx + len(needle)]})
                pos = idx + len(needle)
            return occs

        def _numeric_variants(needle):
            """Generate formatted variants of a numeric string.

            Covers: comma-thousands separator, common currency prefixes,
            percentage suffix, and integer form of whole floats.
            E.g. '2043.09' → ['2,043.09', '$2,043.09', '€2,043.09', ...]
            """
            try:
                f = float(needle)
            except ValueError:
                return []
            variants = []
            # integer form if whole number
            if f == int(f):
                variants.append(str(int(f)))
            # comma-separated thousands
            try:
                comma = f"{f:,.2f}".rstrip("0").rstrip(".")
                if comma != needle:
                    variants.append(comma)
                # also integer comma form
                if f == int(f):
                    variants.append(f"{int(f):,}")
            except Exception:
                pass
            # currency-prefixed versions
            for sym in ("$", "€", "£", "¥", "₹"):
                for v in list(variants) + [needle]:
                    variants.append(sym + v)
            # percentage suffix
            variants.append(needle + "%")
            variants.append(needle + " %")
            return variants

        _NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")

        for val in leaf_values:
            if val is None:
                continue
            needle = str(val).strip()
            if len(needle) < 3:
                continue

            occs = _find_occurrences(needle)

            # Numeric fallback: try formatted variants when exact match fails
            if not occs and _NUM_RE.match(needle):
                for variant in _numeric_variants(needle):
                    if len(variant) < 3:
                        continue
                    occs = _find_occurrences(variant)
                    if occs:
                        break   # use first variant that matches

            if len(occs) > 1:
                return None   # ambiguous — signal caller to skip candidate
            if len(occs) == 1:
                spans.append(occs[0])

        # Deduplicate by position (two leaf values may resolve to the same span)
        seen, out = set(), []
        for sp in sorted(spans, key=lambda x: x["start"]):
            k = (sp["start"], sp["end"])
            if k not in seen:
                seen.add(k)
                out.append(sp)
        return out

    def type2_1_delete(self, tool_response: str, answer: str,
                       skip_multi_span: bool = False):
        """Type 2.1 — cascade deletion with direct span matching.

        Does NOT require tagged_answer.  Uses word-boundary + lemma + n-gram
        matching to locate hallucinated spans in the plain answer text.

        Parameters
        ----------
        tool_response    : JSON string of the tool response
        answer           : plain assistant answer (no markup)
        skip_multi_span  : if True, return None when >1 span is found
                           (keeps only single, unambiguous hallucinations)

        Returns
        -------
        dict with keys:
          reduced_tool_response, delete_path, deleted_value,
          hallucination_spans, cascade, n_spans
        or None if no usable span is found (or multi-span and skip_multi_span=True)
        """
        data         = json.loads(tool_response)
        results      = data.get("results", data)
        prefix       = "results." if "results" in data else ""
        total_leaves = len(self._t21_get_leaves(results))

        leaves = self._t21_get_leaves(results)
        random.shuffle(leaves)

        for leaf_path, leaf_val in leaves:
            delete_path, deleted_value = self._t21_cascade_target(results, leaf_path)
            if delete_path is None:
                continue

            # Rule: deleted subtree must not exceed half of total leaves.
            # (Applies even when total_leaves == 1 — a single-field tool
            # response can never satisfy this, so it is skipped rather than
            # producing a fully-empty reduced_tool_response.)
            deleted_leaf_count = (
                len(self._t21_get_leaves(deleted_value))
                if isinstance(deleted_value, (dict, list))
                else 1
            )
            if deleted_leaf_count / total_leaves > 0.5:
                continue

            # Strategy 1: find spans for the cascade/leaf target
            spans = self._t21_find_spans(answer, deleted_value)

            if spans is None:
                # Ambiguous: some value appears >1 times → skip candidate
                continue

            if not spans and isinstance(deleted_value, (dict, list)):
                # Cascade target leaves had no clean matches.
                # Fallback: try the individual leaf as a flat deletion.
                spans = self._t21_find_spans(answer, leaf_val)
                if spans is None or not spans:
                    continue
                delete_path        = leaf_path
                deleted_value      = leaf_val
                deleted_leaf_count = 1

            elif not spans:
                continue

            if skip_multi_span and len(spans) > 1:
                continue

            # Rule: highlighted span(s) must not cover more than half the
            # answer text. If this leaf/subtree would mark most of the
            # answer as baseless, try a different leaf instead.
            ans_len = len(answer)
            if ans_len > 0:
                highlighted = sum(sp["end"] - sp["start"] for sp in spans)
                if highlighted / ans_len > 0.5:
                    continue

            reduced = deepcopy(data)
            full_delete_path = f"{prefix}{delete_path}"
            ok = self._t21_delete_by_path(reduced, full_delete_path)
            if not ok:
                continue

            cascade = isinstance(deleted_value, (dict, list))

            return {
                "reduced_tool_response":  json.dumps(reduced, ensure_ascii=False),
                "delete_path":            full_delete_path,
                "deleted_value":          deleted_value,
                "hallucination_spans":    spans,
                "cascade":                cascade,
                "n_spans":                len(spans),
                "pct_deleted":            round(100 * deleted_leaf_count / max(total_leaves, 1)),
            }

        return {"status": "skipped", "reason": "no_clean_span"}

    def generate_type2_1_dataset(self, dataset: list, output_path: str,
                                  skip_multi_span: bool = False) -> list:
        """Generate Type 2.1 hallucinations for an entire dataset.

        Parameters
        ----------
        dataset          : list of row dicts (needs 'tool_response' and
                           'original_answer' or 'answer')
        output_path      : path to write the output JSON
        skip_multi_span  : pass True to keep only single-span samples

        Each output row adds:
          reduced_tool_response, delete_path, deleted_value,
          hallucination_spans, cascade, n_spans, hallucination_type
        """
        out      = Path(output_path)
        skip_out = out.with_suffix(".skipped.json")

        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out) as f:
                generated = json.load(f)

        skipped_recs = []
        if skip_out.exists() and skip_out.stat().st_size > 0:
            with open(skip_out) as f:
                skipped_recs = json.load(f)

        start_idx    = len(generated) + len(skipped_recs)
        skipped_hard = 0  # parse errors / no results field

        for idx in tqdm(range(start_idx, len(dataset)), initial=start_idx,
                        total=len(dataset), desc="Type2.1"):
            row    = dataset[idx]
            answer = row.get("original_answer") or row.get("answer", "")
            try:
                _parsed  = json.loads(row["tool_response"])
                _results = _parsed.get("results", _parsed)
                if not isinstance(_results, (dict, list)):
                    skipped_hard += 1
                    continue
            except Exception:
                skipped_hard += 1
                continue
            try:
                result = self.type2_1_delete(
                    tool_response   = row["tool_response"],
                    answer          = answer,
                    skip_multi_span = skip_multi_span,
                )
                if result.get("status") == "skipped":
                    skipped_recs.append({
                        **row,
                        "skip_reason":      result["reason"],
                        "hallucination_type": "type2_1_deletion",
                    })
                else:
                    generated.append({
                        **row,
                        **result,
                        "hallucination_type": "type2_1_deletion",
                    })
            except Exception as e:
                print(f"  [type2.1] Row {idx} error: {e}")
                skipped_hard += 1

            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)
            with open(skip_out, "w") as f:
                json.dump(skipped_recs, f, ensure_ascii=False, indent=2)

        print(
            f"Done: {len(generated)} generated → {output_path}\n"
            f"      {len(skipped_recs)} soft-skipped (no clean span) → {skip_out}\n"
            f"      {skipped_hard} hard-skipped (parse error / no results)"
        )
        return generated

    # ══════════════════════════════════════════════════════════════════════════
    # TESTING TYPE — flexible free-form generation (for experimentation)
    # ══════════════════════════════════════════════════════════════════════════

    def testing_type_api(self, client, model: str, user_prompt: str,
                         original_answer: str, temperature: float = 0.7,
                         max_tokens: int = 150, system_prompt=None) -> str:
        """Send user_prompt + original_answer to the LLM and return one sentence.

        Useful for trying different hallucination ideas before committing to a type.
        Sync version — use testing_type_api_async for async/notebook use.
        """
        sys_prompt = system_prompt or self.TESTING_TYPE_SYSTEM_PROMPT
        user_content = json.dumps({"user_prompt": user_prompt,
                                   "original_answer": original_answer}, indent=2)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user",   "content": user_content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    async def testing_type_api_async(self, client, model: str, user_prompt: str,
                                     original_answer: str, temperature: float = 0.7,
                                     max_tokens: int = 150, system_prompt=None) -> str:
        """Async version of testing_type_api (requires AsyncOpenAI client)."""
        sys_prompt = system_prompt or self.TESTING_TYPE_SYSTEM_PROMPT
        user_content = json.dumps({"user_prompt": user_prompt,
                                   "original_answer": original_answer}, indent=2)
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user",   "content": user_content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # ══════════════════════════════════════════════════════════════════════════
    # TYPE 1 — Schema-based hallucination (guided JSON decoding)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _recover_json(s: str):
        """Best-effort recovery for truncated/malformed guided-JSON output.

        Tries four strategies in order; returns a dict/list or raises
        json.JSONDecodeError if all strategies fail.
        """
        s = s.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # Strip trailing spurious quote or brace
        for suffix in ('"', "'", "}"):
            if s.endswith(suffix):
                try:
                    return json.loads(s[:-1])
                except json.JSONDecodeError:
                    pass
        # Longest valid JSON prefix (handles mid-token truncation)
        for end in range(len(s) - 1, 0, -1):
            try:
                v = json.loads(s[:end])
                if isinstance(v, (dict, list)):
                    return v
            except json.JSONDecodeError:
                continue
        raise json.JSONDecodeError("all recovery strategies failed", s, 0)

    def answer_mismatch_api(self, client, model: str, tool_response: str, user_prompt: str,
                  howhow: Literal["smart", "dumb"] = "smart",
                  temperature: float = 0.8, max_tokens: int = 512,
                  system_prompt=None):
        """Generate a hallucinated tool response via vLLM guided JSON decoding.

        Builds a partially-unlocked JSON schema (see schema.py) and asks the
        LLM to fill it in.  Locked fields reproduce exact originals; unlocked
        fields are hallucinated.

        howhow:
          "smart" → CASCADE masking (create_cascade_schema)
          "dumb"  → FOCUSED masking (get_random_focused_schema)

        Returns:
          (hallucinated_dict, target_path, unlocked_paths)
        """
        sys_prompt = system_prompt or self.ANSWER_MISMATCH_SYSTEM_PROMPT
        locked = self.get_locked_schema_for_tool(tool_response)
        if howhow == "smart":
            target, unlocked_paths, schema_dict = self.create_cascade_schema(locked)
        else:
            target, schema_dict = self.get_random_focused_schema(locked)
            unlocked_paths = [target]
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user",   "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"guided_json": schema_dict},
        )
        return self._recover_json(resp.choices[0].message.content), target, unlocked_paths

    async def answer_mismatch_api_async(self, client, model: str, tool_response: str,
                              user_prompt: str,
                              howhow: Literal["smart", "dumb"] = "smart",
                              temperature: float = 0.8, max_tokens: int = 512,
                              system_prompt=None):
        """Async version of answer_mismatch_api (requires AsyncOpenAI client)."""
        sys_prompt = system_prompt or self.ANSWER_MISMATCH_SYSTEM_PROMPT
        locked = self.get_locked_schema_for_tool(tool_response)
        if howhow == "smart":
            target, unlocked_paths, schema_dict = self.create_cascade_schema(locked)
        else:
            target, schema_dict = self.get_random_focused_schema(locked)
            unlocked_paths = [target]
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user",   "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"guided_json": schema_dict},
        )
        return self._recover_json(resp.choices[0].message.content), target, unlocked_paths

    # ══════════════════════════════════════════════════════════════════════════
    # TYPE 3 — Tool-constrained overgeneration
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_tool_list_from_system(system_prompt: str) -> str:
        """Extract the JSON tool array from a ToolACE-style system prompt."""
        m = re.search(r'(\[\s*\{.*\}\s*\])', system_prompt, re.DOTALL)
        return m.group(1) if m else system_prompt

    @staticmethod
    def _get_used_tool_name(tool_call: str) -> str:
        """Extract the tool name from a tool_call string.

        Handles two formats:
          - Function-style: 'GetWeather(city="Paris")'
          - JSON-style:     '{"name": "get_stock_price", "arguments": {...}}'
        """
        tc = tool_call.strip().lstrip("[").rstrip("]").strip()
        # JSON-style: try json.loads first, then regex fallback for malformed JSON
        if tc.startswith("{"):
            try:
                return json.loads(tc).get("name", "").strip()
            except Exception:
                m = re.search(r'"name"\s*:\s*"([^"]+)"', tc)
                if m:
                    return m.group(1).strip()
        # Function-style: name is everything before the first '('
        m = re.match(r'([^({\s]+)', tc)
        return m.group(1).strip() if m else tc

    @staticmethod
    def _parse_tool_call_params(tool_call: str) -> dict:
        """Extract parameter names (and values where possible) from a tool_call string.

        Used by add_missing_system to build synthetic tool definitions.
        Returns {param_name: example_value} or {}.
        """
        tc = tool_call.strip().lstrip("[").rstrip("]").strip()
        params = {}
        # JSON-style
        if tc.startswith("{"):
            try:
                obj = json.loads(tc)
                args_str = obj.get("arguments", "{}")
                if isinstance(args_str, str):
                    args = json.loads(args_str)
                else:
                    args = args_str
                return {k: v for k, v in args.items()} if isinstance(args, dict) else {}
            except Exception:
                return {}
        # Function-style: extract key=value pairs from inside parentheses
        m = re.search(r'\((.+)\)$', tc, re.DOTALL)
        if not m:
            return {}
        inner = m.group(1)
        for part in re.split(r',\s*(?=[a-zA-Z_])', inner):
            kv = re.match(r'([a-zA-Z_]\w*)\s*=\s*(.*)', part.strip())
            if kv:
                k = kv.group(1)
                v = kv.group(2).strip().strip('"\'')
                params[k] = v
        return params

    @staticmethod
    def _filter_system_tools(system_prompt: str, exclude_tool: str,
                              max_tools: int = 8) -> str:
        """Remove the used tool from the system prompt's tool list.
        Caps the result at max_tools to keep the prompt within context limits.
        """
        m = re.search(r'(\[\s*\{.*\}\s*\])', system_prompt, re.DOTALL)
        if not m:
            return system_prompt
        try:
            tools = json.loads(m.group(1))
            filtered = [t for t in tools if t.get("name", "") != exclude_tool]
            if not filtered:
                filtered = tools
            # Cap to avoid blowing context window when registry is large
            if len(filtered) > max_tools:
                filtered = random.sample(filtered, max_tools)
            return json.dumps(filtered, indent=2)
        except json.JSONDecodeError:
            return m.group(1)

    def add_missing_system(self, dataset: list,
                           extra_datasets: list = None,
                           output_path: str = None) -> list:
        """Add a synthesised system column to rows that are missing it.

        Builds a tool registry by:
          1. Harvesting real tool definitions (name + description + parameters)
             from rows that already have a ``system`` column — in ``dataset``
             and in any ``extra_datasets`` supplied.
          2. Parsing ``tool_call`` strings for tool names and parameter names
             for tools not found in step 1.

        Each row without ``system`` receives a synthetic prompt listing
        **all** tools from the registry.  ``_filter_system_tools`` (called
        later by Type 3 generation) will remove the row's own tool, leaving
        the others for the LLM to reference.

        Parameters
        ----------
        dataset       : list of row dicts to enrich
        extra_datasets: optional list of additional datasets to harvest real
                        tool definitions from (e.g. your labelled type1/2/3
                        outputs which already have full system columns)
        output_path   : if given, save the result to this JSON path

        Returns
        -------
        list of row dicts — same order as input, with ``system`` added where
        it was missing.

        Example
        -------
        glaive = json.load(open("glaive_converted_toolcall_dataset.json"))
        tagged = json.load(open("dataset_v3_tagged_cleaned_sys.json"))

        enriched = ha.add_missing_system(glaive, extra_datasets=[tagged])
        # → every row now has a 'system' field
        """
        # ── Phase 1: harvest real tool definitions ────────────────────────────
        tool_registry: dict = {}   # name → full tool definition dict

        def _harvest_from_system(system_str: str) -> None:
            m = re.search(r'(\[\s*\{.*\}\s*\])', system_str, re.DOTALL)
            if not m:
                return
            try:
                tools = json.loads(m.group(1))
                for t in tools:
                    if isinstance(t, dict) and t.get("name"):
                        tool_registry.setdefault(t["name"], t)
            except Exception:
                pass

        all_sources = list(dataset)
        for xd in (extra_datasets or []):
            all_sources.extend(xd)

        for row in all_sources:
            if row.get("system"):
                _harvest_from_system(row["system"])

        print(f"  Phase 1: {len(tool_registry)} real tool definitions harvested")

        # ── Phase 2: add minimal defs for any unseen tool names ───────────────
        for row in dataset:
            tc = row.get("tool_call", "")
            if not tc:
                continue
            name = self._get_used_tool_name(tc)
            if not name or name in tool_registry:
                continue
            params = self._parse_tool_call_params(tc)
            tool_def: dict = {
                "name": name,
                "description": f"Performs {name.replace('_', ' ').lower()} operations.",
            }
            if params:
                tool_def["parameters"] = {
                    "type": "dict",
                    "properties": {
                        k: {"type": "string", "description": k}
                        for k in params
                    },
                    "required": list(params.keys()),
                }
            tool_registry[name] = tool_def

        print(f"  Phase 2: {len(tool_registry)} total tools in registry (after adding minimal defs)")

        # ── Phase 3: build system prompt template ─────────────────────────────
        tools_json = json.dumps(list(tool_registry.values()), ensure_ascii=False, indent=2)
        system_template = (
            "You are an expert in composing functions. You are given a question and "
            "a set of possible functions. Based on the question, you will need to make "
            "one or more function calls to achieve the purpose.\n\n"
            "Here is a list of functions in JSON format:\n" + tools_json
        )

        # ── Phase 4: enrich rows missing system ───────────────────────────────
        added = 0
        result = []
        for row in dataset:
            if not row.get("system"):
                row = {**row, "system": system_template}
                added += 1
            result.append(row)

        print(f"  Phase 4: {added}/{len(result)} rows enriched with synthetic system")

        if output_path:
            with open(output_path, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  Saved → {output_path}")

        return result

    def _t3_strip_filler(self, answer: str) -> str:
        """Remove trailing filler sentences/paragraphs from an answer.

        Two-pass approach applied repeatedly until stable:
          1. Strip trailing filler *paragraph* — block after the last blank line
             (\\n\\n). This handles the common case where filler appears as its
             own paragraph without a sentence-ending period before it.
          2. Strip trailing filler *sentence* — split on .!? + whitespace,
             remove trailing sentences that match the filler pattern.
        Returns the cleaned answer (may be the same string if no filler found).
        """
        filler_re = self._t3_filler_re
        text = answer.strip()
        changed = True
        while changed:
            changed = False
            # Pass 1: trailing filler paragraph
            parts = text.rsplit('\n\n', 1)
            if len(parts) == 2 and filler_re.search(parts[1]):
                text = parts[0].strip()
                changed = True
                continue
            # Pass 2: trailing filler sentence within a paragraph.
            # Two-alternative split: plain `.!?` + space, OR `.!?"'` + space
            # (fixed-width lookbehind so closing quotes stay in the preceding piece).
            sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?]["\'])\s+', text)
            if sentences and filler_re.search(sentences[-1]):
                sentences.pop()
                text = ' '.join(sentences).strip()
                changed = True
        return text

    def _t3_comment_ok(self, comment: str, base_answer: str) -> bool:
        """Return True if comment is a valid overgeneration sentence.

        Rejects:
        - Empty / too short (< 15 chars)
        - Filler / pleasantry sentences
        - Comment is a substring of the base answer (model repeated original content)
        - Comment spans > 3 sentences (model didn't follow single-sentence rule)
        """
        if not comment or len(comment) < 15:
            return False
        if self._t3_filler_re.search(comment):
            return False
        if comment.lower() in base_answer.lower():
            return False
        if len(re.split(r'(?<=[.!?])\s+', comment)) > 3:
            return False
        return True

    def type3_api(self, client, model: str, user_prompt: str, original_answer: str,
                  system_tools: str, tool_call: str,
                  temperature: float = 0.7, max_tokens: int = 150,
                  system_prompt=None) -> str:
        """Generate one overgeneration sentence referencing an unused tool.

        The used tool is excluded from the prompt so the LLM can only reference
        capabilities of the OTHER available tools.  The sentence sounds like a
        natural continuation, NOT a recommendation.

        Returns:
          A single sentence of subtle overgeneration.
        """
        used_tool = self._get_used_tool_name(tool_call)
        filtered_tools = self._filter_system_tools(system_tools, used_tool)
        sys_prompt = (system_prompt or self.TYPE3_SYSTEM_PROMPT).format(tools=filtered_tools)
        user_content = json.dumps({"user_prompt": user_prompt,
                                   "original_answer": original_answer}, indent=2)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user",   "content": user_content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    async def type3_api_async(self, client, model: str, user_prompt: str,
                              original_answer: str, system_tools: str, tool_call: str,
                              temperature: float = 0.7, max_tokens: int = 150,
                              system_prompt=None) -> str:
        """Async version of type3_api (requires AsyncOpenAI client)."""
        used_tool = self._get_used_tool_name(tool_call)
        filtered_tools = self._filter_system_tools(system_tools, used_tool)
        sys_prompt = (system_prompt or self.TYPE3_SYSTEM_PROMPT).format(tools=filtered_tools)
        user_content = json.dumps({"user_prompt": user_prompt,
                                   "original_answer": original_answer}, indent=2)
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user",   "content": user_content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # ══════════════════════════════════════════════════════════════════════════
    # DATASET GENERATION — sync (one row at a time, resumable)
    # ══════════════════════════════════════════════════════════════════════════

    def generate_answer_mismatch_dataset(self, client, model: str, dataset: list,
                               output_path: str,
                               howhow: Literal["smart", "dumb"] = "smart",
                               temperature: float = 0.8, max_tokens: int = 512,
                               max_retries: int = 3) -> list:
        """Generate Type 1 hallucinations for an entire dataset (sync, resumable).

        Saves after every row.  If output_path already exists with partial
        results, picks up from where it left off.

        Each output row adds:
          hallucinated_tool_response, hallucination_target, unlocked_paths,
          hallucination_type, eval_spans, eval_summary
        """
        out = Path(output_path)
        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out) as f:
                generated = json.load(f)
        start_idx = len(generated)
        skipped = 0

        for idx in tqdm(range(start_idx, len(dataset)), initial=start_idx,
                        total=len(dataset), desc="Type1"):
            row = dataset[idx]
            success = False
            for attempt in range(max_retries):
                try:
                    hall_dict, target, unlocked_paths = self.answer_mismatch_api(
                        client=client, model=model,
                        tool_response=row["tool_response"],
                        user_prompt=row["user_prompt"],
                        howhow=howhow, temperature=temperature, max_tokens=max_tokens,
                    )
                    original_data = json.loads(row["tool_response"])
                    spans, summary = self.evaluate_hallucination(
                        original_data, hall_dict, unlocked_paths
                    )
                    if not spans:
                        # LLM returned same values — no real hallucination, drop row
                        continue
                    generated.append({
                        **row,
                        "hallucinated_tool_response": json.dumps(hall_dict, ensure_ascii=False),
                        "hallucination_target": target,
                        "unlocked_paths": unlocked_paths,
                        "hallucination_type": f"type1_{howhow}",
                        "eval_spans": spans,
                        "eval_summary": summary,
                    })
                    success = True
                    break
                except Exception as _e:
                    if attempt == 0:
                        print(f"\n  [type1/answer_mismatch] row {idx} attempt {attempt} error: "
                              f"{type(_e).__name__}: {_e}")
                    continue
            if not success:
                skipped += 1
            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)

        print(f"Done: {len(generated)} rows saved to {output_path}, {skipped} skipped")
        csv = self._save_ragtruth_alongside(output_path)
        if csv:
            print(f"      RAGTruth JSONL: {csv}")
        return generated

    def generate_type2_dataset(self, dataset: list, output_path: str) -> list:
        """Generate Type 2 hallucinations for an entire dataset (no LLM, resumable).

        Rows missing 'tagged_answer' are auto-tagged via _tag_row before deletion.

        Each output row adds:
          reduced_tool_response, deleted_paths, deletion_target, hallucination_spans,
          hallucination_type
        """
        # Auto-tag rows missing tagged_answer
        missing = sum(1 for r in dataset if not r.get("tagged_answer"))
        if missing:
            print(f"  [type2] {missing} rows missing 'tagged_answer' — auto-tagging...")
            dataset = [self._tag_row(r) if not r.get("tagged_answer") else r
                       for r in dataset]

        out = Path(output_path)
        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out) as f:
                generated = json.load(f)
        start_idx = len(generated)
        skipped = 0

        for idx in tqdm(range(start_idx, len(dataset)), initial=start_idx,
                        total=len(dataset), desc="Type2"):
            row = dataset[idx]
            # Skip rows where results is a plain string — nothing to delete
            try:
                _parsed = json.loads(row["tool_response"])
                _results = _parsed.get("results", _parsed)
                if isinstance(_results, str):
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue
            try:
                reduced_tr, deleted_paths, spans, deletion_target = self.type2_delete(
                    tool_response=row["tool_response"],
                    tagged_answer=row["tagged_answer"],
                )
                generated.append({
                    **row,
                    "reduced_tool_response": reduced_tr,
                    "deleted_paths": deleted_paths,
                    "deletion_target": (deletion_target if isinstance(deletion_target, list)
                                        else [deletion_target]),
                    "hallucination_spans": spans,
                    "hallucination_type": "type2_deletion",
                })
            except Exception as e:
                print(f"  [type2] Row {idx} error: {e}")
                skipped += 1
            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)

        print(f"Done: {len(generated)} rows saved to {output_path}, {skipped} skipped")
        csv = self._save_ragtruth_alongside(output_path)
        if csv:
            print(f"      RAGTruth JSONL: {csv}")
        return generated

    def generate_type3_dataset(self, client, model: str, dataset: list,
                               output_path: str, temperature: float = 0.7,
                               max_tokens: int = 150, max_retries: int = 3) -> list:
        """Generate Type 3 hallucinations for an entire dataset (sync, resumable).

        If any rows are missing the 'system' column, it is synthesised
        automatically via add_missing_system before generation starts.

        Each output row adds:
          overgenerated_answer, overgeneration_comment, used_tool, hallucination_type
        """
        # Auto-enrich missing system columns
        missing = sum(1 for r in dataset if not r.get("system"))
        if missing:
            print(f"  [type3] {missing} rows missing 'system' — synthesising...")
            dataset = self.add_missing_system(dataset)

        out = Path(output_path)
        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out) as f:
                generated = json.load(f)
        start_idx = len(generated)
        skipped = 0

        for idx in tqdm(range(start_idx, len(dataset)), initial=start_idx,
                        total=len(dataset), desc="Type3"):
            row = dataset[idx]
            if not row.get("system"):
                skipped += 1
                continue
            # Strip filler from the end of the original answer before generation
            base_answer = self._t3_strip_filler(row["original_answer"].rstrip())
            if not base_answer:
                skipped += 1
                continue
            success = False
            for attempt in range(max_retries):
                try:
                    comment = self.type3_api(
                        client=client, model=model,
                        user_prompt=row["user_prompt"],
                        original_answer=base_answer,
                        system_tools=row["system"],
                        tool_call=row["tool_call"],
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    comment = comment.strip()
                    if not self._t3_comment_ok(comment, base_answer):
                        continue
                    generated.append({
                        **row,
                        "overgenerated_answer": base_answer + "\n\n" + comment,
                        "overgeneration_comment": comment,
                        "hallucination_type": "type3_tool_overgen",
                        "used_tool": self._get_used_tool_name(row["tool_call"]),
                    })
                    success = True
                    break
                except Exception as _e:
                    if attempt == 0:
                        print(f"\n  [type3] row {idx} attempt {attempt} error: "
                              f"{type(_e).__name__}: {_e}")
                    continue
            if not success:
                skipped += 1
            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)

        print(f"Done: {len(generated)} rows saved to {output_path}, {skipped} skipped")
        csv = self._save_ragtruth_alongside(output_path)
        if csv:
            print(f"      RAGTruth JSONL: {csv}")
        return generated

    # ══════════════════════════════════════════════════════════════════════════
    # DATASET GENERATION — async (batched, faster for large datasets)
    # ══════════════════════════════════════════════════════════════════════════

    async def _answer_mismatch_single(self, client, model, row, howhow, temperature,
                            max_tokens, max_retries):
        """Process one row for async Type 1 generation. Returns row dict or None.
        Retries on no-op (LLM returned same values, no actual hallucination).
        """
        try:
            tr_parsed = json.loads(row["tool_response"])
        except (json.JSONDecodeError, KeyError):
            return None

        # String-leaf: results is a plain string → will always produce a whole-answer span
        results = tr_parsed.get("results", tr_parsed)
        if isinstance(results, str):
            return None

        for attempt in range(max_retries):
            try:
                hall_dict, target, unlocked_paths = await self.answer_mismatch_api_async(
                    client=client, model=model,
                    tool_response=row["tool_response"],
                    user_prompt=row["user_prompt"],
                    howhow=howhow, temperature=temperature, max_tokens=max_tokens,
                )
                original_data = json.loads(row["tool_response"])
                spans, summary = self.evaluate_hallucination(
                    original_data, hall_dict, unlocked_paths
                )
                if not spans:
                    if attempt == max_retries - 1:
                        print(f"  [type1/answer_mismatch row] no spans after {max_retries} attempts "
                              f"(LLM returned same values) — target={target}")
                    continue
                enriched = {
                    **row,
                    "hallucinated_tool_response": json.dumps(hall_dict, ensure_ascii=False),
                    "hallucination_target": target,
                    "unlocked_paths": unlocked_paths,
                    "hallucination_type": f"type1_{howhow}",
                    "eval_spans": spans,
                    "eval_summary": summary,
                }
                # Run detect_answer_mismatch_spans now so export always has clean labels
                spans_labelled, plain = self.detect_answer_mismatch_spans(enriched)
                if self._is_whole_answer_span(spans_labelled, plain):
                    continue
                enriched["hallucination_labels"] = spans_labelled
                enriched["plain_answer"] = plain
                return enriched
            except Exception as _e:
                if attempt == 0:
                    print(f"  [type1/answer_mismatch row] error: {type(_e).__name__}: {_e}")
                continue
        return None

    async def generate_answer_mismatch_dataset_async(self, client, model: str, dataset: list,
                                           output_path: str,
                                           howhow: Literal["smart", "dumb"] = "smart",
                                           temperature: float = 0.8, max_tokens: int = 512,
                                           max_retries: int = 3, batch_size: int = 10) -> list:
        """Async batched Type 1 generation (batch_size rows processed concurrently).

        Requires an AsyncOpenAI client.  Saves after each batch.  Resumable.
        """
        out = Path(output_path)
        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out) as f:
                generated = json.load(f)
        start_idx = len(generated)
        remaining = list(range(start_idx, len(dataset)))
        skipped = 0
        pbar = tqdm(total=len(dataset), initial=start_idx, desc="Type1 Async")

        for batch_start in range(0, len(remaining), batch_size):
            batch_indices = remaining[batch_start:batch_start + batch_size]
            tasks = [
                self._answer_mismatch_single(client, model, dataset[idx], howhow,
                                   temperature, max_tokens, max_retries)
                for idx in batch_indices
            ]
            results = await asyncio.gather(*tasks)
            for res in results:
                if res is not None:
                    generated.append(res)
                else:
                    skipped += 1
            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)
            pbar.update(len(batch_indices))
            pbar.set_postfix(done=len(generated), skipped=skipped)

        pbar.close()
        print(f"Done: {len(generated)} rows saved to {output_path}, {skipped} skipped")
        csv = self._save_ragtruth_alongside(output_path)
        if csv:
            print(f"      RAGTruth JSONL: {csv}")
        return generated

    # ══════════════════════════════════════════════════════════════════════════
    # TYPE 1.1 — Schema-based hallucination + Type 2.1 span matching
    # Only runs on records that passed Type 2.1 (quality gate).
    # ══════════════════════════════════════════════════════════════════════════

    async def _answer_mismatch_gated_single(self, client, model, row, howhow, temperature,
                              max_tokens, max_retries):
        """Process one Type 2.1-successful row for Type 1.1.

        Runs Type 1 guided decoding to obtain a hallucinated tool response,
        then uses _t21_find_spans (simple case-insensitive find + numeric
        normalization) to locate the changed original values in the answer.

        Returns enriched row dict or None on failure / no clean spans.
        """
        try:
            tr_parsed = json.loads(row["tool_response"])
        except (json.JSONDecodeError, KeyError):
            return None

        results = tr_parsed.get("results", tr_parsed)
        if isinstance(results, str):
            return None

        answer = row.get("original_answer") or row.get("answer", "")
        if not answer:
            return None

        for attempt in range(max_retries):
            try:
                hall_dict, target, unlocked_paths = await self.answer_mismatch_api_async(
                    client=client, model=model,
                    tool_response=row["tool_response"],
                    user_prompt=row["user_prompt"],
                    howhow=howhow, temperature=temperature, max_tokens=max_tokens,
                )

                # Find which leaf values actually changed
                changes, orig_leaves = self._span_diff(
                    row["tool_response"], json.dumps(hall_dict, ensure_ascii=False)
                )
                if not changes:
                    continue  # LLM returned same values — retry

                # 50% cap: consistent with Type 2.1 — count leaves in results only,
                # not the whole tool_response (which includes the `name` field).
                total_leaves = len(self._t21_get_leaves(results))
                if total_leaves > 1 and len(changes) / total_leaves > 0.5:
                    continue

                # Search for each changed *original* value in the answer
                all_spans = []
                for path, c in changes.items():
                    orig_val = c["orig"]
                    if not orig_val or len(str(orig_val)) < 3:
                        continue
                    spans = self._t21_find_spans(answer, orig_val)
                    if spans is None:  # ambiguous (>1 occurrence) — skip this path
                        continue
                    all_spans.extend(spans)

                if not all_spans:
                    continue

                # Deduplicate and sort spans
                seen, out_spans = set(), []
                for sp in sorted(all_spans, key=lambda x: x["start"]):
                    k = (sp["start"], sp["end"])
                    if k not in seen:
                        seen.add(k)
                        out_spans.append(sp)

                # Drop records where any span covers >50% of the answer
                ans_len = len(answer)
                if ans_len and any(
                    (sp["end"] - sp["start"]) / ans_len > 0.5 for sp in out_spans
                ):
                    continue

                return {
                    **row,
                    "hallucinated_tool_response": json.dumps(hall_dict, ensure_ascii=False),
                    "hallucination_target": target,
                    "unlocked_paths": unlocked_paths,
                    "hallucination_type": "type1_1",
                    "hallucination_labels": [
                        {"start": sp["start"], "end": sp["end"],
                         "text": sp["text"], "label": "Evident Conflict"}
                        for sp in out_spans
                    ],
                    "plain_answer": answer,
                }
            except Exception as _e:
                if attempt == 0:
                    print(f"  [type1.1/answer_mismatch_gated row] error: {type(_e).__name__}: {_e}")
                continue
        return None

    async def generate_answer_mismatch_gated_dataset_async(self, client, model: str,
                                             type2_1_output_path: str,
                                             output_path: str,
                                             howhow: Literal["smart", "dumb"] = "smart",
                                             temperature: float = 0.8,
                                             max_tokens: int = 512,
                                             max_retries: int = 3,
                                             batch_size: int = 10) -> list:
        """Async batched Type 1.1 generation.

        Loads the Type 2.1 output JSON, filters to successful records (those
        without ``status: skipped`` and with ``hallucination_spans``), then
        runs Type 1 guided-JSON decoding + Type 2.1 span matching on each.

        Resumable: if output_path already exists it picks up from where it left off.
        Saves after every batch.

        Parameters
        ----------
        client               : AsyncOpenAI pointed at a vLLM server
        model                : model name on the vLLM server
        type2_1_output_path  : path to the Type 2.1 successful-records JSON
        output_path          : where to write Type 1.1 results
        howhow               : "smart" (cascade schema) or "dumb" (focused schema)
        temperature          : sampling temperature
        max_tokens           : max tokens for guided JSON generation
        max_retries          : retries per row before giving up
        batch_size           : concurrent rows per batch
        """
        with open(type2_1_output_path) as f:
            type2_1_data = json.load(f)

        successful = [
            r for r in type2_1_data
            if r.get("status") != "skipped" and "hallucination_spans" in r
        ]
        print(f"  Type 2.1 input  : {len(type2_1_data)} records")
        print(f"  Successful (gate): {len(successful)} records")

        out = Path(output_path)
        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out) as f:
                generated = json.load(f)
        start_idx = len(generated)
        remaining = list(range(start_idx, len(successful)))
        skipped = 0
        pbar = tqdm(total=len(successful), initial=start_idx, desc="Type1.1 Async")

        for batch_start in range(0, len(remaining), batch_size):
            batch_indices = remaining[batch_start:batch_start + batch_size]
            tasks = [
                self._answer_mismatch_gated_single(
                    client, model, successful[idx],
                    howhow, temperature, max_tokens, max_retries,
                )
                for idx in batch_indices
            ]
            results = await asyncio.gather(*tasks)
            for res in results:
                if res is not None:
                    generated.append(res)
                else:
                    skipped += 1
            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)
            pbar.update(len(batch_indices))
            pbar.set_postfix(done=len(generated), skipped=skipped)

        pbar.close()
        print(f"Done: {len(generated)} rows saved to {output_path}, {skipped} skipped")
        csv = self._save_ragtruth_alongside(output_path)
        if csv:
            print(f"      RAGTruth JSONL: {csv}")
        return generated

    # ══════════════════════════════════════════════════════════════════════════
    # TYPE 1.2 — Span-targeted schema hallucination
    #
    # Improvement over 1.1: instead of letting guided-JSON change any leaf,
    # we first identify which leaf values actually appear in the original answer
    # (using _t21_find_spans), then unlock ONLY those paths in the schema.
    # This eliminates the "no spans found" failure mode (21% of 1.1 retries)
    # and should raise the yield from ~79% to ~95%+.
    # ══════════════════════════════════════════════════════════════════════════

    def _find_answer_quoted_paths(self, tool_response_str: str, answer: str) -> list:
        """Return dot-paths of all leaves in tool_response whose value appears
        in *answer* (via _t21_find_spans, case-insensitive + numeric variants).

        Skips 'name' and values shorter than 3 chars.
        Skips ambiguous matches (>1 occurrence).
        """
        try:
            tr = json.loads(tool_response_str)
        except Exception:
            return []
        results = tr.get("results", tr)
        if isinstance(results, (str, int, float, bool)) or results is None:
            return []

        leaves = self._t21_get_leaves(results)
        quoted = []
        for path, val in leaves:
            if not val or len(str(val)) < 3:
                continue
            spans = self._t21_find_spans(answer, val)
            if spans:  # non-None and non-empty → value appears exactly once
                quoted.append(f"results.{path}" if not path.startswith("results") else path)
        return quoted

    async def answer_mismatch_targeted_api_async(self, client, model: str, tool_response: str,
                                user_prompt: str, target_paths: list,
                                total_leaves: int,
                                temperature: float = 0.8, max_tokens: int = 512):
        """Async guided-JSON call that unlocks ONLY the answer-quoted leaf paths."""
        locked = self.get_locked_schema_for_tool(tool_response)
        target, unlocked_paths, schema_dict = self.create_span_targeted_schema(
            locked, target_paths, total_leaves
        )
        target_values = []
        try:
            tr = json.loads(tool_response)
            for p in unlocked_paths:
                parts = p.split(".")
                v = tr
                for part in parts:
                    if isinstance(v, list):
                        v = v[int(part)]
                    elif isinstance(v, dict):
                        v = v.get(part)
                    else:
                        v = None
                    if v is None:
                        break
                if v is not None:
                    target_values.append(str(v))
        except Exception:
            pass

        sys_prompt = self.ANSWER_MISMATCH_SYSTEM_PROMPT
        if target_values:
            sys_prompt += (
                "\n\nCRITICAL INSTRUCTION: The following values are INCORRECT and "
                "MUST be replaced with different, plausible alternatives of the same "
                "type and format. Do NOT reproduce any of these values:\n"
                + "\n".join(f'  - "{v}"' for v in target_values)
            )

        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user",   "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"guided_json": schema_dict},
        )
        return self._recover_json(resp.choices[0].message.content), target, unlocked_paths

    async def _answer_mismatch_targeted_single(self, client, model, row, temperature,
                               max_tokens, max_retries):
        """Process one Type 2.1 row for Type 1.2 generation.

        Finds answer-quoted leaf paths, builds a targeted schema that unlocks
        only those paths, then runs guided-JSON decoding.  Because the
        unlocked values are guaranteed to be in the answer, the span gate
        should succeed on every attempt.
        """
        try:
            tr_parsed = json.loads(row["tool_response"])
        except (json.JSONDecodeError, KeyError):
            return None

        results = tr_parsed.get("results", tr_parsed)
        if isinstance(results, str):
            return None

        answer = row.get("original_answer") or row.get("answer", "")
        if not answer:
            return None

        target_paths = self._find_answer_quoted_paths(row["tool_response"], answer)
        if not target_paths:
            return None  # no answer-quoted leaves → nothing to target

        total_leaves = len(self._t21_get_leaves(results))

        for attempt in range(max_retries):
            # Escalate temperature on retries: LLM tends to reproduce original
            # values at low temperature (IDs, short numbers, single-word values).
            attempt_temp = min(temperature + attempt * 0.2, 1.4)
            try:
                hall_dict, target, unlocked_paths = await self.answer_mismatch_targeted_api_async(
                    client=client, model=model,
                    tool_response=row["tool_response"],
                    user_prompt=row["user_prompt"],
                    target_paths=target_paths,
                    total_leaves=total_leaves,
                    temperature=attempt_temp,
                    max_tokens=max_tokens,
                )

                changes, orig_leaves = self._span_diff(
                    row["tool_response"], json.dumps(hall_dict, ensure_ascii=False)
                )
                if not changes:
                    continue

                all_spans = []
                for path, c in changes.items():
                    orig_val = c["orig"]
                    if not orig_val or len(str(orig_val)) < 3:
                        continue
                    spans = self._t21_find_spans(answer, orig_val)
                    if spans is None:
                        continue
                    all_spans.extend(spans)

                if not all_spans:
                    continue

                seen, out_spans = set(), []
                for sp in sorted(all_spans, key=lambda x: x["start"]):
                    k = (sp["start"], sp["end"])
                    if k not in seen:
                        seen.add(k)
                        out_spans.append(sp)

                # Drop records where any span covers >50% of the answer
                ans_len = len(answer)
                if ans_len and any(
                    (sp["end"] - sp["start"]) / ans_len > 0.5 for sp in out_spans
                ):
                    continue

                return {
                    **row,
                    "hallucinated_tool_response": json.dumps(hall_dict, ensure_ascii=False),
                    "hallucination_target": target,
                    "unlocked_paths": unlocked_paths,
                    "hallucination_type": "type1_2",
                    "hallucination_labels": [
                        {"start": sp["start"], "end": sp["end"],
                         "text": sp["text"], "label": "Evident Conflict"}
                        for sp in out_spans
                    ],
                    "plain_answer": answer,
                }
            except Exception as _e:
                if attempt == 0:
                    print(f"  [type1.2/answer_mismatch_targeted row] error: {type(_e).__name__}: {_e}")
                continue
        return None

    async def generate_answer_mismatch_targeted_dataset_async(self, client, model: str,
                                             type2_1_output_path: str,
                                             output_path: str,
                                             temperature: float = 0.8,
                                             max_tokens: int = 512,
                                             max_retries: int = 3,
                                             batch_size: int = 10) -> list:
        """Async batched Type 1.2 generation.

        Same pipeline as 1.1 but uses span-targeted schema: only unlocks leaf
        values that actually appear in the original answer, eliminating the
        'no spans found' failure mode.

        Resumable — skips rows already in output_path.
        """
        with open(type2_1_output_path) as f:
            type2_1_data = json.load(f)

        successful = [
            r for r in type2_1_data
            if r.get("status") != "skipped" and "hallucination_spans" in r
        ]
        print(f"  Type 2.1 input  : {len(type2_1_data)} records")
        print(f"  Successful (gate): {len(successful)} records")

        out = Path(output_path)
        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out) as f:
                generated = json.load(f)
        start_idx = len(generated)
        remaining = list(range(start_idx, len(successful)))
        skipped = 0
        pbar = tqdm(total=len(successful), initial=start_idx, desc="Type1.2 Async")

        for batch_start in range(0, len(remaining), batch_size):
            batch_indices = remaining[batch_start:batch_start + batch_size]
            tasks = [
                self._answer_mismatch_targeted_single(
                    client, model, successful[idx],
                    temperature, max_tokens, max_retries,
                )
                for idx in batch_indices
            ]
            results = await asyncio.gather(*tasks)
            for res in results:
                if res is not None:
                    generated.append(res)
                else:
                    skipped += 1
            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)
            pbar.update(len(batch_indices))
            pbar.set_postfix(done=len(generated), skipped=skipped)

        pbar.close()
        print(f"Done: {len(generated)} rows saved to {output_path}, {skipped} skipped")
        csv = self._save_ragtruth_alongside(output_path)
        if csv:
            print(f"      RAGTruth JSONL: {csv}")
        return generated

    async def _type3_single(self, client, model, row, temperature, max_tokens,
                            max_retries, hall_type="type3_tool_overgen"):
        """Process one row for async Type 3 / 3.1 generation. Returns row dict or None."""
        if not row.get("system"):
            return None
        # Strip filler from the end of the original answer before generation
        base_answer = self._t3_strip_filler(row["original_answer"].rstrip())
        if not base_answer:
            return None
        for attempt in range(max_retries):
            try:
                comment = await self.type3_api_async(
                    client=client, model=model,
                    user_prompt=row["user_prompt"],
                    original_answer=base_answer,
                    system_tools=row["system"],
                    tool_call=row["tool_call"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                comment = comment.strip()
                if not self._t3_comment_ok(comment, base_answer):
                    continue
                return {
                    **row,
                    "overgenerated_answer": base_answer + "\n\n" + comment,
                    "overgeneration_comment": comment,
                    "hallucination_type": hall_type,
                    "used_tool": self._get_used_tool_name(row["tool_call"]),
                }
            except Exception as _e:
                if attempt == 0:
                    print(f"  [type3 row] error: {type(_e).__name__}: {_e}")
                continue
        return None

    async def generate_type3_dataset_async(self, client, model: str, dataset: list,
                                           output_path: str, temperature: float = 0.7,
                                           max_tokens: int = 150, max_retries: int = 3,
                                           batch_size: int = 10) -> list:
        """Async batched Type 3 generation (batch_size rows processed concurrently).

        Requires an AsyncOpenAI client.  Saves after each batch.  Resumable.
        Missing 'system' columns are synthesised automatically.
        """
        # Auto-enrich missing system columns
        missing = sum(1 for r in dataset if not r.get("system"))
        if missing:
            print(f"  [type3 async] {missing} rows missing 'system' — synthesising...")
            dataset = self.add_missing_system(dataset)

        out = Path(output_path)
        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out) as f:
                generated = json.load(f)
        start_idx = len(generated)
        remaining = list(range(start_idx, len(dataset)))
        skipped = 0
        pbar = tqdm(total=len(dataset), initial=start_idx, desc="Type3 Async")

        for batch_start in range(0, len(remaining), batch_size):
            batch_indices = remaining[batch_start:batch_start + batch_size]
            tasks = [
                self._type3_single(client, model, dataset[idx],
                                   temperature, max_tokens, max_retries)
                for idx in batch_indices
            ]
            results = await asyncio.gather(*tasks)
            for res in results:
                if res is not None:
                    generated.append(res)
                else:
                    skipped += 1
            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)
            pbar.update(len(batch_indices))
            pbar.set_postfix(done=len(generated), skipped=skipped)

        pbar.close()
        print(f"Done: {len(generated)} rows saved to {output_path}, {skipped} skipped")
        csv = self._save_ragtruth_alongside(output_path)
        if csv:
            print(f"      RAGTruth JSONL: {csv}")
        return generated

    # ══════════════════════════════════════════════════════════════════════════
    # TYPE 3.1 — Overgeneration with quality gates (filler filtering)
    # ══════════════════════════════════════════════════════════════════════════

    async def generate_type3_1_dataset_async(self, client, model: str, dataset: list,
                                             output_path: str,
                                             temperature: float = 0.7,
                                             max_tokens: int = 150,
                                             max_retries: int = 3,
                                             batch_size: int = 10) -> list:
        """Async batched Type 3.1 generation — cleaner overgeneration.

        Identical to Type 3 but with two quality gates applied at every row:

        1. **Filler stripping** — trailing boilerplate sentences (e.g. 'Let me
           know if you need anything!') are removed from the original answer
           *before* the LLM sees it.  This prevents the judge from flagging
           unlabelled filler in the base answer as a missing hallucination span.

        2. **Comment validation** — generated comments that are themselves
           filler, repeat the original answer, or span more than 3 sentences
           are rejected and retried.

        Output ``hallucination_type`` is ``"type3_1"`` (vs ``"type3_tool_overgen"``
        for the original Type 3).  Resumable.  Saves after each batch.
        """
        missing = sum(1 for r in dataset if not r.get("system"))
        if missing:
            print(f"  [type3.1 async] {missing} rows missing 'system' — synthesising...")
            dataset = self.add_missing_system(dataset)

        out = Path(output_path)
        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out) as f:
                generated = json.load(f)
        start_idx = len(generated)
        remaining = list(range(start_idx, len(dataset)))
        skipped = 0
        pbar = tqdm(total=len(dataset), initial=start_idx, desc="Type3.1 Async")

        for batch_start in range(0, len(remaining), batch_size):
            batch_indices = remaining[batch_start:batch_start + batch_size]
            tasks = [
                self._type3_single(
                    client, model, dataset[idx],
                    temperature, max_tokens, max_retries,
                    hall_type="type3_1",
                )
                for idx in batch_indices
            ]
            results = await asyncio.gather(*tasks)
            for res in results:
                if res is not None:
                    generated.append(res)
                else:
                    skipped += 1
            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)
            pbar.update(len(batch_indices))
            pbar.set_postfix(done=len(generated), skipped=skipped)

        pbar.close()
        print(f"Done: {len(generated)} rows saved to {output_path}, {skipped} skipped")
        csv = self._save_ragtruth_alongside(output_path)
        if csv:
            print(f"      RAGTruth JSONL: {csv}")
        return generated


# Need random import for type2_delete
import random
