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

    # ── System prompts (override by passing system_prompt= to the API methods) ──

    TYPE1_SYSTEM_PROMPT = (
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
        "Your task is to add exactly one short, natural-sounding sentence at the end of the answer. "
        "This sentence must subtly reference a capability or data that belongs to one of the OTHER "
        "tools listed above (not the tool already used), BUT:\n"
        "- Do NOT name the tool.\n"
        "- Do NOT tell the user to 'use' or 'try' anything.\n"
        "- Instead, present the information as if it were a natural continuation — "
        "e.g. 'Historical price trends are also available for these indexes' or "
        "'Detailed nutritional breakdowns can be retrieved for each item'.\n"
        "- The sentence should sound like a helpful factual remark, not an explicit recommendation.\n"
        "- Keep it brief (one sentence). Do NOT invent capabilities not covered by the listed tools."
    )

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

    def type1_api(self, client, model: str, tool_response: str, user_prompt: str,
                  howhow: Literal["smart", "dumb"] = "smart",
                  temperature: float = 0.8, max_tokens: int = 2048,
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
        sys_prompt = system_prompt or self.TYPE1_SYSTEM_PROMPT
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
        return json.loads(resp.choices[0].message.content), target, unlocked_paths

    async def type1_api_async(self, client, model: str, tool_response: str,
                              user_prompt: str,
                              howhow: Literal["smart", "dumb"] = "smart",
                              temperature: float = 0.8, max_tokens: int = 2048,
                              system_prompt=None):
        """Async version of type1_api (requires AsyncOpenAI client)."""
        sys_prompt = system_prompt or self.TYPE1_SYSTEM_PROMPT
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
        return json.loads(resp.choices[0].message.content), target, unlocked_paths

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
    def _filter_system_tools(system_prompt: str, exclude_tool: str) -> str:
        """Remove the used tool from the system prompt's tool list.
        The LLM then can only reference OTHER tools in its overgeneration.
        """
        m = re.search(r'(\[\s*\{.*\}\s*\])', system_prompt, re.DOTALL)
        if not m:
            return system_prompt
        try:
            tools = json.loads(m.group(1))
            filtered = [t for t in tools if t.get("name", "") != exclude_tool]
            return json.dumps(filtered or tools, indent=2)
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

    def generate_type1_dataset(self, client, model: str, dataset: list,
                               output_path: str,
                               howhow: Literal["smart", "dumb"] = "smart",
                               temperature: float = 0.8, max_tokens: int = 4096,
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
                    hall_dict, target, unlocked_paths = self.type1_api(
                        client=client, model=model,
                        tool_response=row["tool_response"],
                        user_prompt=row["user_prompt"],
                        howhow=howhow, temperature=temperature, max_tokens=max_tokens,
                    )
                    original_data = json.loads(row["tool_response"])
                    spans, summary = self.evaluate_hallucination(
                        original_data, hall_dict, unlocked_paths
                    )
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
                        print(f"\n  [type1] row {idx} attempt {attempt} error: "
                              f"{type(_e).__name__}: {_e}")
                    continue
            if not success:
                skipped += 1
            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)

        print(f"Done: {len(generated)} rows saved to {output_path}, {skipped} skipped")
        return generated

    def generate_type2_dataset(self, dataset: list, output_path: str) -> list:
        """Generate Type 2 hallucinations for an entire dataset (no LLM, resumable).

        Each output row adds:
          reduced_tool_response, deleted_paths, deletion_target, hallucination_spans,
          hallucination_type
        """
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
            success = False
            for attempt in range(max_retries):
                try:
                    comment = self.type3_api(
                        client=client, model=model,
                        user_prompt=row["user_prompt"],
                        original_answer=row["original_answer"].rstrip(),
                        system_tools=row["system"],
                        tool_call=row["tool_call"],
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    if comment:
                        generated.append({
                            **row,
                            "overgenerated_answer": row["original_answer"].rstrip() + "\n\n" + comment,
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
        return generated

    # ══════════════════════════════════════════════════════════════════════════
    # DATASET GENERATION — async (batched, faster for large datasets)
    # ══════════════════════════════════════════════════════════════════════════

    async def _type1_single(self, client, model, row, howhow, temperature,
                            max_tokens, max_retries):
        """Process one row for async Type 1 generation. Returns row dict or None."""
        for attempt in range(max_retries):
            try:
                hall_dict, target, unlocked_paths = await self.type1_api_async(
                    client=client, model=model,
                    tool_response=row["tool_response"],
                    user_prompt=row["user_prompt"],
                    howhow=howhow, temperature=temperature, max_tokens=max_tokens,
                )
                original_data = json.loads(row["tool_response"])
                spans, summary = self.evaluate_hallucination(
                    original_data, hall_dict, unlocked_paths
                )
                return {
                    **row,
                    "hallucinated_tool_response": json.dumps(hall_dict, ensure_ascii=False),
                    "hallucination_target": target,
                    "unlocked_paths": unlocked_paths,
                    "hallucination_type": f"type1_{howhow}",
                    "eval_spans": spans,
                    "eval_summary": summary,
                }
            except Exception:
                continue
        return None

    async def generate_type1_dataset_async(self, client, model: str, dataset: list,
                                           output_path: str,
                                           howhow: Literal["smart", "dumb"] = "smart",
                                           temperature: float = 0.8, max_tokens: int = 4096,
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
                self._type1_single(client, model, dataset[idx], howhow,
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
        return generated

    async def _type3_single(self, client, model, row, temperature, max_tokens, max_retries):
        """Process one row for async Type 3 generation. Returns row dict or None."""
        if not row.get("system"):
            return None
        original_answer = row["original_answer"].rstrip()
        for attempt in range(max_retries):
            try:
                comment = await self.type3_api_async(
                    client=client, model=model,
                    user_prompt=row["user_prompt"],
                    original_answer=original_answer,
                    system_tools=row["system"],
                    tool_call=row["tool_call"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if comment:
                    return {
                        **row,
                        "overgenerated_answer": original_answer + "\n\n" + comment,
                        "overgeneration_comment": comment,
                        "hallucination_type": "type3_tool_overgen",
                        "used_tool": self._get_used_tool_name(row["tool_call"]),
                    }
            except Exception:
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
        return generated


# Need random import for type2_delete
import random
