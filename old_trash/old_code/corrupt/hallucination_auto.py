"""HallucinationAuto - Automated tool-call hallucination generator."""

import torch 
import json
import re
import math
import asyncio
# from vllm import LLM, SamplingParams
from typing import Literal, Union, Optional, List
import os
import sys
from pathlib import Path
from pydantic import Field, BaseModel
import random 
from copy import deepcopy
from openai import AsyncOpenAI
from tqdm import tqdm

import math
import asyncio
from pathlib import Path
from tqdm.auto import tqdm

class HallucinationAuto():
    # Field names that look numeric but should stay as strings
    _KEEP_AS_STRING_KEYS = {"id", "zip", "zip_code", "postal_code", "postal", "code", 
                            "phone", "phone_number", "fax", "ssn", "isbn", "ean", "upc"}

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", max_model_len: int = 1024, gpu_memory_utilization: float = 0.7):
        """Automated tool-call hallucination generator.

        Generates three types of hallucinations from tool-calling dialogue data:

        - **Type 1** -- Schema-based JSON corruption via guided decoding
        - **Type 2** -- Deletion-based overgeneration (no LLM required)
        - **Type 3** -- Subtle tool-constrained overgeneration

        Parameters
        ----------
        model_name : str, default "Qwen/Qwen2.5-3B-Instruct"
            HuggingFace model identifier (reserved for local vLLM; not used
            when calling a remote OpenAI-compatible endpoint).
        max_model_len : int, default 1024
            Maximum context length for local vLLM engine (unused for remote).
        gpu_memory_utilization : float, default 0.7
            Fraction of GPU memory for local vLLM engine (unused for remote).

        Notes
        -----
        For remote usage, create an ``openai.OpenAI`` client and pass it to
        ``type1_api``, ``type3_api``, etc.

        Examples
        --------
        >>> ha = HallucinationAuto()
        >>> from openai import OpenAI
        >>> client = OpenAI(base_url="http://host:9091/v1", api_key="key")
        >>> hall, target, paths = ha.type1_api(client, "model", tool_resp, prompt)
        """
        pass

    #------------------------------------------------------
    # Schema builders
    #------------------------------------------------------

    def build_locked_schema(self, data, is_root=True, field_name: str = ""):
        """Build a fully locked JSON schema from actual data.

        Every leaf value is pinned via ``enum`` constraints so guided decoding
        reproduces the exact original JSON.  Arrays use ``prefixItems`` to lock
        each element independently (no cross-contamination between elements).

        Parameters
        ----------
        data : dict | list | str | int | float | bool | None
            The parsed JSON data to build the schema from.
        is_root : bool, default True
            Whether this is the root call (used internally for recursion).
        field_name : str, default ""
            Current field name (used internally to detect string-like IDs).

        Returns
        -------
        dict
            A JSON-Schema-compatible dictionary where every leaf has an
            ``enum`` constraint locking it to its original value.

        See Also
        --------
        get_locked_schema_for_tool : Convenience wrapper that parses JSON first.
        create_cascade_schema : Selectively unlocks parts of the locked schema.

        Examples
        --------
        >>> schema = ha.build_locked_schema({"name": "foo", "results": {"temp": 72}})
        >>> schema["properties"]["results"]["properties"]["temp"]
        {'type': 'integer', 'enum': [72]}
        """
        if isinstance(data, dict):
            properties = {}
            for key, value in data.items():
                properties[key] = self.build_locked_schema(value, is_root=False, field_name=key)
            return {
                "type": "object",
                "properties": properties,
                "required": list(data.keys()),
                "additionalProperties": False
            }
        elif isinstance(data, list):
            if len(data) > 0:
                prefix_items = [self.build_locked_schema(item, is_root=False, field_name=field_name) for item in data]
            else:
                prefix_items = []
            return {
                "type": "array",
                "prefixItems": prefix_items,
                "minItems": len(data),
                "maxItems": len(data)
            }
        # LEAVES — preserve exact original type, no conversion
        elif isinstance(data, bool):
            return {"type": "boolean", "enum": [data]}
        elif isinstance(data, int):
            return {"type": "integer", "enum": [data]}
        elif isinstance(data, float):
            return {"type": "number", "enum": [data]}
        elif isinstance(data, str):
            return {"type": "string", "enum": [data]}
        elif data is None:
            return {"type": "null"}
        else:
            return {"type": "string", "enum": [str(data)]}

    def get_locked_schema_for_tool(self, tool_call_json_str: str) -> dict:
        parsed_json = json.loads(tool_call_json_str)
        return self.build_locked_schema(parsed_json, is_root=True)

    #------------------------------------------------------
    # Tree inspection
    #------------------------------------------------------

    def print_schema_tree(self, schema: dict, name="root", indent="", defs=None, is_last=True, current_path=""):
        if defs is None:
            defs = schema.get("$defs", {})
        if "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            schema = defs[ref_name]
        marker = "└── " if is_last else "├── "
        node_type = schema.get("type", "object")
        locked = " [LOCKED]" if "enum" in schema else ""
        path_display = f"  [Path: {current_path}]" if current_path else ""
        print(f"{indent}{marker}{name} ({node_type}){locked}{path_display}")
        indent += "    " if is_last else "│   "
        if node_type == "object" and "properties" in schema:
            props = schema["properties"]
            keys = list(props.keys())
            for i, k in enumerate(keys):
                next_path = f"{current_path}.{k}" if current_path else k
                self.print_schema_tree(props[k], k, indent, defs, i == len(keys)-1, next_path)
        elif node_type == "array" and "prefixItems" in schema:
            for i, item_schema in enumerate(schema["prefixItems"]):
                next_path = f"{current_path}.{i}" if current_path else str(i)
                self.print_schema_tree(item_schema, f"[{i}]", indent, defs, i == len(schema["prefixItems"])-1, next_path)
        elif node_type == "array" and "items" in schema:
            next_path = f"{current_path}.items" if current_path else "items"
            self.print_schema_tree(schema["items"], "items", indent, defs, True, next_path)

    def get_all_leaf_paths(self, schema: dict, current_path: str = "") -> list:
        """Collect all leaf-node dot-paths from a JSON schema.

        Leaf nodes are terminal values (string, integer, number, boolean, null).
        Array indices are represented as integers in the path.

        Parameters
        ----------
        schema : dict
            A JSON schema dictionary (locked or unlocked).
        current_path : str, default ""
            Prefix for recursion (used internally).

        Returns
        -------
        list of str
            Dot-notation paths to every leaf, e.g.
            ``["name", "results.items.0.title", "results.items.0.price"]``.

        See Also
        --------
        get_all_internal_paths : Returns non-leaf (object/array) paths instead.

        Examples
        --------
        >>> ha.get_all_leaf_paths(schema)
        ['name', 'results.temp', 'results.humidity']
        """
        paths = []
        node_type = schema.get("type", "object")
        if node_type == "object" and "properties" in schema:
            for key, val in schema["properties"].items():
                next_path = f"{current_path}.{key}" if current_path else key
                paths.extend(self.get_all_leaf_paths(val, next_path))
        elif node_type == "array" and "prefixItems" in schema:
            for i, item_schema in enumerate(schema["prefixItems"]):
                next_path = f"{current_path}.{i}" if current_path else str(i)
                paths.extend(self.get_all_leaf_paths(item_schema, next_path))
        elif node_type == "array" and "items" in schema:
            next_path = f"{current_path}.items" if current_path else "items"
            paths.extend(self.get_all_leaf_paths(schema["items"], next_path))
        else:
            paths.append(current_path)
        return paths

    def get_all_internal_paths(self, schema: dict, current_path: str = "") -> list:
        """Collect all internal (non-leaf, non-root) node paths from a schema.

        Internal nodes are objects, arrays, and array elements that contain
        further structure.  Used by ``get_random_focused_schema`` (DUMB
        masking) to pick a subtree to unlock.

        Parameters
        ----------
        schema : dict
            A JSON schema dictionary.
        current_path : str, default ""
            Prefix for recursion (used internally).

        Returns
        -------
        list of str
            Dot-notation paths to every internal node, e.g.
            ``["results", "results.items", "results.items.0"]``.

        See Also
        --------
        get_all_leaf_paths : Returns leaf paths instead.
        """
        paths = []
        node_type = schema.get("type", "object")
        if node_type == "object" and "properties" in schema:
            if current_path:  # skip root
                paths.append(current_path)
            for key, val in schema["properties"].items():
                next_path = f"{current_path}.{key}" if current_path else key
                paths.extend(self.get_all_internal_paths(val, next_path))
        elif node_type == "array" and "prefixItems" in schema:
            if current_path:
                paths.append(current_path)
            for i, item_schema in enumerate(schema["prefixItems"]):
                next_path = f"{current_path}.{i}" if current_path else str(i)
                paths.extend(self.get_all_internal_paths(item_schema, next_path))
        elif node_type == "array" and "items" in schema:
            if current_path:
                paths.append(current_path)
        return paths

    #------------------------------------------------------
    # Group / sibling detection
    #------------------------------------------------------

    def find_group(self, schema: dict, leaf_path: str) -> str | None:
        """Find the innermost array-element group that contains a leaf.

        When a leaf is inside an array (``prefixItems``), this returns the
        dot-path to the specific array element (e.g. ``"results.items.0"``).
        Used by ``create_cascade_schema`` to decide between array-mode
        and flat-mode unlocking.

        Parameters
        ----------
        schema : dict
            The locked JSON schema.
        leaf_path : str
            Dot-notation path to a leaf node.

        Returns
        -------
        str or None
            Path to the innermost array element containing the leaf, or
            ``None`` if the leaf is not inside any array.

        Examples
        --------
        >>> ha.find_group(schema, 'results.items.0.title')
        'results.items.0'
        >>> ha.find_group(schema, 'results.temp')  # flat, no array
        """
        parts = leaf_path.split(".")
        node, so_far, group = schema, [], None
        for part in parts:
            if node.get("type") == "array" and "prefixItems" in node:
                so_far.append(part)
                group = ".".join(so_far)
                node = node["prefixItems"][int(part)]
            elif node.get("type") == "object" and part in node.get("properties", {}):
                so_far.append(part)
                node = node["properties"][part]
            else:
                break
        return group

    @staticmethod
    def _get_node_at_path(schema, path_parts):
        """Navigate schema and return the node at the given path."""
        node = schema
        for part in path_parts:
            if node.get("type") == "array" and "prefixItems" in node:
                node = node["prefixItems"][int(part)]
            elif node.get("type") == "object" and part in node.get("properties", {}):
                node = node["properties"][part]
            else:
                break
        return node

    @staticmethod
    def _find_parent_path(leaf_path):
        """Return (parent_path, leaf_key). E.g. 'results.city' → ('results', 'city')"""
        parts = leaf_path.split(".")
        if len(parts) == 1:
            return "", parts[0]
        return ".".join(parts[:-1]), parts[-1]

    def _get_siblings(self, schema, leaf_path):
        """Get sibling leaf keys under the same parent object.
        Skips root-level 'name' (always locked)."""
        parent_path, leaf_key = self._find_parent_path(leaf_path)
        if not parent_path:
            parent_node = schema
        else:
            parent_node = self._get_node_at_path(schema, parent_path.split("."))
        if parent_node.get("type") != "object" or "properties" not in parent_node:
            return parent_path, [leaf_key]
        all_keys = list(parent_node["properties"].keys())
        if not parent_path:
            siblings = [k for k in all_keys if k != "name"]
        else:
            siblings = all_keys
        return parent_path, siblings

    #------------------------------------------------------
    # Unlock operations
    #------------------------------------------------------

    @staticmethod
    def _free_all(node):
        """Recursively strip all enum/minItems/maxItems constraints from a subtree.
        Types are preserved exactly as in the locked schema."""
        n = deepcopy(node)
        t = n.get("type", "")
        if t == "object" and "properties" in n:
            n["properties"] = {k: HallucinationAuto._free_all(v) for k, v in n["properties"].items()}
        elif t == "array":
            if "prefixItems" in n:
                n["prefixItems"] = [HallucinationAuto._free_all(s) for s in n["prefixItems"]]
            if "items" in n:
                n["items"] = HallucinationAuto._free_all(n["items"])
            n.pop("minItems", None)
            n.pop("maxItems", None)
        else:
            n.pop("enum", None)
        return n

    @staticmethod
    def _apply_at_path(schema, path_parts, fn):
        """Navigate schema tree along path_parts and apply fn at the target node.
        Mutates schema in-place — caller should deepcopy first."""
        if not path_parts:
            return fn(schema)
        key = path_parts[0]
        rest = path_parts[1:]
        if schema.get("type") == "array" and "prefixItems" in schema:
            idx = int(key)
            schema["prefixItems"][idx] = HallucinationAuto._apply_at_path(schema["prefixItems"][idx], rest, fn)
        elif schema.get("type") == "object" and key in schema.get("properties", {}):
            schema["properties"][key] = HallucinationAuto._apply_at_path(schema["properties"][key], rest, fn)
        return schema

    #------------------------------------------------------
    # Masking strategies
    #------------------------------------------------------

    def create_cascade_schema(self, locked_schema: dict) -> tuple:
        """CASCADE masking -- group-aware selective unlocking.

        Picks a random leaf (depth-weighted, skips root ``name``), then unlocks
        a contextually coherent region of the schema:

        * **Array case** -- leaf is inside an array element: unlock the entire
          element so the LLM can hallucinate a coherent row of data.
        * **Flat case** -- leaf is NOT in an array: unlock sibling fields under
          the same parent (<=3 siblings: all; >3: random 2..ceil(N/2) including
          the chosen leaf).

        Parameters
        ----------
        locked_schema : dict
            Fully locked schema from ``build_locked_schema``.

        Returns
        -------
        chosen_leaf : str
            Dot-path of the initially selected leaf.
        unlocked_paths : list of str
            All dot-paths that were actually unlocked (superset of chosen_leaf).
        schema : dict
            Deep copy of ``locked_schema`` with the selected region unlocked.

        See Also
        --------
        get_random_focused_schema : Simpler "DUMB" masking alternative.

        Examples
        --------
        >>> target, paths, schema = ha.create_cascade_schema(locked)
        >>> target
        'results.items.0.price'
        """
        all_paths = self.get_all_leaf_paths(locked_schema)
        candidates = [p for p in all_paths if p != "name"] or all_paths
        if not candidates:
            raise ValueError("No hallucinatable leaf paths found")

        weights = [p.count(".") + 1 for p in candidates]
        chosen = random.choices(candidates, weights=weights, k=1)[0]

        group = self.find_group(locked_schema, chosen)
        schema = deepcopy(locked_schema)

        if group:
            # ── ARRAY CASE: unlock entire array element ──
            self._apply_at_path(schema, group.split("."), self._free_all)
            unlocked_paths = [p for p in all_paths if p.startswith(group + ".") or p == group]
        else:
            # ── FLAT CASE: unlock sibling group ──
            parent_path, siblings = self._get_siblings(locked_schema, chosen)
            _, chosen_key = self._find_parent_path(chosen)

            if len(siblings) <= 3:
                keys_to_unlock = siblings
            else:
                max_unlock = max(2, math.ceil(len(siblings) / 2))
                n_unlock = random.randint(2, max_unlock)
                other_siblings = [k for k in siblings if k != chosen_key]
                n_extras = min(n_unlock - 1, len(other_siblings))
                extras = random.sample(other_siblings, n_extras)
                keys_to_unlock = [chosen_key] + extras

            unlocked_paths = []
            for key in keys_to_unlock:
                full_path = f"{parent_path}.{key}" if parent_path else key
                self._apply_at_path(schema, full_path.split("."), self._free_all)
                unlocked_paths.append(full_path)

        return chosen, unlocked_paths, schema

    def get_random_focused_schema(self, locked_schema: dict):
        """DUMB masking -- unlock a random internal subtree.

        Picks a random internal (non-leaf) node weighted by depth and unlocks
        all constraints beneath it.  Simpler but less targeted than
        ``create_cascade_schema``; good for diverse corruption.

        Parameters
        ----------
        locked_schema : dict
            Fully locked schema from ``build_locked_schema``.

        Returns
        -------
        target : str
            Dot-path of the chosen internal node.
        schema : dict
            Deep copy of ``locked_schema`` with the subtree unlocked.

        See Also
        --------
        create_cascade_schema : Smarter group-aware masking.

        Examples
        --------
        >>> target, schema = ha.get_random_focused_schema(locked)
        >>> target
        'results.items.0'
        """
        internal = self.get_all_internal_paths(locked_schema)
        candidates = [p for p in internal if p != "name"] or internal
        if not candidates:
            fields = list(locked_schema['properties'].keys())
            candidates = [f for f in fields if "enum" not in locked_schema['properties'][f]] or fields
            target = random.choice(candidates)
            s = deepcopy(locked_schema)
            if target in s.get("properties", {}):
                s["properties"][target] = self._free_all(s["properties"][target])
            return target, s

        weights = [p.count(".") + 1 for p in candidates]
        target = random.choices(candidates, weights=weights, k=1)[0]
        s = deepcopy(locked_schema)
        self._apply_at_path(s, target.split("."), self._free_all)
        return target, s

    #------------------------------------------------------
    # Value extraction helper
    #------------------------------------------------------

    @staticmethod
    def extract_value(obj, path):
        """Extract a value from a nested dict/list using a dot-notation path.

        Parameters
        ----------
        obj : dict or list
            The nested data structure to navigate.
        path : str
            Dot-notation path, e.g. ``"results.items.0.price"``.
            Numeric segments are treated as list indices.

        Returns
        -------
        any
            The value found at the given path.

        Raises
        ------
        KeyError
            If a dict key in the path does not exist.
        IndexError
            If a list index in the path is out of range.

        Examples
        --------
        >>> data = {"results": {"items": [{"price": 9.99}]}}
        >>> HallucinationAuto.extract_value(data, "results.items.0.price")
        9.99
        """
        for part in path.split("."):
            if isinstance(obj, list):
                obj = obj[int(part)]
            else:
                obj = obj[part]
        return obj

    #------------------------------------------------------
    # Tool name extraction
    #------------------------------------------------------

    @staticmethod
    def get_tool_names(tool_response_str: str) -> list[str]:
        """Extract tool/function names from a tool-response JSON string.

        Parameters
        ----------
        tool_response_str : str
            Raw JSON string of a tool response.  May be a single object
            ``{"name": "...", "results": {...}}`` or a list of such objects.

        Returns
        -------
        list of str
            Tool names found (usually one element).

        Examples
        --------
        >>> HallucinationAuto.get_tool_names('{"name":"GetWeather","results":{}}')
        ['GetWeather']
        """
        parsed = json.loads(tool_response_str)
        names = []
        if isinstance(parsed, dict):
            if "name" in parsed:
                names.append(parsed["name"])
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "name" in item:
                    names.append(item["name"])
        return names

    #------------------------------------------------------
    # Hallucination evaluation / logging
    #------------------------------------------------------

    @staticmethod
    def evaluate_hallucination(original_data, hallucinated_data, unlocked_paths: list[str]):
        """Compare original and hallucinated tool responses and classify changes.

        Walks every unlocked path, compares original vs generated values, and
        labels each change as *Evident Conflict* (value replaced) or
        *Baseless Info* (value fabricated from ``None``).

        Parameters
        ----------
        original_data : dict or str
            Original tool response (parsed dict or JSON string).
        hallucinated_data : dict or str
            Hallucinated tool response (parsed dict or JSON string).
        unlocked_paths : list of str
            Dot-paths that were unlocked during schema masking.

        Returns
        -------
        spans : list of dict
            Each dict contains ``path``, ``text``, ``meta``, ``label_type``
            (``"Evident Conflict"`` or ``"Baseless Info"``), ``implicit_true``,
            and ``due_to_null``.
        summary : dict
            ``{"evident_conflict": int, "baseless_info": int}``

        Examples
        --------
        >>> spans, summary = ha.evaluate_hallucination(orig, hall, paths)
        >>> summary
        {'evident_conflict': 2, 'baseless_info': 0}
        """
        if isinstance(original_data, str):
            original_data = json.loads(original_data)
        if isinstance(hallucinated_data, str):
            hallucinated_data = json.loads(hallucinated_data)

        spans = []
        evident_conflict = 0
        baseless_info = 0

        for path in unlocked_paths:
            try:
                orig_val = HallucinationAuto.extract_value(original_data, path)
            except (KeyError, IndexError, TypeError):
                orig_val = None
            try:
                hall_val = HallucinationAuto.extract_value(hallucinated_data, path)
            except (KeyError, IndexError, TypeError):
                hall_val = None

            if orig_val == hall_val:
                continue

            due_to_null = orig_val is None
            implicit_true = False

            if orig_val is not None and hall_val is not None and orig_val != hall_val:
                label_type = "Evident Conflict"
                evident_conflict += 1
            elif due_to_null and hall_val is not None:
                label_type = "Baseless Info"
                baseless_info += 1
            elif hall_val is None and orig_val is not None:
                label_type = "Evident Conflict"
                evident_conflict += 1
            else:
                label_type = "Baseless Info"
                baseless_info += 1

            orig_str = str(orig_val) if orig_val is not None else "null"
            hall_str = str(hall_val) if hall_val is not None else "null"

            spans.append({
                "path": path,
                "text": f"{path}: {hall_str}",
                "meta": f"{label_type}\nOriginal: {orig_str}\nGenerated: {hall_str}",
                "label_type": label_type,
                "implicit_true": implicit_true,
                "due_to_null": due_to_null,
            })

        summary = {
            "evident_conflict": evident_conflict,
            "baseless_info": baseless_info,
        }
        return spans, summary


    #------------------------------------------------------
    # Default prompts (overridable)
    #------------------------------------------------------

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


    #------------------------------------------------------
    # Type 2: Overgeneration via deletion (no LLM needed)
    #------------------------------------------------------

    @staticmethod
    def _get_data_leaves(obj, path=""):
        """Get all (path, value) leaf pairs from actual data (not schema)."""
        leaves = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else k
                leaves.extend(HallucinationAuto._get_data_leaves(v, p))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                p = f"{path}.{i}" if path else str(i)
                leaves.extend(HallucinationAuto._get_data_leaves(v, p))
        else:
            leaves.append((path, obj))
        return leaves

    @staticmethod
    def _get_data_depth(obj, depth=0):
        """Get max depth of nested data structure."""
        if isinstance(obj, dict):
            if not obj:
                return depth
            return max(HallucinationAuto._get_data_depth(v, depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return depth
            return max(HallucinationAuto._get_data_depth(v, depth + 1) for v in obj)
        return depth

    @staticmethod
    def _delete_at_path(obj, path_parts):
        """Delete a key/index from nested data. Returns (modified_obj, deleted_value).
        For dict: removes the key. For list: removes the element."""
        if len(path_parts) == 1:
            key = path_parts[0]
            if isinstance(obj, dict):
                val = obj.pop(key, None)
                return obj, val
            elif isinstance(obj, list):
                idx = int(key)
                val = obj.pop(idx)
                return obj, val
        else:
            key = path_parts[0]
            if isinstance(obj, dict):
                child, val = HallucinationAuto._delete_at_path(obj[key], path_parts[1:])
                obj[key] = child
            elif isinstance(obj, list):
                idx = int(key)
                child, val = HallucinationAuto._delete_at_path(obj[idx], path_parts[1:])
                obj[idx] = child
            return obj, val

    @staticmethod
    def _find_array_element_paths(results):
        """Find paths to array elements (for cascade deletion).
        Returns list of (path_to_element, path_to_array) tuples.
        E.g., ('trends.0', 'trends') for results.trends[0]."""
        paths = []
        def _walk(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    p = f"{path}.{k}" if path else k
                    _walk(v, p)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    elem_path = f"{path}.{i}" if path else str(i)
                    paths.append((elem_path, path))
        _walk(results)
        return paths

    def type2_delete(self, tool_response: str, tagged_answer: str):
        """Type 2 hallucination: delete data so the original answer over-generates.

        Removes part of the tool response so that the *original* answer now
        references data that no longer exists -- creating overgeneration without
        any LLM call.

        Strategy depends on data depth:

        * **Depth <= 1** (flat) -- delete 1-2 random leaf fields.
        * **Depth >= 2** (nested/arrays) -- cascade-delete an entire array
          element or subtree.

        Parameters
        ----------
        tool_response : str
            Original tool-response JSON string.
        tagged_answer : str
            The ``tagged_answer`` column with ``[field]value[/field]`` markup
            indicating which answer tokens came from which JSON fields.

        Returns
        -------
        reduced_tool_response : str
            Tool-response JSON string with selected data removed.
        deleted_paths : list of str
            Dot-paths of all leaf values that were deleted.
        hallucination_spans : list of dict
            Spans in the answer that now reference deleted (missing) data.
            Each dict has keys ``path``, ``tag``, ``span_start``, ``span_end``,
            ``span_text``, ``original_value``, ``label_type``.
        deletion_target : str or list
            The path(s) that were chosen for deletion.

        Examples
        --------
        >>> reduced, paths, spans, target = ha.type2_delete(tool_resp, tagged_ans)
        >>> paths
        ['results.items.1.title', 'results.items.1.price']
        >>> len(spans)
        2
        """
        data = json.loads(tool_response)
        results = data.get("results", data)
        results_key = "results" if "results" in data else None

        # Get depth of results subtree
        depth = self._get_data_depth(results)

        # Get all leaves under results
        if results_key:
            all_leaves = [(p, v) for p, v in self._get_data_leaves(results)]
            prefix = "results."
        else:
            all_leaves = [(p, v) for p, v in self._get_data_leaves(data) if p != "name"]
            prefix = ""

        reduced = deepcopy(data)

        if depth <= 1:
            # FLAT: delete 1-2 random leaves from results
            candidates = [p for p, v in all_leaves]
            n_delete = min(random.randint(1, 2), len(candidates))
            to_delete = random.sample(candidates, n_delete)
            deleted_paths = []
            deletion_target = to_delete

            for path in to_delete:
                full_path = f"{prefix}{path}" if prefix else path
                parts = full_path.split(".")
                self._delete_at_path(reduced, parts)
                # Collect all leaf paths under this deletion
                deleted_paths.append(full_path)
        else:
            # DEEP: try cascade-delete an array element first
            array_elements = self._find_array_element_paths(results)
            if array_elements:
                # Pick a random array element
                elem_path, arr_path = random.choice(array_elements)
                full_elem_path = f"{prefix}{elem_path}" if prefix else elem_path

                # Collect leaf paths BEFORE deletion
                target_in_results = results
                for part in elem_path.split("."):
                    if isinstance(target_in_results, dict):
                        target_in_results = target_in_results[part]
                    elif isinstance(target_in_results, list):
                        target_in_results = target_in_results[int(part)]

                elem_leaves = self._get_data_leaves(target_in_results, full_elem_path)
                deleted_paths = [p for p, v in elem_leaves]
                deletion_target = full_elem_path

                # Delete the array element
                parts = full_elem_path.split(".")
                self._delete_at_path(reduced, parts)
            else:
                # No arrays — delete 1-2 leaves
                candidates = [p for p, v in all_leaves]
                n_delete = min(random.randint(1, 2), len(candidates))
                to_delete = random.sample(candidates, n_delete)
                deleted_paths = []
                deletion_target = to_delete

                for path in to_delete:
                    full_path = f"{prefix}{path}" if prefix else path
                    parts = full_path.split(".")
                    self._delete_at_path(reduced, parts)
                    deleted_paths.append(full_path)

        # Find hallucination spans using tagged_answer
        hallucination_spans = self._find_overgen_spans(tagged_answer, deleted_paths, tool_response)

        return (
            json.dumps(reduced, ensure_ascii=False),
            deleted_paths,
            hallucination_spans,
            deletion_target,
        )

    @staticmethod
    def _find_overgen_spans(tagged_answer: str, deleted_paths: list[str], original_tool_response: str):
        """Find spans in tagged_answer that reference deleted data.

        Uses tag names from deleted paths and matches against [tag]value[/tag] in tagged_answer.
        Returns list of span dicts.
        """
        original_data = json.loads(original_tool_response)
        spans = []

        for path in deleted_paths:
            # Get the tag name (last component of path, skip numeric indices)
            parts = path.split(".")
            tag_name = parts[-1]
            # Skip numeric-only tags (array indices are not tag names)
            if tag_name.isdigit():
                # Use the parent field name if available
                for p in reversed(parts[:-1]):
                    if not p.isdigit() and p != "results":
                        tag_name = p
                        break
                else:
                    continue

            # Get the original value at this path
            try:
                orig_val = HallucinationAuto.extract_value(original_data, path)
            except (KeyError, IndexError, TypeError):
                continue

            orig_str = str(orig_val)

            # Find [tag_name]...[/tag_name] in tagged answer
            pattern = re.escape(f"[{tag_name}]") + r"(.*?)" + re.escape(f"[/{tag_name}]")
            for m in re.finditer(pattern, tagged_answer):
                matched_text = m.group(1)
                # Check if this occurrence matches the deleted value
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


    #------------------------------------------------------
    # Testing type: flexible prompt-based generation
    #------------------------------------------------------

    def testing_type_api(
        self,
        client,
        model: str,
        user_prompt: str,
        original_answer: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        system_prompt: str | None = None,
    ) -> str:
        """Flexible free-form text generation for experimentation.

        Sends user prompt + original answer to the LLM with a customisable
        system prompt.  Useful for testing different hallucination strategies
        before committing to a specific type.

        Parameters
        ----------
        client : openai.OpenAI
            An ``openai.OpenAI`` client configured with the vLLM endpoint.
        model : str
            Model identifier, e.g. ``"qwen2.5-3b-instruct"``.
        user_prompt : str
            The original user question.
        original_answer : str
            The original (correct) assistant answer.
        temperature : float, default 0.7
            Sampling temperature (higher = more creative).
        max_tokens : int, default 150
            Maximum tokens for the generated comment.
        system_prompt : str or None, default None
            Custom system prompt.  Falls back to
            ``TESTING_TYPE_SYSTEM_PROMPT`` if ``None``.

        Returns
        -------
        str
            The generated comment / recommendation.

        See Also
        --------
        testing_type_api_async : Async variant (requires ``AsyncOpenAI``).

        Examples
        --------
        >>> comment = ha.testing_type_api(client, "model", user_q, answer)
        """
        sys_prompt = system_prompt or self.TESTING_TYPE_SYSTEM_PROMPT
        user_content = json.dumps({
            "user_prompt": user_prompt,
            "original_answer": original_answer,
        }, indent=2)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    async def testing_type_api_async(
        self,
        client,
        model: str,
        user_prompt: str,
        original_answer: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        system_prompt: str | None = None,
    ) -> str:
        """Async version of ``testing_type_api``.

        Parameters
        ----------
        client : openai.AsyncOpenAI
            An ``AsyncOpenAI`` client configured with the vLLM endpoint.
        model : str
            Model identifier.
        user_prompt : str
            The original user question.
        original_answer : str
            The original (correct) assistant answer.
        temperature : float, default 0.7
            Sampling temperature.
        max_tokens : int, default 150
            Maximum tokens for the generated comment.
        system_prompt : str or None, default None
            Custom system prompt.

        Returns
        -------
        str
            The generated comment / recommendation.
        """
        sys_prompt = system_prompt or self.TESTING_TYPE_SYSTEM_PROMPT
        user_content = json.dumps({
            "user_prompt": user_prompt,
            "original_answer": original_answer,
        }, indent=2)

        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


    #------------------------------------------------------
    # Type 1: Generation functions
    #------------------------------------------------------

    def type1_api(self, client, model: str, 
                  tool_response: str, user_prompt: str,
                  howhow: Literal['smart', 'dumb'] = 'smart',
                  temperature: float = 0.8, max_tokens: int = 2048,
                  system_prompt: str | None = None):
        """Type 1 hallucination: generate a corrupted tool response via guided JSON decoding.

        Builds a partially-unlocked JSON schema from the original tool response,
        then asks the LLM to fill in the structure.  Locked fields reproduce
        exact original values; unlocked fields are hallucinated by the model.

        Parameters
        ----------
        client : openai.OpenAI
            An ``openai.OpenAI`` client configured with the vLLM endpoint.
        model : str
            Model identifier, e.g. ``"qwen2.5-3b-instruct"``.
        tool_response : str
            Original tool-response JSON string.
        user_prompt : str
            The original user question (used as LLM input).
        howhow : {"smart", "dumb"}, default "smart"
            Masking strategy:

            * ``"smart"`` -- CASCADE masking (``create_cascade_schema``),
              unlocks a coherent group of sibling fields.
            * ``"dumb"`` -- unlocks an entire random subtree
              (``get_random_focused_schema``).
        temperature : float, default 0.8
            Sampling temperature.
        max_tokens : int, default 2048
            Maximum tokens for the generated JSON.
        system_prompt : str or None, default None
            Custom system prompt.  Falls back to ``TYPE1_SYSTEM_PROMPT``.

        Returns
        -------
        hallucinated_dict : dict
            The generated (corrupted) tool response as a parsed dictionary.
        target_path : str
            Dot-path of the initially chosen leaf.
        unlocked_paths : list of str
            All dot-paths that were unlocked (for evaluation).

        See Also
        --------
        type1_api_async : Async variant (requires ``AsyncOpenAI``).
        evaluate_hallucination : Compare original vs hallucinated responses.

        Examples
        --------
        >>> hall, target, paths = ha.type1_api(
        ...     client, "qwen2.5-3b-instruct",
        ...     tool_response=row["tool_response"],
        ...     user_prompt=row["user_prompt"],
        ... )
        >>> target
        'results.temperature'
        """
        sys_prompt = system_prompt or self.TYPE1_SYSTEM_PROMPT
        locked = self.get_locked_schema_for_tool(tool_response)

        if howhow == "smart":
            target, unlocked_paths, schema_dict = self.create_cascade_schema(locked)
        else:  # dumb
            target, schema_dict = self.get_random_focused_schema(locked)
            unlocked_paths = [target]

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"guided_json": schema_dict},
        )

        hallucinated = json.loads(resp.choices[0].message.content)
        return hallucinated, target, unlocked_paths

    async def type1_api_async(self, client, model: str,
                              tool_response: str, user_prompt: str,
                              howhow: Literal['smart', 'dumb'] = 'smart',
                              temperature: float = 0.8, max_tokens: int = 2048,
                              system_prompt: str | None = None):
        """Async version of ``type1_api``.

        Parameters
        ----------
        client : openai.AsyncOpenAI
            An ``AsyncOpenAI`` client.
        model : str
            Model identifier.
        tool_response : str
            Original tool-response JSON string.
        user_prompt : str
            The original user question.
        howhow : {"smart", "dumb"}, default "smart"
            Masking strategy (see ``type1_api``).
        temperature : float, default 0.8
            Sampling temperature.
        max_tokens : int, default 2048
            Maximum tokens.
        system_prompt : str or None, default None
            Custom system prompt.

        Returns
        -------
        hallucinated_dict : dict
            Corrupted tool response.
        target_path : str
            Dot-path of the chosen leaf.
        unlocked_paths : list of str
            All unlocked dot-paths.
        """
        sys_prompt = system_prompt or self.TYPE1_SYSTEM_PROMPT
        locked = self.get_locked_schema_for_tool(tool_response)

        if howhow == "smart":
            target, unlocked_paths, schema_dict = self.create_cascade_schema(locked)
        else:
            target, schema_dict = self.get_random_focused_schema(locked)
            unlocked_paths = [target]

        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"guided_json": schema_dict},
        )

        hallucinated = json.loads(resp.choices[0].message.content)
        return hallucinated, target, unlocked_paths


    #------------------------------------------------------
    # Type 3: Overgeneration constrained to system tools
    #------------------------------------------------------

    @staticmethod
    def _extract_tool_list_from_system(system_prompt: str) -> str:
        """Extract the JSON tool list from a ToolACE-style system prompt."""
        match = re.search(r'(\[\s*\{.*\}\s*\])', system_prompt, re.DOTALL)
        if match:
            return match.group(1)
        return system_prompt

    @staticmethod
    def _get_used_tool_name(tool_call: str) -> str:
        """Extract the tool name from a tool_call string like 'ToolName(arg=val)'."""
        match = re.match(r'([^(]+)', tool_call)
        if match:
            return match.group(1).strip()
        return tool_call.strip()

    @staticmethod
    def _filter_system_tools(system_prompt: str, exclude_tool: str) -> str:
        """Remove the used tool from the system prompt's tool list.
        Returns the filtered tool list as a JSON string."""
        match = re.search(r'(\[\s*\{.*\}\s*\])', system_prompt, re.DOTALL)
        if not match:
            return system_prompt
        try:
            tools = json.loads(match.group(1))
            filtered = [t for t in tools if t.get("name", "") != exclude_tool]
            if not filtered:
                return match.group(1)  # don't return empty list
            return json.dumps(filtered, indent=2)
        except json.JSONDecodeError:
            return match.group(1)

    def type3_api(
        self,
        client,
        model: str,
        user_prompt: str,
        original_answer: str,
        system_tools: str,
        tool_call: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        system_prompt: str | None = None,
    ) -> str:
        """Type 3 hallucination: subtle overgeneration referencing unused tools.

        Generates one natural-sounding sentence that subtly references a
        capability belonging to one of the OTHER available tools -- without
        naming the tool or explicitly recommending it.

        The tool that was actually used in the dialogue is excluded from the
        prompt so the LLM can only reference capabilities of *other* tools.

        Parameters
        ----------
        client : openai.OpenAI
            An ``openai.OpenAI`` client configured with the vLLM endpoint.
        model : str
            Model identifier, e.g. ``"qwen2.5-3b-instruct"``.
        user_prompt : str
            The original user question.
        original_answer : str
            The original (correct) assistant answer.
        system_tools : str
            The raw ``system`` column from ToolACE containing the JSON tool
            definitions.
        tool_call : str
            The tool-call string, e.g. ``"GetWeather(city='London')"``.
            The tool name is extracted and excluded from the prompt.
        temperature : float, default 0.7
            Sampling temperature.
        max_tokens : int, default 150
            Maximum tokens for the generated sentence.
        system_prompt : str or None, default None
            Custom system prompt with a ``{tools}`` placeholder.
            Falls back to ``TYPE3_SYSTEM_PROMPT``.

        Returns
        -------
        str
            A single sentence of subtle overgeneration.

        See Also
        --------
        type3_api_async : Async variant (requires ``AsyncOpenAI``).

        Examples
        --------
        >>> comment = ha.type3_api(
        ...     client, "model", user_q, answer,
        ...     system_tools=row["system"], tool_call=row["tool_call"],
        ... )
        """
        used_tool = self._get_used_tool_name(tool_call)
        filtered_tools = self._filter_system_tools(system_tools, used_tool)
        base_prompt = system_prompt or self.TYPE3_SYSTEM_PROMPT
        sys_prompt = base_prompt.format(tools=filtered_tools)
        user_content = json.dumps({
            "user_prompt": user_prompt,
            "original_answer": original_answer,
        }, indent=2)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    async def type3_api_async(
        self,
        client,
        model: str,
        user_prompt: str,
        original_answer: str,
        system_tools: str,
        tool_call: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        system_prompt: str | None = None,
    ) -> str:
        """Async version of ``type3_api``.

        Parameters
        ----------
        client : openai.AsyncOpenAI
            An ``AsyncOpenAI`` client.
        model : str
            Model identifier.
        user_prompt : str
            The original user question.
        original_answer : str
            The original (correct) assistant answer.
        system_tools : str
            Raw ``system`` column with tool definitions.
        tool_call : str
            Tool-call string (used tool is excluded).
        temperature : float, default 0.7
            Sampling temperature.
        max_tokens : int, default 150
            Maximum tokens.
        system_prompt : str or None, default None
            Custom system prompt with ``{tools}`` placeholder.

        Returns
        -------
        str
            A single sentence of subtle overgeneration.
        """
        used_tool = self._get_used_tool_name(tool_call)
        filtered_tools = self._filter_system_tools(system_tools, used_tool)
        base_prompt = system_prompt or self.TYPE3_SYSTEM_PROMPT
        sys_prompt = base_prompt.format(tools=filtered_tools)
        user_content = json.dumps({
            "user_prompt": user_prompt,
            "original_answer": original_answer,
        }, indent=2)

        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


    #------------------------------------------------------
    # Showcase / visualisation
    #------------------------------------------------------

    def showcase(
        self,
        client,
        model: str,
        dataset: list[dict],
        n_examples: int = 10,
        types: list[int] | None = None,
        seed: int = 42,
        timeout: float = 120.0,
    ):
        """Generate and display a rich HTML showcase of all hallucination types.

        Produces ``n_examples`` per type with colour-coded cards showing the
        full pipeline: user prompt, tool call, original tool response,
        hallucinated output, with changed/added parts highlighted in red.

        Parameters
        ----------
        client : openai.OpenAI
            An ``openai.OpenAI`` client configured with the vLLM endpoint.
            A generous timeout is recommended (see ``timeout`` parameter).
        model : str
            Model identifier, e.g. ``"qwen2.5-3b-instruct"``.
        dataset : list of dict
            The enriched dataset.  Each row must contain at least:
            ``user_prompt``, ``tool_call``, ``tool_response``,
            ``original_answer``, ``tagged_answer``.
            For Type 3, the ``system`` column (from ToolACE) is also required.
        n_examples : int, default 10
            Number of examples to generate per type.
        types : list of int or None, default None
            Which types to showcase.  ``None`` means all three ``[1, 2, 3]``.
            Pass e.g. ``[2]`` to show only Type 2 examples.
        seed : int, default 42
            Random seed for reproducible sampling.
        timeout : float, default 120.0
            Read timeout in seconds for API calls (Type 1 and 3).
            Increase if the server is slow.

        Returns
        -------
        None
            Displays HTML output directly in the notebook via
            ``IPython.display``.

        Notes
        -----
        - Type 1 and 3 require a live vLLM server.  Type 2 is instant.
        - Cards are colour-coded: red (Type 1), blue (Type 2), green (Type 3).
        - Original vs hallucinated outputs are shown side-by-side.
        - Failed API calls are shown as error cards; oversampling (2x) is used
          internally to compensate for timeouts.

        Examples
        --------
        >>> from openai import OpenAI
        >>> client = OpenAI(base_url="http://host:9091/v1", api_key="key")
        >>> ha = HallucinationAuto()
        >>> ha.showcase(client, "qwen2.5-3b-instruct", dataset, n_examples=5)
        """
        import html as html_module
        from IPython.display import display, HTML

        if types is None:
            types = [1, 2, 3]

        def esc(text):
            return html_module.escape(str(text))

        def fmt_json(obj, max_lines=35):
            s = json.dumps(obj, indent=2, ensure_ascii=False)
            lines = s.split("\n")
            if len(lines) > max_lines:
                return "\n".join(lines[:max_lines]) + f"\n  ... ({len(lines) - max_lines} more lines)"
            return s

        css = """<style>
.sc-card{border:2px solid #d0d0d0;border-radius:10px;margin:18px 0;overflow:hidden;font-family:'SF Mono',Menlo,Consolas,monospace;font-size:13px}
.sc-hdr{padding:10px 14px;font-size:14px;font-weight:700;color:#fff}
.sc-hdr.t1{background:#c0392b}.sc-hdr.t2{background:#2471a3}.sc-hdr.t3{background:#1e8449}
.sc-sec{padding:10px 14px;border-top:1px solid #eee}
.sc-lbl{font-weight:700;color:#666;font-size:11px;text-transform:uppercase;margin-bottom:4px}
.sc-body{white-space:pre-wrap;word-break:break-word;line-height:1.45}
.sc-cols{display:flex;gap:0}.sc-col{flex:1;padding:10px 14px;border-top:1px solid #eee}
.sc-col+.sc-col{border-left:2px solid #eee}
.sc-red{background:#ffcccc;color:#b30000;font-weight:700;padding:1px 4px;border-radius:3px}
.sc-eval{background:#f5f5f5;padding:6px 10px;border-radius:5px;display:inline-block;font-size:12px;margin-top:4px}
.sc-err{border:2px solid #e74c3c;border-radius:10px;margin:18px 0;padding:14px;background:#fdf0ef;color:#c0392b;font-family:monospace}
.sc-orig-lbl{color:#27ae60;font-weight:700}.sc-hall-lbl{color:#c0392b;font-weight:700}
</style>"""
        display(HTML(css))
        oversample = n_examples * 2

        # ── TYPE 1 ──────────────────────────────────────────────
        if 1 in types:
            parts = ['<h2 style="color:#c0392b;border-bottom:3px solid #c0392b;padding-bottom:6px;">TYPE 1: Schema-based JSON corruption</h2>']
            random.seed(seed)
            indices = random.sample(range(len(dataset)), min(oversample, len(dataset)))
            done = 0
            for idx in indices:
                if done >= n_examples:
                    break
                row = dataset[idx]
                try:
                    hall_dict, target, unlocked_paths = self.type1_api(
                        client=client, model=model,
                        tool_response=row["tool_response"],
                        user_prompt=row["user_prompt"],
                        howhow="smart", temperature=0.8,
                    )
                    original_data = json.loads(row["tool_response"])
                    spans, summary = self.evaluate_hallucination(original_data, hall_dict, unlocked_paths)
                    tool_name = original_data.get("name", "?")
                    orig_j = esc(fmt_json(original_data))
                    hall_j = esc(fmt_json(hall_dict))
                    for sp in spans:
                        try:
                            v = esc(str(self.extract_value(hall_dict, sp["path"])))
                            if v in hall_j:
                                hall_j = hall_j.replace(v, '<span class="sc-red">' + v + '</span>', 1)
                        except Exception:
                            pass
                    done += 1
                    parts.append(
                        '<div class="sc-card">'
                        '<div class="sc-hdr t1">#' + str(done) + ' [sample ' + str(idx) + '] -- ' + esc(tool_name) + '</div>'
                        '<div class="sc-sec"><div class="sc-lbl">User Prompt</div><div class="sc-body">' + esc(row["user_prompt"]) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Tool Call</div><div class="sc-body">' + esc(row["tool_call"]) + '</div></div>'
                        '<div class="sc-cols">'
                        '<div class="sc-col"><div class="sc-lbl sc-orig-lbl">Original Tool Response</div><div class="sc-body">' + orig_j + '</div></div>'
                        '<div class="sc-col"><div class="sc-lbl sc-hall-lbl">Hallucinated Tool Response (red = changed)</div><div class="sc-body">' + hall_j + '</div></div>'
                        '</div>'
                        '<div class="sc-sec"><div class="sc-lbl">Original Answer</div><div class="sc-body">' + esc(row["original_answer"]) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Evaluation</div>'
                        '<span class="sc-eval">Target: <b>' + esc(target) + '</b> | ' + esc(str(summary)) + '</span></div>'
                        '</div>'
                    )
                except Exception as e:
                    done += 1
                    parts.append('<div class="sc-err"><b>#' + str(done) + ' [sample ' + str(idx) + ']</b> -- ' + esc(str(e)[:200]) + '</div>')
            display(HTML("".join(parts)))
            print(f"Type 1: {done} examples")

        # ── TYPE 2 ──────────────────────────────────────────────
        if 2 in types:
            parts = ['<h2 style="color:#2471a3;border-bottom:3px solid #2471a3;padding-bottom:6px;">TYPE 2: Deletion-based overgeneration</h2>']
            random.seed(seed + 1)
            indices = random.sample(range(len(dataset)), min(oversample * 2, len(dataset)))
            done = 0
            for idx in indices:
                if done >= n_examples:
                    break
                row = dataset[idx]
                try:
                    reduced_tr, deleted_paths, spans, deletion_target = self.type2_delete(
                        tool_response=row["tool_response"],
                        tagged_answer=row["tagged_answer"],
                    )
                    if not spans:
                        continue
                    original_data = json.loads(row["tool_response"])
                    reduced_data = json.loads(reduced_tr)
                    tool_name = original_data.get("name", "?")
                    orig_j = esc(fmt_json(original_data))
                    red_j = esc(fmt_json(reduced_data))
                    tagged = row["tagged_answer"]
                    disp = tagged
                    del_tags = {sp["tag"] for sp in spans}
                    for tag in del_tags:
                        pat = re.escape("[" + tag + "]") + r"(.*?)" + re.escape("[/" + tag + "]")
                        def _rr(m, t=tag):
                            return '<span class="sc-red" title="references deleted: ' + t + '">' + esc(m.group(1)) + '</span>'
                        disp = re.sub(pat, _rr, disp)
                    disp = re.sub(r'\[/?[^\]]+\]', '', disp)
                    done += 1
                    parts.append(
                        '<div class="sc-card">'
                        '<div class="sc-hdr t2">#' + str(done) + ' [sample ' + str(idx) + '] -- ' + esc(tool_name) + '</div>'
                        '<div class="sc-sec"><div class="sc-lbl">User Prompt</div><div class="sc-body">' + esc(row["user_prompt"]) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Tool Call</div><div class="sc-body">' + esc(row["tool_call"]) + '</div></div>'
                        '<div class="sc-cols">'
                        '<div class="sc-col"><div class="sc-lbl sc-orig-lbl">Original Tool Response</div><div class="sc-body">' + orig_j + '</div></div>'
                        '<div class="sc-col"><div class="sc-lbl sc-hall-lbl">Reduced Tool Response (data removed)</div><div class="sc-body">' + red_j + '</div></div>'
                        '</div>'
                        '<div class="sc-sec"><div class="sc-lbl">Deleted paths</div><div class="sc-body">' + esc(str(deleted_paths)) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Answer (red = references deleted data)</div><div class="sc-body">' + disp + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Hallucination spans</div>'
                        '<span class="sc-eval">' + str(len(spans)) + ' span(s): ' + esc(str([s["tag"] for s in spans])) + '</span></div>'
                        '</div>'
                    )
                except Exception as e:
                    done += 1
                    parts.append('<div class="sc-err"><b>#' + str(done) + ' [sample ' + str(idx) + ']</b> -- ' + esc(str(e)[:200]) + '</div>')
            display(HTML("".join(parts)))
            print(f"Type 2: {done} examples")

        # ── TYPE 3 ──────────────────────────────────────────────
        if 3 in types:
            parts = ['<h2 style="color:#1e8449;border-bottom:3px solid #1e8449;padding-bottom:6px;">TYPE 3: Subtle tool overgeneration</h2>']
            t3_cands = []
            for i, row in enumerate(dataset):
                if not row.get("system"):
                    continue
                m = re.search(r'(\[\s*\{.*\}\s*\])', row["system"], re.DOTALL)
                if not m:
                    continue
                try:
                    tools = json.loads(m.group(1))
                    used = row["tool_call"].split("(")[0].strip()
                    others = [t["name"] for t in tools if t.get("name") != used]
                    if others:
                        t3_cands.append(i)
                except Exception:
                    pass
            random.seed(seed + 2)
            indices = random.sample(t3_cands, min(oversample, len(t3_cands)))
            done = 0
            for idx in indices:
                if done >= n_examples:
                    break
                row = dataset[idx]
                try:
                    comment = self.type3_api(
                        client=client, model=model,
                        user_prompt=row["user_prompt"],
                        original_answer=row["original_answer"].rstrip(),
                        system_tools=row["system"],
                        tool_call=row["tool_call"],
                        temperature=0.7, max_tokens=150,
                    )
                    original_data = json.loads(row["tool_response"])
                    used_tool = row["tool_call"].split("(")[0].strip()
                    m = re.search(r'(\[\s*\{.*\}\s*\])', row["system"], re.DOTALL)
                    other_tools = []
                    if m:
                        t_json = json.loads(m.group(1))
                        other_tools = [t["name"] for t in t_json if t.get("name") != used_tool]
                    done += 1
                    parts.append(
                        '<div class="sc-card">'
                        '<div class="sc-hdr t3">#' + str(done) + ' [sample ' + str(idx) + '] -- Used: ' + esc(used_tool) + '</div>'
                        '<div class="sc-sec"><div class="sc-lbl">User Prompt</div><div class="sc-body">' + esc(row["user_prompt"]) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Tool Call</div><div class="sc-body">' + esc(row["tool_call"]) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Tool Response</div><div class="sc-body">' + esc(fmt_json(original_data)) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Other available tools (NOT used)</div>'
                        '<span class="sc-eval">' + esc(", ".join(other_tools)) + '</span></div>'
                        '<div class="sc-cols">'
                        '<div class="sc-col"><div class="sc-lbl sc-orig-lbl">Original Answer</div><div class="sc-body">' + esc(row["original_answer"].rstrip()) + '</div></div>'
                        '<div class="sc-col"><div class="sc-lbl sc-hall-lbl">Hallucinated Answer (red = overgeneration)</div><div class="sc-body">' + esc(row["original_answer"].rstrip()) + '<br><br><span class="sc-red">' + esc(comment) + '</span></div></div>'
                        '</div>'
                        '</div>'
                    )
                except Exception as e:
                    done += 1
                    parts.append('<div class="sc-err"><b>#' + str(done) + ' [sample ' + str(idx) + ']</b> -- ' + esc(str(e)[:200]) + '</div>')
            display(HTML("".join(parts)))
            print(f"Type 3: {done} examples")

        print("\nShowcase complete.")


    #------------------------------------------------------
    # Full dataset generation (sync)
    #------------------------------------------------------

    def generate_type1_dataset(
        self,
        client,
        model: str,
        dataset: list[dict],
        output_path: str,
        howhow: Literal['smart', 'dumb'] = 'smart',
        temperature: float = 0.8,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> list[dict]:
        """Generate Type 1 hallucinations for an entire dataset (sync).

        Iterates over every row, calls ``type1_api``, evaluates the
        result, and saves incrementally to ``output_path``.  If the output
        file already exists with partial results, generation resumes from
        where it left off.

        Parameters
        ----------
        client : openai.OpenAI
            An ``openai.OpenAI`` client.
        model : str
            Model identifier.
        dataset : list of dict
            The input dataset (list of row dicts with ``tool_response``,
            ``user_prompt``, etc.).
        output_path : str
            File path for the output JSON.  Written after every row.
        howhow : {"smart", "dumb"}, default "smart"
            Masking strategy (see ``type1_api``).
        temperature : float, default 0.8
            Sampling temperature.
        max_tokens : int, default 4096
            Maximum tokens per generation.
        max_retries : int, default 3
            Retries per row on failure.

        Returns
        -------
        list of dict
            The full list of generated rows (each row is the original dict
            augmented with ``hallucinated_tool_response``, ``hallucination_target``,
            ``unlocked_paths``, ``eval_spans``, ``eval_summary``).
        """
        out = Path(output_path)

        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out, "r") as f:
                generated = json.load(f)
        start_idx = len(generated)

        skipped = 0
        for idx in tqdm(range(start_idx, len(dataset)), initial=start_idx, total=len(dataset), desc="Type1"):
            row = dataset[idx]
            success = False
            for attempt in range(max_retries):
                try:
                    hall_dict, target, unlocked_paths = self.type1_api(
                        client=client, model=model,
                        tool_response=row["tool_response"],
                        user_prompt=row["user_prompt"],
                        howhow=howhow,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    original_data = json.loads(row["tool_response"])
                    spans, summary = self.evaluate_hallucination(original_data, hall_dict, unlocked_paths)
                    new_row = {
                        **row,
                        "hallucinated_tool_response": json.dumps(hall_dict, ensure_ascii=False),
                        "hallucination_target": target,
                        "unlocked_paths": unlocked_paths,
                        "hallucination_type": f"type1_{howhow}",
                        "eval_spans": spans,
                        "eval_summary": summary,
                    }
                    generated.append(new_row)
                    success = True
                    break
                except Exception:
                    continue
            if not success:
                skipped += 1

            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)

        print(f"Done: {len(generated)} rows saved to {output_path}, {skipped} skipped")
        return generated

    def generate_type2_dataset(
        self,
        dataset: list[dict],
        output_path: str,
    ) -> list[dict]:
        """Generate Type 2 hallucinations for an entire dataset (no LLM needed).

        Applies ``type2_delete`` to every row, saving incrementally.
        Resumable if interrupted.

        Parameters
        ----------
        dataset : list of dict
            Input dataset with ``tool_response`` and ``tagged_answer`` columns.
        output_path : str
            File path for the output JSON.

        Returns
        -------
        list of dict
            Generated rows augmented with ``reduced_tool_response``,
            ``deleted_paths``, ``deletion_target``, ``hallucination_spans``.
        """
        out = Path(output_path)

        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out, "r") as f:
                generated = json.load(f)
        start_idx = len(generated)

        skipped = 0
        for idx in tqdm(range(start_idx, len(dataset)), initial=start_idx, total=len(dataset), desc="Type2"):
            row = dataset[idx]
            try:
                reduced_tr, deleted_paths, spans, deletion_target = self.type2_delete(
                    tool_response=row["tool_response"],
                    tagged_answer=row["tagged_answer"],
                )
                new_row = {
                    **row,
                    "reduced_tool_response": reduced_tr,
                    "deleted_paths": deleted_paths,
                    "deletion_target": deletion_target if isinstance(deletion_target, list) else [deletion_target],
                    "hallucination_spans": spans,
                    "hallucination_type": "type2_deletion",
                }
                generated.append(new_row)
            except Exception as e:
                print(f"  [type2] Row {idx} error: {e}")
                skipped += 1

            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)

        print(f"Done: {len(generated)} rows saved to {output_path}, {skipped} skipped")
        return generated

    def generate_type3_dataset(
        self,
        client,
        model: str,
        dataset: list[dict],
        output_path: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        max_retries: int = 3,
    ) -> list[dict]:
        """Generate Type 3 hallucinations for an entire dataset (sync).

        Applies ``type3_api`` to every row that has a ``system`` column.
        Saves incrementally and is resumable.

        Parameters
        ----------
        client : openai.OpenAI
            An ``openai.OpenAI`` client.
        model : str
            Model identifier.
        dataset : list of dict
            Input dataset.  Rows must include ``system``, ``tool_call``,
            ``user_prompt``, ``original_answer``.
        output_path : str
            File path for the output JSON.
        temperature : float, default 0.7
            Sampling temperature.
        max_tokens : int, default 150
            Maximum tokens per sentence.
        max_retries : int, default 3
            Retries per row on failure.

        Returns
        -------
        list of dict
            Generated rows augmented with ``overgenerated_answer``,
            ``overgeneration_comment``, ``used_tool``.
        """
        out = Path(output_path)

        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out, "r") as f:
                generated = json.load(f)
        start_idx = len(generated)

        skipped = 0
        for idx in tqdm(range(start_idx, len(dataset)), initial=start_idx, total=len(dataset), desc="Type3"):
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
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if comment:
                        new_row = {
                            **row,
                            "overgenerated_answer": row["original_answer"].rstrip() + "\n\n" + comment,
                            "overgeneration_comment": comment,
                            "hallucination_type": "type3_tool_overgen",
                            "used_tool": self._get_used_tool_name(row["tool_call"]),
                        }
                        generated.append(new_row)
                        success = True
                        break
                except Exception:
                    continue
            if not success:
                skipped += 1

            with open(out, "w") as f:
                json.dump(generated, f, ensure_ascii=False, indent=2)

        print(f"Done: {len(generated)} rows saved to {output_path}, {skipped} skipped")
        return generated

    #------------------------------------------------------
    # Full dataset generation (async / batched)
    #------------------------------------------------------

    async def _type1_single(self, client, model, row, howhow, temperature, max_tokens, max_retries):
        """Process a single row for type1 async generation."""
        for attempt in range(max_retries):
            try:
                hall_dict, target, unlocked_paths = await self.type1_api_async(
                    client=client, model=model,
                    tool_response=row["tool_response"],
                    user_prompt=row["user_prompt"],
                    howhow=howhow,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                original_data = json.loads(row["tool_response"])
                spans, summary = self.evaluate_hallucination(original_data, hall_dict, unlocked_paths)
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

    async def generate_type1_dataset_async(
        self,
        client,
        model: str,
        dataset: list[dict],
        output_path: str,
        howhow: Literal['smart', 'dumb'] = 'smart',
        temperature: float = 0.8,
        max_tokens: int = 4096,
        max_retries: int = 3,
        batch_size: int = 10,
    ) -> list[dict]:
        """Async batched Type 1 generation.

        Processes ``batch_size`` rows concurrently using ``asyncio.gather``.
        Saves after each batch.  Resumable.

        Parameters
        ----------
        client : openai.AsyncOpenAI
            An ``AsyncOpenAI`` client.
        model : str
            Model identifier.
        dataset : list of dict
            Input dataset.
        output_path : str
            Output JSON path.
        howhow : {"smart", "dumb"}, default "smart"
            Masking strategy.
        temperature : float, default 0.8
            Sampling temperature.
        max_tokens : int, default 4096
            Maximum tokens per generation.
        max_retries : int, default 3
            Retries per row.
        batch_size : int, default 10
            Concurrent requests per batch.

        Returns
        -------
        list of dict
            Generated rows.
        """
        out = Path(output_path)

        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out, "r") as f:
                generated = json.load(f)
        start_idx = len(generated)

        remaining = list(range(start_idx, len(dataset)))
        skipped = 0
        pbar = tqdm(total=len(dataset), initial=start_idx, desc="Type1 Async")

        for batch_start in range(0, len(remaining), batch_size):
            batch_indices = remaining[batch_start:batch_start + batch_size]
            tasks = [
                self._type1_single(client, model, dataset[idx], howhow, temperature, max_tokens, max_retries)
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
        """Process a single row for type3 async generation."""
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

    async def generate_type3_dataset_async(
        self,
        client,
        model: str,
        dataset: list[dict],
        output_path: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        max_retries: int = 3,
        batch_size: int = 10,
    ) -> list[dict]:
        """Async batched Type 3 generation.

        Processes ``batch_size`` rows concurrently.  Saves after each batch.
        Resumable.

        Parameters
        ----------
        client : openai.AsyncOpenAI
            An ``AsyncOpenAI`` client.
        model : str
            Model identifier.
        dataset : list of dict
            Input dataset (must include ``system`` column).
        output_path : str
            Output JSON path.
        temperature : float, default 0.7
            Sampling temperature.
        max_tokens : int, default 150
            Maximum tokens per sentence.
        max_retries : int, default 3
            Retries per row.
        batch_size : int, default 10
            Concurrent requests per batch.

        Returns
        -------
        list of dict
            Generated rows.
        """
        out = Path(output_path)

        generated = []
        if out.exists() and out.stat().st_size > 0:
            with open(out, "r") as f:
                generated = json.load(f)
        start_idx = len(generated)

        remaining = list(range(start_idx, len(dataset)))
        skipped = 0
        pbar = tqdm(total=len(dataset), initial=start_idx, desc="Type3 Async")

        for batch_start in range(0, len(remaining), batch_size):
            batch_indices = remaining[batch_start:batch_start + batch_size]
            tasks = [
                self._type3_single(client, model, dataset[idx], temperature, max_tokens, max_retries)
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

    #------------------------------------------------------
    # Dataset export (RAGTruth / JSON)
    #------------------------------------------------------

    _RAGTRUTH_FIELDS = [
        "id", "query", "context", "output", "task_type", "quality",
        "model", "temperature", "hallucination_labels",
        "hallucination_labels_processed", "input_str",
    ]

    _TAG_RE = re.compile(r"\[/?[A-Za-z_][A-Za-z0-9_]*\]")

    @staticmethod
    def _strip_tags(text: str) -> str:
        """Remove [tag]/[/tag] annotation markers from text."""
        return HallucinationAuto._TAG_RE.sub("", text)

    @staticmethod
    def _find_in_output(text: str, clean_output: str) -> tuple:
        """Find *text* in *clean_output* with progressively fuzzier strategies.

        Returns ``(start, end, matched_text)``; ``matched_text`` is the
        substring of *clean_output* that was actually matched (may differ
        in formatting from *text*).
        """
        if not text:
            return -1, -1, text

        # 1. Exact match
        idx = clean_output.find(text)
        if idx != -1:
            return idx, idx + len(text), text

        # 2. Integers with commas  (300000 -> 300,000)
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

        # 3. Floats with commas in integer part (12345.67 -> 12,345.67)
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

        # 4. Case-insensitive search
        lower_output = clean_output.lower()
        lower_text = text.lower()
        idx = lower_output.find(lower_text)
        if idx != -1:
            matched = clean_output[idx:idx + len(text)]
            return idx, idx + len(text), matched

        # 5. ISO dates: 2025-01-15T10:00:00Z -> date part 2025-01-15
        if re.match(r'^\d{4}-\d{2}-\d{2}[T ]', text):
            date_part = text[:10]
            idx = clean_output.find(date_part)
            if idx != -1:
                return idx, idx + len(date_part), date_part

        # 6. Short alphanumeric strings — word-boundary regex
        if 3 <= len(text) <= 40 and text.isalnum():
            m = re.search(re.escape(text), clean_output, re.IGNORECASE)
            if m:
                return m.start(), m.end(), m.group()

        # 7. Short numbers with word boundaries
        if text.isdigit() and len(text) <= 6:
            m = re.search(r'(?<!\d)' + re.escape(text) + r'(?!\d)', clean_output)
            if m:
                return m.start(), m.end(), m.group()

        return -1, -1, text

    @staticmethod
    def _ragtruth_label(
        text: str,
        clean_output: str,
        label_type: str,
        meta: str,
        implicit_true: bool = False,
        due_to_null: bool = False,
    ) -> dict:
        """Build a single RAGTruth-compatible label dict with start/end offsets."""
        start, end, matched_text = HallucinationAuto._find_in_output(text, clean_output)
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
        """Extract RAGTruth labels from Type 1 (schema corruption) eval_spans."""
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
        """Extract RAGTruth labels from Type 2 (deletion) hallucination_spans."""
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
            "evident_conflict": 0, "baseless_info": 0,
            "overgeneration": 1 if comment else 0,
        }

    def _item_to_ragtruth_row(self, item: dict, row_id: int) -> dict:
        """Convert a single generated item into a RAGTruth-compatible row dict.

        Automatically detects the hallucination type and routes to the
        appropriate label extractor.
        """
        h_type = item.get("hallucination_type", "")
        is_overgen = h_type in ("type2_overgen", "type3_tool_overgen")
        is_type1 = h_type.startswith("type1_")

        # Select the correct output text
        if is_overgen:
            clean_output = item.get("overgenerated_answer", item.get("original_answer", ""))
        else:
            clean_output = self._strip_tags(item.get("tagged_answer", item.get("original_answer", "")))

        # Select context and extract labels based on type
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
        system = item.get("system", "")
        parts = []
        if system:
            parts.append(f"System:\n{system}")
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

    def to_dataset(
        self,
        input_data,
        output_format: Literal["json", "ragtruth"] = "ragtruth",
        output_path: Optional[str] = None,
    ):
        """Convert generated hallucination data to a structured dataset.

        Accepts output from any of ``generate_type1_dataset``,
        ``generate_type2_dataset``, or ``generate_type3_dataset``.
        The hallucination type is detected automatically from each item's
        ``hallucination_type`` field.

        Parameters
        ----------
        input_data : str, Path, or list of dict
            Path to a JSON file produced by ``generate_type*_dataset``, or a
            list of dicts already in memory.
        output_format : {"ragtruth", "json"}, default "ragtruth"
            ``"ragtruth"`` writes a CSV matching the ``wandb/RAGTruth-processed``
            schema (start/end offsets, meta, etc.).
            ``"json"`` returns / saves a list of converted row dicts.
        output_path : str or None
            Override the output path.

            * ``"ragtruth"``: defaults to ``<input_stem>_ragtruth.csv``.
            * ``"json"``: if ``None``, returns rows in memory.

        Returns
        -------
        str or list
            Path string if saved to file, or list of row dicts for in-memory
            ``"json"`` mode.
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

        # ragtruth -> CSV
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
