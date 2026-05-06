"""
schema.py — JSON Schema tools (the backbone of Type 1 hallucination).

Defines SchemaMixin.  HallucinationAuto inherits from it.

Big picture
-----------
Type 1 works by "guided decoding": we build a JSON schema where some fields
are LOCKED (forced to their original value via `enum`) and some are UNLOCKED
(the LLM is free to hallucinate those).

The two masking strategies decide WHICH fields to unlock:
  - CASCADE (smart): group-aware — unlocks sibling fields or a whole array element
  - FOCUSED  (dumb): subtree-based — unlocks an entire random subtree

After generation, evaluate_hallucination compares original vs generated values
and labels each change as "Evident Conflict" or "Baseless Info".
"""

import json
import math
import random
from copy import deepcopy


class SchemaMixin:

    # Field names that look numeric but must stay as strings
    _KEEP_AS_STRING_KEYS = {
        "id", "zip", "zip_code", "postal_code", "postal", "code",
        "phone", "phone_number", "fax", "ssn", "isbn", "ean", "upc",
    }

    # ── Build / lock ──────────────────────────────────────────────────────────

    def build_locked_schema(self, data, is_root=True, field_name: str = ""):
        """Build a fully-locked JSON schema from actual data.

        Every leaf is pinned via `enum` so guided decoding reproduces the
        exact original JSON.  Arrays use `prefixItems` (one schema per element).
        """
        if isinstance(data, dict):
            properties = {
                key: self.build_locked_schema(value, is_root=False, field_name=key)
                for key, value in data.items()
            }
            return {
                "type": "object",
                "properties": properties,
                "required": list(data.keys()),
                "additionalProperties": False,
            }
        elif isinstance(data, list):
            prefix_items = [
                self.build_locked_schema(item, is_root=False, field_name=field_name)
                for item in data
            ]
            return {
                "type": "array",
                "prefixItems": prefix_items,
                "minItems": len(data),
                "maxItems": len(data),
            }
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
        """Parse tool_call_json_str and return its fully-locked schema."""
        return self.build_locked_schema(json.loads(tool_call_json_str), is_root=True)

    # ── Schema inspection ─────────────────────────────────────────────────────

    def print_schema_tree(self, schema, name="root", indent="", defs=None,
                          is_last=True, current_path=""):
        """Print a human-readable tree of the schema (for debugging)."""
        if defs is None:
            defs = schema.get("$defs", {})
        if "$ref" in schema:
            schema = defs[schema["$ref"].split("/")[-1]]
        marker = "└── " if is_last else "├── "
        node_type = schema.get("type", "object")
        locked = " [LOCKED]" if "enum" in schema else ""
        path_display = f"  [Path: {current_path}]" if current_path else ""
        print(f"{indent}{marker}{name} ({node_type}){locked}{path_display}")
        indent += "    " if is_last else "│   "
        if node_type == "object" and "properties" in schema:
            keys = list(schema["properties"].keys())
            for i, k in enumerate(keys):
                next_path = f"{current_path}.{k}" if current_path else k
                self.print_schema_tree(schema["properties"][k], k, indent, defs,
                                       i == len(keys) - 1, next_path)
        elif node_type == "array" and "prefixItems" in schema:
            for i, item_schema in enumerate(schema["prefixItems"]):
                next_path = f"{current_path}.{i}" if current_path else str(i)
                self.print_schema_tree(item_schema, f"[{i}]", indent, defs,
                                       i == len(schema["prefixItems"]) - 1, next_path)
        elif node_type == "array" and "items" in schema:
            self.print_schema_tree(schema["items"], "items", indent, defs, True,
                                   f"{current_path}.items" if current_path else "items")

    def get_all_leaf_paths(self, schema: dict, current_path: str = "") -> list:
        """Return dot-notation paths to every leaf (terminal value) in the schema.

        Example: ["name", "results.temp", "results.items.0.title"]
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
            paths.extend(self.get_all_leaf_paths(
                schema["items"],
                f"{current_path}.items" if current_path else "items",
            ))
        else:
            paths.append(current_path)
        return paths

    def get_all_internal_paths(self, schema: dict, current_path: str = "") -> list:
        """Return dot-notation paths to every non-leaf (object/array) node.

        Used by the FOCUSED masking strategy to pick a subtree to unlock.
        """
        paths = []
        node_type = schema.get("type", "object")
        if node_type == "object" and "properties" in schema:
            if current_path:
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

    # ── Group / sibling helpers ───────────────────────────────────────────────

    def find_group(self, schema: dict, leaf_path: str):
        """Return the innermost array-element path that contains leaf_path, or None.

        E.g. find_group(schema, 'results.items.0.title') → 'results.items.0'
             find_group(schema, 'results.temp')           → None  (flat, no array)
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
        """('results.city') → ('results', 'city')"""
        parts = leaf_path.split(".")
        return ("", parts[0]) if len(parts) == 1 else (".".join(parts[:-1]), parts[-1])

    def _get_siblings(self, schema, leaf_path):
        """Return (parent_path, sibling_keys) for a leaf node."""
        parent_path, leaf_key = self._find_parent_path(leaf_path)
        parent_node = (schema if not parent_path
                       else self._get_node_at_path(schema, parent_path.split(".")))
        if parent_node.get("type") != "object" or "properties" not in parent_node:
            return parent_path, [leaf_key]
        all_keys = list(parent_node["properties"].keys())
        # Root-level "name" is always locked — skip it from flat siblings
        siblings = [k for k in all_keys if k != "name"] if not parent_path else all_keys
        return parent_path, siblings

    # ── Unlock operations ─────────────────────────────────────────────────────

    @staticmethod
    def _free_all(node):
        """Recursively strip all enum/minItems/maxItems constraints from a subtree.
        Types are preserved exactly as in the locked schema.
        """
        n = deepcopy(node)
        t = n.get("type", "")
        if t == "object" and "properties" in n:
            n["properties"] = {k: SchemaMixin._free_all(v) for k, v in n["properties"].items()}
        elif t == "array":
            if "prefixItems" in n:
                n["prefixItems"] = [SchemaMixin._free_all(s) for s in n["prefixItems"]]
            if "items" in n:
                n["items"] = SchemaMixin._free_all(n["items"])
            n.pop("minItems", None)
            n.pop("maxItems", None)
        else:
            n.pop("enum", None)
        return n

    @staticmethod
    def _apply_at_path(schema, path_parts, fn):
        """Navigate schema along path_parts and apply fn at the target node.
        Mutates in-place — caller should deepcopy first.
        """
        if not path_parts:
            return fn(schema)
        key, rest = path_parts[0], path_parts[1:]
        if schema.get("type") == "array" and "prefixItems" in schema:
            idx = int(key)
            schema["prefixItems"][idx] = SchemaMixin._apply_at_path(
                schema["prefixItems"][idx], rest, fn
            )
        elif schema.get("type") == "object" and key in schema.get("properties", {}):
            schema["properties"][key] = SchemaMixin._apply_at_path(
                schema["properties"][key], rest, fn
            )
        return schema

    # ── Masking strategies ────────────────────────────────────────────────────

    def create_cascade_schema(self, locked_schema: dict) -> tuple:
        """CASCADE (smart) masking — unlock a contextually coherent cluster.

        Algorithm:
          1. Pick a random leaf (depth-weighted, skip root "name").
          2a. If the leaf is inside an array element → unlock the WHOLE element
              (so the LLM hallucinates a coherent row of data).
          2b. Otherwise → unlock 2..ceil(N/2) sibling fields under the same parent.

        Returns:
          (chosen_leaf, unlocked_paths, schema_copy)
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
            # Array element case: unlock the full element
            self._apply_at_path(schema, group.split("."), self._free_all)
            unlocked_paths = [p for p in all_paths if p.startswith(group + ".") or p == group]
        else:
            # Flat case: unlock siblings
            parent_path, siblings = self._get_siblings(locked_schema, chosen)
            _, chosen_key = self._find_parent_path(chosen)
            if len(siblings) <= 3:
                keys_to_unlock = siblings
            else:
                max_unlock = max(2, math.ceil(len(siblings) / 2))
                n_unlock = random.randint(2, max_unlock)
                extras = random.sample(
                    [k for k in siblings if k != chosen_key],
                    min(n_unlock - 1, len(siblings) - 1),
                )
                keys_to_unlock = [chosen_key] + extras
            unlocked_paths = []
            for key in keys_to_unlock:
                full_path = f"{parent_path}.{key}" if parent_path else key
                self._apply_at_path(schema, full_path.split("."), self._free_all)
                unlocked_paths.append(full_path)

        return chosen, unlocked_paths, schema

    def get_random_focused_schema(self, locked_schema: dict):
        """FOCUSED (dumb) masking — unlock a random internal subtree entirely.

        Returns:
          (target_path, schema_copy)
        """
        internal = self.get_all_internal_paths(locked_schema)
        candidates = [p for p in internal if p != "name"] or internal
        if not candidates:
            fields = list(locked_schema["properties"].keys())
            candidates = ([f for f in fields if "enum" not in locked_schema["properties"][f]]
                          or fields)
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

    # ── Misc helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def extract_value(obj, path):
        """Extract a value from a nested dict/list using a dot-path.

        E.g. extract_value(data, "results.items.0.price") → 9.99
        Numeric path segments are treated as list indices.
        """
        for part in path.split("."):
            obj = obj[int(part)] if isinstance(obj, list) else obj[part]
        return obj

    @staticmethod
    def get_tool_names(tool_response_str: str) -> list:
        """Return tool/function names from a tool-response JSON string."""
        parsed = json.loads(tool_response_str)
        if isinstance(parsed, dict):
            return [parsed["name"]] if "name" in parsed else []
        return [item["name"] for item in parsed if isinstance(item, dict) and "name" in item]

    @staticmethod
    def evaluate_hallucination(original_data, hallucinated_data, unlocked_paths: list):
        """Compare original vs hallucinated tool responses and label each change.

        Labels:
          "Evident Conflict" — a real value was replaced with a different value
          "Baseless Info"    — a value was fabricated from None / null

        Returns:
          (spans, summary)
          spans   — list of dicts with path, text, meta, label_type, ...
          summary — {"evident_conflict": int, "baseless_info": int}
        """
        if isinstance(original_data, str):
            original_data = json.loads(original_data)
        if isinstance(hallucinated_data, str):
            hallucinated_data = json.loads(hallucinated_data)

        spans, evident_conflict, baseless_info = [], 0, 0

        for path in unlocked_paths:
            try:
                orig_val = SchemaMixin.extract_value(original_data, path)
            except (KeyError, IndexError, TypeError):
                orig_val = None
            try:
                hall_val = SchemaMixin.extract_value(hallucinated_data, path)
            except (KeyError, IndexError, TypeError):
                hall_val = None

            if orig_val == hall_val:
                continue

            due_to_null = orig_val is None
            if orig_val is not None and hall_val is not None:
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

            spans.append({
                "path": path,
                "text": f"{path}: {hall_val}",
                "meta": f"{label_type}\nOriginal: {orig_val}\nGenerated: {hall_val}",
                "label_type": label_type,
                "implicit_true": False,
                "due_to_null": due_to_null,
            })

        return spans, {"evident_conflict": evident_conflict, "baseless_info": baseless_info}
