"""
multihop.py — Multistep dialogue hallucination injection + pruning generation.

Defines MultihopMixin.  HallucinationAuto inherits from it.

Two capabilities
-----------------

1. LAST-TURN INJECTION (existing, ported from trash/hallucination_auto.py):
   - detect_type1_spans / detect_type2_spans / detect_type3_spans
   - _apply_hall_tags, _inject_last_turn
   - generate_multistep_type1 / type2 / type3
   These take pre-generated singlehop data and inject hallucinations into
   the LAST completed tool turn of each multistep dialogue.

2. PRUNING-BASED GENERATION (new):
   For a dialogue with N completed tool turns, generate hallucinated samples
   at EVERY prefix depth from N down to 2 (skip 1 = single-hop):

     4-turn dialogue → [turn1, turn2, turn3, hall@turn4]   (depth=4)
                        [turn1, turn2, hall@turn3]           (depth=3)
                        [turn1, hall@turn2]                  (depth=2)
                        single-hop → STOP

   Each sample is a PRUNED copy of the dialogue (truncated at the target turn)
   with a freshly generated hallucination at its last turn.
   LLM calls are made for type1/type3; type2 uses only deletion (no LLM).

Usage (quick-start):
    ha = HallucinationAuto()
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="http://host:8000/v1", api_key="dummy")

    import asyncio, json
    multistep = json.load(open("last dub/toolace_multistep_clean.json"))

    asyncio.run(ha.generate_pruned_multistep(
        client, "Qwen/Qwen2.5-14B-Instruct",
        multistep, output_dir="pruning_gen/results",
    ))
"""

import json
import re
import asyncio
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from tqdm.auto import tqdm


class MultihopMixin:

    # ══════════════════════════════════════════════════════════════════════════
    # TAGGING HELPERS — auto-tag answers with [field]value[/field] markup
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _tag_row(row: dict) -> dict:
        """Return *row* with ``tagged_answer`` produced by matching tool_response
        leaf values into original_answer (longest-match-first, case-insensitive).
        Returns a shallow copy; original dict is not mutated.
        """
        tool_resp = json.loads(row["tool_response"])
        ans = row["original_answer"]

        leaves: list[tuple[str, str, object]] = []

        def _traverse(obj, path: list[str]) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _traverse(v, path + [k])
            elif isinstance(obj, list):
                for v in obj:
                    _traverse(v, path)
            elif obj is not None:
                leaves.append((".".join(path), path[-1] if path else "", obj))

        _traverse(tool_resp, [])

        str_leaves = [
            (fp, kn, str(v).lower() if isinstance(v, bool) else str(v))
            for fp, kn, v in leaves
        ]
        str_leaves.sort(key=lambda x: len(x[2]), reverse=True)

        mask = [False] * len(ans)
        replacements: list[tuple[int, int, str, str]] = []

        for _fp, key_name, val_str in str_leaves:
            if not val_str.strip():
                continue
            start = 0
            while True:
                idx = ans.lower().find(val_str.lower(), start)
                if idx == -1:
                    break
                end = idx + len(val_str)
                if not any(mask[idx:end]):
                    for i in range(idx, end):
                        mask[i] = True
                    replacements.append((idx, end, key_name, ans[idx:end]))
                    break
                start = idx + 1

        replacements.sort(key=lambda x: x[0], reverse=True)
        tagged = ans
        for start, end, key_name, orig_text in replacements:
            tagged = tagged[:start] + f"[{key_name}]{orig_text}[/{key_name}]" + tagged[end:]

        unmatched = [
            fp for fp, _kn, vs in str_leaves
            if vs.strip() and not any(vs.lower() in ans[r[0]:r[1]].lower() for r in replacements)
        ]
        return {
            **row,
            "tagged_answer":  tagged,
            "num_tags":       len(replacements),
            "num_unmatched":  len(unmatched),
            "unmatched_keys": unmatched,
        }

    def tag_dataset(self, dataset: list, output_path=None, overwrite: bool = False) -> list:
        """Tag every row in *dataset* that lacks ``tagged_answer`` (or all if overwrite).

        Each row must have ``tool_response`` and ``original_answer``.
        Optionally saves the result to *output_path*.
        """
        tagged_count, result = 0, []
        for row in tqdm(dataset, desc="Tagging"):
            if overwrite or not row.get("tagged_answer"):
                row = self._tag_row(row)
                tagged_count += 1
            result.append(row)
        print(f"tag_dataset: {tagged_count}/{len(result)} rows tagged")
        if output_path:
            with open(output_path, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  Saved → {output_path}")
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # SPAN DETECTION — find hallucinated char offsets in the answer
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _span_leaves(obj, prefix=""):
        """Yield (dot.path, str_value) for every leaf in a nested object."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from MultihopMixin._span_leaves(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from MultihopMixin._span_leaves(v, f"{prefix}.{i}")
        else:
            yield prefix, str(obj)

    @staticmethod
    def _span_parse_tags(tagged_answer):
        """Strip [key]value[/key] markup; return (tags_list, plain_text).
        Each tag dict: {key, value, start, end} with offsets into plain_text.
        """
        tag_re = re.compile(r"\[(\w+)\](.*?)\[/\1\]", re.DOTALL)
        tags, plain, prev = [], "", 0
        for m in tag_re.finditer(tagged_answer):
            plain += tagged_answer[prev:m.start()]
            s = len(plain)
            key, val = m.group(1), m.group(2)
            plain += val
            tags.append({"key": key, "value": val, "start": s, "end": len(plain)})
            prev = m.end()
        plain += tagged_answer[prev:]
        return tags, plain

    @staticmethod
    def _span_diff(orig_str, hall_str):
        """Diff two tool-response JSON strings.
        Returns (changes dict, orig_leaves list).
        changes: path -> {type: 'changed'|'deleted', orig, hall}
        """
        orig_leaves = list(MultihopMixin._span_leaves(json.loads(orig_str)))
        hall_dict = dict(MultihopMixin._span_leaves(json.loads(hall_str)))
        changes = {}
        for path, val in orig_leaves:
            h = hall_dict.get(path)
            if h is None:
                changes[path] = {"type": "deleted", "orig": val, "hall": None}
            elif h != val:
                changes[path] = {"type": "changed", "orig": val, "hall": h}
        return changes, orig_leaves

    @staticmethod
    def _span_match_tags(tags, orig_leaves):
        """Map each parsed tag to its most-likely JSON leaf path."""
        by_key = defaultdict(list)
        for path, val in orig_leaves:
            by_key[path.rsplit(".", 1)[-1]].append((path, val))
        cursor = defaultdict(int)
        out = []
        for tag in tags:
            k = tag["key"]
            cands = by_key.get(k, [])
            idx, hit = cursor[k], None
            for j in range(idx, len(cands)):
                p, v = cands[j]
                if v == tag["value"] or tag["value"] in v or v in tag["value"]:
                    hit, cursor[k] = p, j + 1
                    break
            if hit is None and idx < len(cands):
                hit, cursor[k] = cands[idx][0], idx + 1
            out.append({**tag, "path": hit})
        return out

    @staticmethod
    def _span_insert(spans, start, end, text, meta, label, implicit, due_to_null):
        """Append a span only if it does not overlap any existing span."""
        if any(s["start"] < end and start < s["end"] for s in spans):
            return False
        spans.append({
            "start": start, "end": end, "text": text, "meta": meta,
            "label_type": label, "implicit_true": implicit, "due_to_null": due_to_null,
        })
        return True

    def detect_type1_spans(self, item: dict) -> tuple:
        """Find char-level hallucinated spans for a Type-1 item.

        Uses 4-tier strategy: tag-based → original text search → corrupted text
        search → implicit (whole answer).

        False-positive filter: a span is only kept if its text does NOT appear
        in the corrupted tool response — if it does, the model was correct to
        quote that value (unchanged field) and marking it is a false positive.

        Returns (spans, plain_text).
        """
        if not item.get("tagged_answer"):
            item = self._tag_row(item)
        changes, orig_leaves = self._span_diff(
            item["tool_response"], item["hallucinated_tool_response"]
        )
        tags, plain = self._span_parse_tags(item["tagged_answer"])
        matched = self._span_match_tags(tags, orig_leaves)
        spans, found = [], set()

        # Build set of all string values in the CORRUPTED tool response
        hall_values = set(
            v.lower() for _, v in self._span_leaves(json.loads(item["hallucinated_tool_response"]))
            if len(v) > 2
        )

        def _in_corrupted(text: str) -> bool:
            """True if text also appears in the corrupted context → not a real conflict."""
            t = text.lower()
            return t in hall_values or any(t in v for v in hall_values)

        for m in matched:
            path = m.get("path")
            if not path or path not in changes:
                continue
            c = changes[path]
            found.add(path)
            label = "Evident Conflict" if c["type"] == "changed" else "Evident Baseless Info"
            meta = f"{label.upper()}\nOriginal: {c['orig']}"
            if c["hall"]:
                meta += f"\nGenerated: {c['hall']}"
            # Skip if span text is unchanged (present in corrupted context too)
            if _in_corrupted(m["value"]):
                continue
            self._span_insert(spans, m["start"], m["end"], m["value"], meta, label, False, False)

        for path, c in changes.items():
            if path in found:
                continue
            label = "Evident Conflict" if c["type"] == "changed" else "Evident Baseless Info"
            meta_base = f"{label.upper()}\nOriginal: {c['orig']}"
            if c["hall"]:
                meta_base += f"\nGenerated: {c['hall']}"
            if c["orig"] and len(c["orig"]) > 1 and not _in_corrupted(c["orig"]):
                idx = plain.find(c["orig"])
                if idx >= 0 and self._span_insert(spans, idx, idx + len(c["orig"]),
                                                   c["orig"], meta_base, label, False, False):
                    found.add(path)
                    continue
            if c["hall"] and len(c["hall"]) > 1 and not _in_corrupted(c["hall"]):
                idx = plain.find(c["hall"])
                if idx >= 0 and self._span_insert(spans, idx, idx + len(c["hall"]),
                                                   c["hall"], meta_base, label, False, False):
                    found.add(path)
                    continue

        unfound = {p: c for p, c in changes.items() if p not in found}
        if unfound and plain.strip():
            parts = [f"Path: {p}\nOriginal: {c['orig']}" + (f"\nGenerated: {c['hall']}"
                     if c["hall"] else "")
                     for p, c in unfound.items()]
            meta = ("IMPLICIT HALLUCINATION\nThe answer relies on corrupted data "
                    "not directly quoted.\n\n" + "\n---\n".join(parts))
            self._span_insert(spans, 0, len(plain), plain, meta, "Evident Conflict", True, True)

        spans.sort(key=lambda s: s["start"])
        return spans, plain

    @staticmethod
    def _expand_to_line(plain: str, start: int, end: int) -> tuple:
        """Expand a span to cover the full line(s) it sits in."""
        s = plain.rfind("\n", 0, start)
        s = 0 if s < 0 else s + 1
        e = plain.find("\n", end)
        e = len(plain) if e < 0 else e
        return s, e

    @staticmethod
    def _find_all(text: str, val: str) -> list:
        """Return all start indices of val in text."""
        idx, positions = 0, []
        while True:
            i = text.find(val, idx)
            if i < 0:
                break
            positions.append(i)
            idx = i + 1
        return positions

    def detect_type2_spans(self, item: dict) -> tuple:
        """Find char-level hallucinated spans for a Type-2 (deletion) item.

        Improvements over v1:
        - Finds ALL occurrences of each deleted value (not just first)
        - Expands each span to the full containing line for better context
        - Falls back through tag-match → text search → implicit

        Returns (spans, plain_text).
        """
        if not item.get("tagged_answer"):
            item = self._tag_row(item)
        tags, plain = self._span_parse_tags(item["tagged_answer"])
        hall_spans = item.get("hallucination_spans", [])
        del_paths  = item.get("deleted_paths", [])
        spans: list = []
        found_paths: set = set()

        tag_map: dict = {}
        for t in tags:
            tag_map.setdefault(t["key"], []).append(t)

        def _add_all_occurrences(val, meta, path):
            """Find every occurrence of val in plain, expand to line, insert."""
            added = False
            for idx in self._find_all(plain, val):
                s, e = self._expand_to_line(plain, idx, idx + len(val))
                text = plain[s:e]
                if self._span_insert(spans, s, e, text, meta, "Evident Baseless Info", False, True):
                    added = True
            return added

        for hs in hall_spans:
            tag_name = hs["tag"]
            orig_val = str(hs["original_value"])
            meta = ("EVIDENT BASELESS INFO\nPath (deleted): " + hs["path"]
                    + "\nOriginal value: " + orig_val)
            # Tag-based match
            hit = None
            for t in tag_map.get(tag_name, []):
                if t["value"] == orig_val or orig_val in t["value"] or t["value"] in orig_val:
                    hit = t
                    break
            if hit is None and tag_map.get(tag_name):
                hit = tag_map[tag_name][0]
            if hit:
                s, e = self._expand_to_line(plain, hit["start"], hit["end"])
                if self._span_insert(spans, s, e, plain[s:e], meta, "Evident Baseless Info", False, True):
                    found_paths.add(hs["path"])
                continue
            # Text search — all occurrences
            if len(orig_val) > 1 and _add_all_occurrences(orig_val, meta, hs["path"]):
                found_paths.add(hs["path"])

        if del_paths:
            orig_data = json.loads(item["tool_response"])
            for path in del_paths:
                if path in found_paths:
                    continue
                try:
                    val = str(self.extract_value(orig_data, path))
                except Exception:
                    continue
                if len(val) <= 1:
                    continue
                meta = ("EVIDENT BASELESS INFO\nPath (deleted): " + path
                        + "\nOriginal value: " + val)
                if _add_all_occurrences(val, meta, path):
                    found_paths.add(path)

        if not spans and del_paths and plain.strip():
            meta = ("IMPLICIT HALLUCINATION\nAnswer references data deleted from tool_response.\n\n"
                    + "\n".join("Deleted: " + p for p in del_paths))
            self._span_insert(spans, 0, len(plain), plain, meta, "Evident Baseless Info", True, True)

        spans.sort(key=lambda s: s["start"])

        # Merge adjacent spans (same deleted block, consecutive lines)
        if len(spans) > 1:
            merged = [spans[0]]
            for sp in spans[1:]:
                prev = merged[-1]
                gap = plain[prev["end"]:sp["start"]]
                if not sp.get("implicit_true") and not prev.get("implicit_true") and gap.strip() == "":
                    prev["end"]  = sp["end"]
                    prev["text"] = plain[prev["start"]:prev["end"]]
                else:
                    merged.append(sp)
            spans = merged

        return spans, plain

    def detect_type3_spans(self, item: dict) -> tuple:
        """Find the hallucinated span for a Type-3 (overgeneration) item.

        The structure is always: overgenerated_answer = original_answer + "\\n\\n" + comment
        so the span is always at the end — no text search needed.
        Returns (spans, plain_text).
        """
        overgen = item["overgenerated_answer"]
        comment = item["overgeneration_comment"]
        used = item.get("used_tool", "?")
        if not comment:
            return [], overgen
        end = len(overgen)
        start = end - len(comment)
        if start < 0 or overgen[start:] != comment:
            # Fallback: last occurrence search (handles minor whitespace drift)
            idx = overgen.rfind(comment)
            if idx < 0:
                return [], overgen
            start, end = idx, idx + len(comment)
        meta = ("EVIDENT BASELESS INFO\n"
                "Overgenerated sentence referencing tool(s) not used in this dialogue.\n"
                "Used tool: " + str(used))
        return [{
            "start": start, "end": end, "text": comment,
            "meta": meta, "label_type": "Evident Baseless Info",
            "implicit_true": False, "due_to_null": False,
        }], overgen

    # ══════════════════════════════════════════════════════════════════════════
    # LAST-TURN INJECTION — inject pre-generated singlehop data into multistep
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _apply_hall_tags(plain: str, labels: list) -> str:
        """Insert <hall>…</hall> tags right-to-left so char offsets stay valid."""
        if not labels:
            return plain
        for span in sorted(labels, key=lambda s: s["start"], reverse=True):
            s, e = span["start"], span["end"]
            plain = plain[:s] + "<hall>" + plain[s:e] + "</hall>" + plain[e:]
        return plain

    def _inject_last_turn(self, multistep: list, lookup: dict, hall_type: str) -> tuple:
        """Core injection: deep-copy multistep and corrupt only the last tool turn.

        Parameters
        ----------
        multistep : list of dialogue dicts
        lookup    : {user_prompt: hallucinated_row_dict}
        hall_type : "type1" | "type2" | "type3"

        Returns
        -------
        (injected_dataset, log)
        """
        dataset, log = deepcopy(multistep), []

        for dlg_idx, item in enumerate(dataset):
            convs = item["conversations"]
            ct = item["analysis"]["completed_tool_turns"][-1]
            ui = ct["user_index"]
            ti = ct["tool_result_indices"][-1]
            ai = ct["assistant_answer_index"]
            user_text = convs[ui]["value"]

            if user_text not in lookup:
                item["hallucination_type"] = None
                continue

            row = lookup[user_text]

            if hall_type == "type1":
                hall_tool = row["hallucinated_tool_response"]
                labels = row.get("hallucination_labels", [])
                plain_answer = row.get("plain_answer", convs[ai]["value"])
            elif hall_type == "type2":
                hall_tool = row["reduced_tool_response"]
                labels, plain_answer = self.detect_type2_spans(row)
            else:  # type3
                hall_tool = None
                labels, plain_answer = self.detect_type3_spans(row)

            if hall_tool and hall_type in ("type1", "type2"):
                try:
                    ho = json.loads(hall_tool)
                    tl = json.loads(convs[ti]["value"])
                    for k, entry in enumerate(tl):
                        if entry.get("name") == ho.get("name"):
                            tl[k] = ho
                            break
                    else:
                        tl[0] = ho
                    convs[ti]["value"] = json.dumps(tl, ensure_ascii=False)
                    convs[ti]["hallucination_type"] = hall_type
                    convs[ti]["hallucination_target"] = row.get(
                        "hallucination_target", row.get("deletion_target", ""))
                except Exception as exc:
                    print(f"  [WARN] dlg={dlg_idx} tool-replace: {exc}")

            tagged = self._apply_hall_tags(plain_answer, labels)
            skipped = len(labels) == 0

            convs[ai]["value"] = tagged
            convs[ai]["hallucination_type"] = hall_type
            item["hallucination_type"] = hall_type
            item["skipped"] = skipped

            log.append({
                "dialogue_idx": dlg_idx,
                "tool_turn_idx": ti,
                "ass_turn_idx": ai,
                "user_prompt": user_text[:80],
                "hall_type": hall_type,
                "n_spans": len(labels),
                "has_tags": "<hall>" in tagged,
                "skipped": skipped,
            })

        return dataset, log

    def generate_multistep_type1(self, multistep: list, type1_labelled: list,
                                  output_path: str) -> list:
        """Inject Type-1 hallucinations into multistep dialogues (last turn).

        type1_labelled rows must have: hallucinated_tool_response,
        hallucination_labels, plain_answer (from label_type1_dataset).
        """
        lookup = {r["user_prompt"]: r for r in type1_labelled
                  if r.get("user_prompt") and r.get("hallucinated_tool_response")
                  and r.get("hallucination_labels")}   # drop no-op rows from source
        dataset, log = self._inject_last_turn(multistep, lookup, "type1")
        # Drop dialogues where guided decoding changed nothing
        before = len(dataset)
        dataset = [d for d in dataset if not d.get("skipped")]
        dropped = before - len(dataset)
        with open(output_path, "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        tagged = sum(e["has_tags"] for e in log)
        print(f"Type1 multistep: {len(log)}/{len(multistep)} injected, "
              f"{tagged} with <hall> tags, {dropped} no-op dropped → {output_path}")
        csv = self._save_ragtruth_alongside(output_path)
        if csv:
            print(f"      RAGTruth JSONL: {csv}")
        return dataset

    def generate_multistep_type2(self, multistep: list, type2_data: list,
                                  output_path: str) -> list:
        """Inject Type-2 hallucinations into multistep dialogues (last turn).

        Spans computed on-the-fly via detect_type2_spans.
        """
        lookup = {r["user_prompt"]: r for r in type2_data
                  if r.get("user_prompt") and r.get("reduced_tool_response")}
        dataset, log = self._inject_last_turn(multistep, lookup, "type2")
        with open(output_path, "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        tagged = sum(e["has_tags"] for e in log)
        print(f"Type2 multistep: {len(log)}/{len(multistep)} injected, "
              f"{tagged} with <hall> tags → {output_path}")
        csv = self._save_ragtruth_alongside(output_path)
        if csv:
            print(f"      RAGTruth JSONL: {csv}")
        return dataset

    def generate_multistep_type3(self, multistep: list, type3_data: list,
                                  output_path: str) -> list:
        """Inject Type-3 hallucinations into multistep dialogues (last turn).

        Tool turn is NOT changed; only the assistant answer is replaced and tagged.
        """
        lookup = {r["user_prompt"]: r for r in type3_data
                  if r.get("user_prompt") and r.get("overgenerated_answer")}
        dataset, log = self._inject_last_turn(multistep, lookup, "type3")
        with open(output_path, "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        tagged = sum(e["has_tags"] for e in log)
        print(f"Type3 multistep: {len(log)}/{len(multistep)} injected, "
              f"{tagged} with <hall> tags → {output_path}")
        csv = self._save_ragtruth_alongside(output_path)
        if csv:
            print(f"      RAGTruth JSONL: {csv}")
        return dataset

    # ══════════════════════════════════════════════════════════════════════════
    # PRUNING — new methods for prefix-based generation
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def extract_turn_as_row(dialogue: dict, turn_k: int) -> dict:
        """Extract turn *turn_k* (0-indexed) from *dialogue* as a singlehop row.

        The tool_response is unwrapped from its JSON array to a single object
        so the singlehop APIs (type1_api, type2_delete, type3_api) can handle it.

        Returns a dict with: user_prompt, tool_response, tool_call,
        original_answer, system.
        """
        ct = dialogue["analysis"]["completed_tool_turns"][turn_k]
        convs = dialogue["conversations"]
        # Tool response in multistep is a JSON array: [{name, results}]
        # Unwrap to single object for singlehop compatibility
        tool_raw = json.loads(convs[ct["tool_result_indices"][-1]]["value"])
        tool_obj = tool_raw[0] if isinstance(tool_raw, list) else tool_raw
        return {
            "user_prompt":    convs[ct["user_index"]]["value"],
            "tool_response":  json.dumps(tool_obj, ensure_ascii=False),
            "tool_call":      convs[ct["assistant_tool_call_indices"][0]]["value"],
            "original_answer": convs[ct["assistant_answer_index"]]["value"],
            "system":         dialogue.get("system", ""),
        }

    @staticmethod
    def prune_dialogue(dialogue: dict, target_turn_k: int) -> dict:
        """Return a deep copy of *dialogue* truncated at *target_turn_k* (0-indexed).

        Keeps conversations[0 : assistant_answer_index + 1] for that turn.
        Updates analysis.completed_tool_turns and completed_tool_turn_count.
        """
        ct = dialogue["analysis"]["completed_tool_turns"][target_turn_k]
        cutoff = ct["assistant_answer_index"] + 1
        new_dlg = deepcopy(dialogue)
        new_dlg["conversations"] = new_dlg["conversations"][:cutoff]
        new_dlg["analysis"]["completed_tool_turns"] = (
            new_dlg["analysis"]["completed_tool_turns"][:target_turn_k + 1]
        )
        new_dlg["analysis"]["completed_tool_turn_count"] = target_turn_k + 1
        new_dlg["analysis"]["last_assistant_answer_index"] = ct["assistant_answer_index"]
        return new_dlg

    def inject_row_into_pruned(self, pruned_dialogue: dict, hall_row: dict,
                                hall_type: str) -> dict:
        """Inject *hall_row*'s hallucination into the last turn of *pruned_dialogue*.

        Wraps _inject_last_turn with a single-item lookup.
        Returns the single injected dialogue dict.
        """
        lookup = {hall_row["user_prompt"]: hall_row}
        injected_list, _ = self._inject_last_turn([pruned_dialogue], lookup, hall_type)
        return injected_list[0]

    # ── Per-row generation helpers ────────────────────────────────────────────

    async def generate_type1_for_row(self, client, model: str, row: dict,
                                      max_retries: int = 3) -> dict | None:
        """Generate Type-1 hallucination for a single singlehop row (async).

        Calls type1_api (from SinglehopMixin) with retries.
        Returns the row enriched with hallucinated_tool_response,
        hallucination_labels, plain_answer; or None on failure.
        """
        try:
            json.loads(row["tool_response"])
        except (json.JSONDecodeError, KeyError):
            return None

        for attempt in range(max_retries):
            try:
                hall_dict, target, unlocked_paths = await self.type1_api_async(
                    client=client, model=model,
                    tool_response=row["tool_response"],
                    user_prompt=row["user_prompt"],
                    howhow="smart",
                )
                original_data = json.loads(row["tool_response"])
                spans, summary = self.evaluate_hallucination(
                    original_data, hall_dict, unlocked_paths
                )
                if not spans:
                    # LLM returned same values — no real hallucination, retry
                    continue
                enriched = {
                    **row,
                    "hallucinated_tool_response": json.dumps(hall_dict, ensure_ascii=False),
                    "hallucination_target": target,
                    "unlocked_paths": unlocked_paths,
                    "hallucination_type": "type1_smart",
                    "eval_spans": spans,
                    "eval_summary": summary,
                }
                spans_labelled, plain = self.detect_type1_spans(enriched)
                if self._is_whole_answer_span(spans_labelled, plain):
                    continue  # string-leaf noise — retry won't help, but keeps loop clean
                enriched["hallucination_labels"] = spans_labelled
                enriched["plain_answer"] = plain
                return enriched
            except Exception as _e:
                if attempt == 0:
                    print(f"  [type1 row] attempt {attempt} error: {type(_e).__name__}: {_e}")
        return None

    def generate_type2_for_row(self, row: dict) -> dict | None:
        """Generate Type-2 hallucination for a single singlehop row (sync, no LLM).

        Auto-tags the answer if needed, then calls type2_delete (from SinglehopMixin).
        Returns enriched row or None on failure.
        """
        try:
            if not row.get("tagged_answer"):
                row = self._tag_row(row)
            reduced_tr, deleted_paths, spans, deletion_target = self.type2_delete(
                tool_response=row["tool_response"],
                tagged_answer=row["tagged_answer"],
            )
            enriched = {
                **row,
                "reduced_tool_response": reduced_tr,
                "deleted_paths": deleted_paths,
                "deletion_target": (deletion_target if isinstance(deletion_target, list)
                                    else [deletion_target]),
                "hallucination_spans": spans,
                "hallucination_type": "type2_deletion",
            }
            return enriched
        except Exception as _e:
            print(f"  [type2 row] error: {type(_e).__name__}: {_e}")
            return None

    async def generate_type3_for_row(self, client, model: str, row: dict,
                                      max_retries: int = 3) -> dict | None:
        """Generate Type-3 hallucination for a single singlehop row (async).

        Calls type3_api_async (from SinglehopMixin) with retries.
        If the row has no 'system' field, one is synthesised automatically
        from the row's own tool_call before calling the API.
        The LLM must output ONLY the new sentence (prompt enforces this).
        Returns enriched row or None on failure / empty extension.
        """
        if not row.get("system"):
            row = self.add_missing_system([row])[0]
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
                )
                comment = comment.strip()
                if not comment:
                    continue
                enriched = {
                    **row,
                    "overgenerated_answer": original_answer + "\n\n" + comment,
                    "overgeneration_comment": comment,
                    "hallucination_type": "type3_tool_overgen",
                    "used_tool": self._get_used_tool_name(row["tool_call"]),
                }
                return enriched
            except Exception as _e:
                if attempt == 0:
                    print(f"  [type3 row] attempt {attempt} error: {type(_e).__name__}: {_e}")
        return None

    # ── Main pruning orchestration ────────────────────────────────────────────

    async def generate_pruned_multistep(
        self,
        client,
        model: str,
        multistep: list,
        output_dir: str,
        batch_size: int = 5,
        types: list = None,
    ) -> tuple:
        """Pruning-based generation: hallucinated multistep samples at every depth.

        For each dialogue with N completed tool turns, generates samples at
        depths N, N-1, ..., 2 (stops before depth 1 = single-hop).

        Each sample is a PRUNED dialogue (truncated to that depth) with a
        freshly generated hallucination at its last tool turn.

        Parameters
        ----------
        client      : AsyncOpenAI client
        model       : model identifier
        multistep   : list of multistep dialogue dicts
        output_dir  : directory to save pruned_type1/2/3.json
        batch_size  : concurrent async requests per batch (type1 and type3)

        Returns
        -------
        (all_type1, all_type2, all_type3) — lists of injected dialogue dicts

        Output files saved to output_dir:
          pruned_type1.json, pruned_type2.json, pruned_type3.json
        """
        types = [str(t) for t in (types or ["1", "2", "3"])]
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        all_type1: list = []
        all_type2: list = []
        all_type3: list = []

        # Load any existing partial results (resumable)
        for lst, fname in [(all_type1, "pruned_type1.json"),
                           (all_type2, "pruned_type2.json"),
                           (all_type3, "pruned_type3.json")]:
            p = out / fname
            if p.exists() and p.stat().st_size > 0:
                try:
                    with open(p) as f:
                        lst.extend(json.load(f))
                    print(f"  Resumed {fname}: {len(lst)} existing rows")
                except Exception:
                    pass

        # Build the list of (dialogue, turn_k) work items
        # turn_k is 0-indexed; we go from n-1 down to 1 (skip 0 = single-hop)
        work_items = []
        for dlg in multistep:
            n = dlg["analysis"]["completed_tool_turn_count"]
            for turn_k in range(n - 1, 0, -1):
                work_items.append((dlg, turn_k))

        print(f"\nPruning: {len(multistep)} dialogues → {len(work_items)} (dialogue, turn) pairs")
        print(f"  (each pair generates up to 3 samples: type1, type2, type3)\n")

        # Process in batches for async type1/type3
        processed = 0
        for batch_start in range(0, len(work_items), batch_size):
            batch = work_items[batch_start:batch_start + batch_size]

            # ── Type 2: synchronous, no LLM ──────────────────────────────────
            if "2" in types:
                for dlg, turn_k in batch:
                    row = self.extract_turn_as_row(dlg, turn_k)
                    pruned = self.prune_dialogue(dlg, turn_k)
                    t2_row = self.generate_type2_for_row(row)
                    if t2_row:
                        injected = self.inject_row_into_pruned(pruned, t2_row, "type2")
                        injected["_pruning_depth"] = turn_k + 1
                        injected["_dialogue_id"] = dlg["analysis"].get("dialogue_id", "")
                        all_type2.append(injected)

            # ── Type 1: async, LLM ───────────────────────────────────────────
            if "1" in types:
                t1_tasks = [
                    self.generate_type1_for_row(client, model, self.extract_turn_as_row(dlg, turn_k))
                    for dlg, turn_k in batch
                ]
                t1_results = await asyncio.gather(*t1_tasks)
                for (dlg, turn_k), t1_row in zip(batch, t1_results):
                    if t1_row:
                        pruned = self.prune_dialogue(dlg, turn_k)
                        injected = self.inject_row_into_pruned(pruned, t1_row, "type1")
                        injected["_pruning_depth"] = turn_k + 1
                        injected["_dialogue_id"] = dlg["analysis"].get("dialogue_id", "")
                        all_type1.append(injected)

            # ── Type 3: async, LLM ───────────────────────────────────────────
            if "3" in types:
                t3_tasks = [
                    self.generate_type3_for_row(client, model, self.extract_turn_as_row(dlg, turn_k))
                    for dlg, turn_k in batch
                ]
                t3_results = await asyncio.gather(*t3_tasks)
                for (dlg, turn_k), t3_row in zip(batch, t3_results):
                    if t3_row:
                        pruned = self.prune_dialogue(dlg, turn_k)
                        injected = self.inject_row_into_pruned(pruned, t3_row, "type3")
                        injected["_pruning_depth"] = turn_k + 1
                        injected["_dialogue_id"] = dlg["analysis"].get("dialogue_id", "")
                        all_type3.append(injected)

            # Save after every batch (incremental / resumable)
            processed += len(batch)
            for lst, fname, t in [(all_type1, "pruned_type1.json", "1"),
                                   (all_type2, "pruned_type2.json", "2"),
                                   (all_type3, "pruned_type3.json", "3")]:
                if t in types:
                    with open(out / fname, "w") as f:
                        json.dump(lst, f, ensure_ascii=False, indent=2)

            print(f"  Batch {batch_start // batch_size + 1}: "
                  f"{processed}/{len(work_items)} pairs done | "
                  f"type1={len(all_type1)} type2={len(all_type2)} type3={len(all_type3)}")

        print(f"\nPruning complete:")
        print(f"  type1 → {len(all_type1)} dialogues ({out/'pruned_type1.json'})")
        print(f"  type2 → {len(all_type2)} dialogues ({out/'pruned_type2.json'})")
        print(f"  type3 → {len(all_type3)} dialogues ({out/'pruned_type3.json'})")

        # Also save RAGTruth CSVs alongside each JSON
        for fname in ("pruned_type1.json", "pruned_type2.json", "pruned_type3.json"):
            csv = self._save_ragtruth_alongside(str(out / fname))
            if csv:
                print(f"  RAGTruth JSONL: {csv}")

        return all_type1, all_type2, all_type3
