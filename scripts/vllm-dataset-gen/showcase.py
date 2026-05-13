"""
showcase.py — Rich HTML visualisation for Jupyter notebooks.

Defines ShowcaseMixin.  HallucinationAuto inherits from it.

Usage (in a Jupyter cell):
    ha = HallucinationAuto()
    from openai import OpenAI
    client = OpenAI(base_url="http://host:8000/v1", api_key="dummy")
    ha.showcase(client, "Qwen/Qwen2.5-14B-Instruct", dataset, n_examples=5)

Each type is shown in colour-coded cards:
    Type 1 (red)   — original vs hallucinated tool response side-by-side
    Type 2 (blue)  — original vs reduced tool response, answer with red spans
    Type 3 (green) — original answer vs answer + overgeneration sentence (red)

Type 1 and 3 require a live vLLM server.  Type 2 is instant (no LLM).
"""

import json
import re
import random


class ShowcaseMixin:

    def showcase(self, client, model: str, dataset: list,
                 n_examples: int = 10, types: list = None,
                 seed: int = 42, timeout: float = 120.0):
        """Generate and display a rich HTML showcase of all hallucination types.

        Parameters
        ----------
        client      : openai.OpenAI client (increase timeout for Type 1/3)
        model       : model identifier
        dataset     : list of row dicts (needs user_prompt, tool_response,
                      tool_call, original_answer, tagged_answer; Type 3 also needs system)
        n_examples  : how many cards per type
        types       : [1, 2, 3] by default; pass e.g. [2] to show only Type 2
        seed        : random seed for reproducible sampling
        timeout     : read timeout for API calls (increase if server is slow)
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

        # ── TYPE 1 ──────────────────────────────────────────────────────────
        if 1 in types:
            parts = ['<h2 style="color:#c0392b;border-bottom:3px solid #c0392b;padding-bottom:6px;">'
                     'TYPE 1: Schema-based JSON corruption</h2>']
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
                        '<div class="sc-hdr t1">#' + str(done) + ' [sample ' + str(idx) + '] — ' + esc(tool_name) + '</div>'
                        '<div class="sc-sec"><div class="sc-lbl">User Prompt</div><div class="sc-body">' + esc(row["user_prompt"]) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Tool Call</div><div class="sc-body">' + esc(row["tool_call"]) + '</div></div>'
                        '<div class="sc-cols">'
                        '<div class="sc-col"><div class="sc-lbl sc-orig-lbl">Original Tool Response</div><div class="sc-body">' + orig_j + '</div></div>'
                        '<div class="sc-col"><div class="sc-lbl sc-hall-lbl">Hallucinated (red = changed)</div><div class="sc-body">' + hall_j + '</div></div>'
                        '</div>'
                        '<div class="sc-sec"><div class="sc-lbl">Original Answer</div><div class="sc-body">' + esc(row["original_answer"]) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Evaluation</div>'
                        '<span class="sc-eval">Target: <b>' + esc(target) + '</b> | ' + esc(str(summary)) + '</span></div>'
                        '</div>'
                    )
                except Exception as e:
                    done += 1
                    parts.append('<div class="sc-err"><b>#' + str(done) + ' [sample ' + str(idx) + ']</b> — ' + esc(str(e)[:200]) + '</div>')
            display(HTML("".join(parts)))
            print(f"Type 1: {done} examples")

        # ── TYPE 2 ──────────────────────────────────────────────────────────
        if 2 in types:
            parts = ['<h2 style="color:#2471a3;border-bottom:3px solid #2471a3;padding-bottom:6px;">'
                     'TYPE 2: Deletion-based overgeneration</h2>']
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
                    disp = row["tagged_answer"]
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
                        '<div class="sc-hdr t2">#' + str(done) + ' [sample ' + str(idx) + '] — ' + esc(tool_name) + '</div>'
                        '<div class="sc-sec"><div class="sc-lbl">User Prompt</div><div class="sc-body">' + esc(row["user_prompt"]) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Tool Call</div><div class="sc-body">' + esc(row["tool_call"]) + '</div></div>'
                        '<div class="sc-cols">'
                        '<div class="sc-col"><div class="sc-lbl sc-orig-lbl">Original Tool Response</div><div class="sc-body">' + orig_j + '</div></div>'
                        '<div class="sc-col"><div class="sc-lbl sc-hall-lbl">Reduced (data removed)</div><div class="sc-body">' + red_j + '</div></div>'
                        '</div>'
                        '<div class="sc-sec"><div class="sc-lbl">Deleted paths</div><div class="sc-body">' + esc(str(deleted_paths)) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Answer (red = references deleted data)</div><div class="sc-body">' + disp + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Hallucination spans</div>'
                        '<span class="sc-eval">' + str(len(spans)) + ' span(s): ' + esc(str([s["tag"] for s in spans])) + '</span></div>'
                        '</div>'
                    )
                except Exception as e:
                    done += 1
                    parts.append('<div class="sc-err"><b>#' + str(done) + ' [sample ' + str(idx) + ']</b> — ' + esc(str(e)[:200]) + '</div>')
            display(HTML("".join(parts)))
            print(f"Type 2: {done} examples")

        # ── TYPE 3 ──────────────────────────────────────────────────────────
        if 3 in types:
            parts = ['<h2 style="color:#1e8449;border-bottom:3px solid #1e8449;padding-bottom:6px;">'
                     'TYPE 3: Subtle tool overgeneration</h2>']
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
                    if any(t.get("name") != used for t in tools):
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
                        '<div class="sc-hdr t3">#' + str(done) + ' [sample ' + str(idx) + '] — Used: ' + esc(used_tool) + '</div>'
                        '<div class="sc-sec"><div class="sc-lbl">User Prompt</div><div class="sc-body">' + esc(row["user_prompt"]) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Tool Call</div><div class="sc-body">' + esc(row["tool_call"]) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Tool Response</div><div class="sc-body">' + esc(fmt_json(original_data)) + '</div></div>'
                        '<div class="sc-sec"><div class="sc-lbl">Other available tools (NOT used)</div>'
                        '<span class="sc-eval">' + esc(", ".join(other_tools)) + '</span></div>'
                        '<div class="sc-cols">'
                        '<div class="sc-col"><div class="sc-lbl sc-orig-lbl">Original Answer</div><div class="sc-body">' + esc(row["original_answer"].rstrip()) + '</div></div>'
                        '<div class="sc-col"><div class="sc-lbl sc-hall-lbl">Hallucinated (red = added)</div><div class="sc-body">' + esc(row["original_answer"].rstrip()) + '<br><br><span class="sc-red">' + esc(comment) + '</span></div></div>'
                        '</div>'
                        '</div>'
                    )
                except Exception as e:
                    done += 1
                    parts.append('<div class="sc-err"><b>#' + str(done) + ' [sample ' + str(idx) + ']</b> — ' + esc(str(e)[:200]) + '</div>')
            display(HTML("".join(parts)))
            print(f"Type 3: {done} examples")

        print("\nShowcase complete.")

    # ── Terminal viewer ───────────────────────────────────────────────────────

    @staticmethod
    def show_samples(data: list, n: int = 3, seed: int = 42, idx: int = None):
        """Print n random (or one specific) dialogues from a pruned dataset.

        Works in any terminal or Jupyter cell — no HTML, plain text with ANSI colour.

        Parameters
        ----------
        data  : list of dialogue dicts (loaded from pruned_type*.json)
        n     : number of random samples to show (ignored when idx is given)
        seed  : random seed for reproducible picks
        idx   : if given, show only this single dialogue index

        Example
        -------
        import json
        from hallucination_auto import HallucinationAuto
        ha = HallucinationAuto()

        data = json.load(open("/pruneddataset/pruned_type2.json"))
        ha.show_samples(data, n=3)          # 3 random
        ha.show_samples(data, idx=7)        # specific index
        ha.show_samples(data, n=5, seed=99) # different random picks
        """
        import random, textwrap

        RED   = "\033[1;31m"
        BOLD  = "\033[1m"
        RESET = "\033[0m"
        SEP   = "═" * 72
        SEP2  = "─" * 72
        ROLE  = {"user": "USER", "assistant": "ASSISTANT", "tool": "TOOL RESPONSE"}

        if idx is not None:
            picks = [idx]
        else:
            random.seed(seed)
            picks = random.sample(range(len(data)), min(n, len(data)))

        for pick_num, i in enumerate(picks, 1):
            d     = data[i]
            convs = d["conversations"]

            print(f"\n{SEP}")
            print(f"  {BOLD}[{pick_num}/{len(picks)}]  index={i}  |  "
                  f"{d['analysis'].get('dialogue_id','?')}{RESET}")
            print(f"  hall_type={d.get('hallucination_type','?')}  |  "
                  f"depth={d['analysis']['completed_tool_turn_count']}  |  "
                  f"skipped={d.get('skipped', False)}")
            print(SEP)

            for ci, c in enumerate(convs):
                role    = c.get("from", "?")
                value   = c.get("value", "")
                label   = ROLE.get(role, role.upper())
                is_hall = "<hall>" in value

                marker = f"  {RED}★ HALLUCINATED{RESET}" if is_hall else ""
                print(f"\n  [{ci}] {BOLD}{label}{RESET}{marker}")
                print(f"  {SEP2}")

                display = (value
                           .replace("<hall>",  f"{RED}❬")
                           .replace("</hall>", f"❭{RESET}"))

                for line in display.split("\n"):
                    if len(line) > 76:
                        line = textwrap.fill(line, width=76,
                                             subsequent_indent="      ")
                    print(f"    {line}")

            hall_turns = [ci for ci, c in enumerate(convs)
                          if "<hall>" in c.get("value", "")]
            print(f"\n  Hallucinated turn(s): {hall_turns}")
            print(SEP)
