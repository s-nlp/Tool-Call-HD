#!/usr/bin/env python3
"""
fewshot_llm_baseline.py
=======================
Training-free few-shot baseline: prompt a big model (via OpenRouter, OpenAI lib)
to do BOTH 5-class hallucination classification AND span labeling on the TEST
split, then score it against gold. A check-up of how well large models find these
tool-augmented hallucinations out of the box.

Taxonomy (5 classes):
  clean            answer fully supported by tool outputs / user query
  answer_mismatch  answer states a value that CONFLICTS with the tool response
  overgeneration   answer asserts a fact/value NOT in any tool output (fabricated value)
  missing_tool     answer offers/claims a CAPABILITY no available tool supports (fabricated ability)
  undergeneration  answer OMITS tool info the user needed (omission -> no in-text span)

Key design choices
------------------
* The model returns verbatim span SUBSTRINGS, never character offsets -- offsets
  are recomputed here by locating each substring in the final answer (models are
  unreliable at index arithmetic). Localization is drift-tolerant.
* Grounding includes the system/tool schemas, available tools, user turns,
  tool-call arguments, and tool responses -- not just response values.
* Few-shot exemplars include the contrastive overgeneration vs missing_tool pair,
  the most failure-prone boundary.
* Concurrency, retries, incremental writes, and --resume for long/expensive runs.

Env:  OPENROUTER_API_KEY
Deps: openai, datasets (HF input), pandas (CSV input)

Examples
--------
  python fewshot_llm_baseline.py ./test_dir \
      --models openai/gpt-4o anthropic/claude-3.5-sonnet \
      --out-dir runs/ --sample-per-class 40 --concurrency 8
  python fewshot_llm_baseline.py test.csv --models qwen/qwen-2.5-72b-instruct --out-dir runs/
"""
from __future__ import annotations
import argparse, ast, json, os, re, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

LABELS = ["clean", "answer_mismatch", "overgeneration", "missing_tool", "undergeneration"]
NO_SPAN_CLASSES = {"clean", "undergeneration"}

# ----------------------------------------------------------- parse helpers ---
PUNCT_MAP = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u2212": "-", "\u2018": "'", "\u2019": "'", "\u201a": "'",
    "\u201b": "'", "\u2032": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u201f": '"', "\u2033": '"', "\u00a0": " ", "\u202f": " ", "\u2007": " ",
    "\u2009": " ", "\u200a": " ", "\u2002": " ", "\u2003": " ", "\u2005": " ",
    "\u2008": " ",
})
def _norm(s): return s.translate(PUNCT_MAP)

def parse_maybe_literal(x):
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except (ValueError, SyntaxError): return None
    return x

def final_answer_of(conv):
    finals = [t["value"] for t in conv
              if isinstance(t, dict) and t.get("turn_role") == "final_answer"]
    if finals: return finals[-1]
    assts = [t["value"] for t in conv
             if isinstance(t, dict) and t.get("from") == "assistant"]
    return assts[-1] if assts else ""

def locate(answer, needle):
    """First [start,end) of needle in answer (drift-tolerant), else None."""
    n = needle.strip()
    if not n: return None
    i = _norm(answer).find(_norm(n))
    return (i, i + len(n)) if i != -1 else None


# -------------------------------------------------------------- rendering ----
def render_call(value):
    calls = parse_maybe_literal(value)
    if not isinstance(calls, list):
        try: calls = json.loads(value)
        except Exception: return value
    out = []
    for c in (calls if isinstance(calls, list) else [calls]):
        if isinstance(c, dict):
            out.append(f"{c.get('name','?')}({json.dumps(c.get('arguments', {}), ensure_ascii=False)})")
    return "\n".join(out) if out else str(value)

def render_dialogue(system, available_tools, conv, max_tool_chars, include_system=True):
    parts = []
    if include_system and system:
        parts.append("# SYSTEM / TOOL SCHEMAS\n" + str(system).strip())
    at = available_tools
    if isinstance(at, str): at = parse_maybe_literal(at) or at
    parts.append("# AVAILABLE TOOLS\n" + (", ".join(at) if isinstance(at, list) else str(at)))
    parts.append("# CONVERSATION")
    for t in conv:
        role, val = t.get("turn_role"), str(t.get("value", ""))
        if role == "user":
            parts.append(f"[USER]\n{val}")
        elif role == "tool_call":
            parts.append(f"[ASSISTANT TOOL CALL]\n{render_call(val)}")
        elif role == "tool_response":
            if max_tool_chars and len(val) > max_tool_chars:
                val = val[:max_tool_chars] + " …[truncated]"
            parts.append(f"[TOOL RESPONSE]\n{val}")
        elif role == "final_answer":
            parts.append(f"[ASSISTANT FINAL ANSWER]\n{val}")
    return "\n\n".join(parts)


# ----------------------------------------------------------- prompt build ----
SYSTEM_INSTRUCTION = """You are an expert auditor of tool-augmented assistant dialogues. \
You are given a system prompt with tool schemas, the list of available tools, and a \
conversation ending in the assistant's FINAL ANSWER. Decide whether the FINAL ANSWER \
contains a hallucination, and if so which single type, and mark the exact span(s).

Classify into EXACTLY ONE type:
- clean: the final answer is fully supported by the tool responses and the user's request; no fabricated facts, no unsupported capability claims, no conflicts, no harmful omissions.
- answer_mismatch: the final answer states a value or fact that CONTRADICTS what a tool actually returned (the tool response holds a different value). Mark the conflicting value in the answer.
- overgeneration: the final answer asserts a specific FACT or VALUE that appears in NO tool response and was not given by the user — fabricated content stated as if true. Mark the fabricated content.
- missing_tool: the final answer offers or claims an ACTION or CAPABILITY that none of the available tools can perform (e.g. "I'll monitor…", "Would you like me to schedule…", "our system can also pull…"). Mark the unsupported capability claim/offer.
- undergeneration: the final answer OMITS information the tool returned that the user needed, leaving the answer incomplete. The omission is not present as text, so spans MUST be empty.

Critical boundary — overgeneration vs missing_tool:
  overgeneration = a fabricated VALUE/FACT presented as true.
  missing_tool   = a fabricated CAPABILITY/ACTION offered or claimed.

Output rules:
- Respond with ONLY a JSON object, no markdown, no prose around it.
- Schema: {"reason": "<one sentence>", "type": "<one of the five>", "spans": ["<verbatim substring of the FINAL ANSWER>", ...]}
- Each span MUST be copied character-for-character from the FINAL ANSWER text.
- For "clean" and "undergeneration", spans MUST be []."""

def _ex(dialogue, reason, typ, spans):
    return ({"role": "user", "content": dialogue},
            {"role": "assistant",
             "content": json.dumps({"reason": reason, "type": typ, "spans": spans},
                                   ensure_ascii=False)})

BUILTIN_SHOTS = [
    _ex(
        "# AVAILABLE TOOLS\nweather.get_current\n\n# CONVERSATION\n[USER]\nWhat's the temperature in Oslo?\n\n"
        "[ASSISTANT TOOL CALL]\nweather.get_current({\"city\": \"Oslo\"})\n\n"
        "[TOOL RESPONSE]\n{\"results\": {\"temp_c\": 4}}\n\n"
        "[ASSISTANT FINAL ANSWER]\nIt's currently 4°C in Oslo.",
        "The stated 4°C matches the tool's temp_c of 4.", "clean", []),
    _ex(
        "# AVAILABLE TOOLS\nfx.convert\n\n# CONVERSATION\n[USER]\nConvert 100 USD to EUR.\n\n"
        "[ASSISTANT TOOL CALL]\nfx.convert({\"amount\": 100, \"from\": \"USD\", \"to\": \"EUR\"})\n\n"
        "[TOOL RESPONSE]\n{\"results\": {\"converted\": 92.0}}\n\n"
        "[ASSISTANT FINAL ANSWER]\n100 USD is about 87 EUR.",
        "The tool returned 92.0 but the answer says 87, a conflicting value.",
        "answer_mismatch", ["87 EUR"]),
    _ex(
        "# AVAILABLE TOOLS\nstock.get_price\n\n# CONVERSATION\n[USER]\nWhat's ACME trading at?\n\n"
        "[ASSISTANT TOOL CALL]\nstock.get_price({\"ticker\": \"ACME\"})\n\n"
        "[TOOL RESPONSE]\n{\"results\": {\"price\": 12.5}}\n\n"
        "[ASSISTANT FINAL ANSWER]\nACME is at $12.50, up 3% on strong Q3 earnings of $4B.",
        "The Q3 earnings of $4B appear in no tool output and were not provided; fabricated fact.",
        "overgeneration", ["up 3% on strong Q3 earnings of $4B"]),
    _ex(
        "# AVAILABLE TOOLS\ncalendar.list_events\n\n# CONVERSATION\n[USER]\nWhat's on my calendar today?\n\n"
        "[ASSISTANT TOOL CALL]\ncalendar.list_events({\"day\": \"today\"})\n\n"
        "[TOOL RESPONSE]\n{\"results\": {\"events\": [\"Standup 9am\"]}}\n\n"
        "[ASSISTANT FINAL ANSWER]\nYou have Standup at 9am. Would you like me to text the team and reschedule it for you?",
        "Texting/rescheduling is an action no available tool supports; fabricated capability.",
        "missing_tool", ["Would you like me to text the team and reschedule it for you?"]),
    _ex(
        "# AVAILABLE TOOLS\ndb.query\n\n# CONVERSATION\n[USER]\nList all three open tickets with their IDs.\n\n"
        "[ASSISTANT TOOL CALL]\ndb.query({\"status\": \"open\"})\n\n"
        "[TOOL RESPONSE]\n{\"results\": {\"tickets\": [{\"id\": 1}, {\"id\": 2}, {\"id\": 3}]}}\n\n"
        "[ASSISTANT FINAL ANSWER]\nThere is one open ticket: ID 1.",
        "The tool returned three tickets but the answer reports only one; needed info omitted.",
        "undergeneration", []),
]

def build_messages(row, shots, max_tool_chars):
    conv = parse_maybe_literal(row["conversations"])
    target = render_dialogue(row.get("system"), row.get("available_tools"),
                             conv, max_tool_chars, include_system=True)
    msgs = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for u, a in shots:
        msgs.append(u); msgs.append(a)
    msgs.append({"role": "user", "content": target})
    return msgs, final_answer_of(conv)


# ------------------------------------------------------------- model call ----
def make_completer(model, api_key, temperature, json_mode, retries=4):
    from openai import OpenAI
    #client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    client = OpenAI(base_url="http://localhost:8000/v1", api_key=api_key)
    def complete(messages):
        kw = dict(model=model, messages=messages, temperature=temperature)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        last = None
        for a in range(retries):
            try:
                r = client.chat.completions.create(**kw)
                return r.choices[0].message.content or ""
            except Exception as e:                       # noqa: BLE001
                last = e
                if json_mode and "response_format" in str(e).lower():
                    kw.pop("response_format", None)       # provider lacks json mode
                time.sleep(1.5 * (a + 1))
        raise RuntimeError(f"model call failed after {retries} tries: {last}")
    return complete

def parse_output(text):
    """Defensive JSON parse -> (type, spans, reason, ok)."""
    if not text: return None, [], "", False
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    obj = None
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)            # grab first {...}
        if m:
            try: obj = json.loads(m.group(0))
            except json.JSONDecodeError: obj = None
    if not isinstance(obj, dict):
        return None, [], "", False
    typ = str(obj.get("type", "")).strip().lower()
    if typ not in LABELS: typ = None
    spans = obj.get("spans", []) or []
    spans = [s for s in spans if isinstance(s, str) and s.strip()]
    if typ in NO_SPAN_CLASSES: spans = []
    return typ, spans, str(obj.get("reason", "")), (typ is not None)


# ----------------------------------------------------------- per-row work ----
def predict_row(row, complete, shots, max_tool_chars):
    msgs, answer = build_messages(row, shots, max_tool_chars)
    raw = complete(msgs)
    typ, span_texts, reason, ok = parse_output(raw)
    pred_spans, unlocated = [], 0
    for s in span_texts:
        loc = locate(answer, s)
        if loc: pred_spans.append({"start": loc[0], "end": loc[1], "text": answer[loc[0]:loc[1]]})
        else: unlocated += 1
    return dict(pred_type=typ, pred_spans=pred_spans, pred_span_unlocated=unlocated,
                parse_ok=ok, reason=reason, raw=raw)


# --------------------------------------------------------------- scoring -----
def char_set(answer_len, spans):
    s = set()
    for sp in spans:
        a, b = sp.get("start"), sp.get("end")
        if isinstance(a, int) and isinstance(b, int):
            s |= set(range(max(0, a), min(answer_len, b)))
    return s

def score(records):
    conf = defaultdict(lambda: defaultdict(int))
    bin_tp = bin_tn = bin_fp = bin_fn = 0
    ctp = cfp = cfn = 0
    ious, n_parse_fail = [], 0
    for r in records:
        g, p = r["gold_type"], r["pred_type"] or "clean"   # unparseable -> clean (conservative)
        if not r["parse_ok"]: n_parse_fail += 1
        conf[g][p] += 1
        gb, pb = (g != "clean"), (p != "clean")
        bin_tp += gb and pb; bin_tn += (not gb) and (not pb)
        bin_fp += (not gb) and pb; bin_fn += gb and (not pb)
        gc = char_set(r["answer_len"], r["gold_spans"])
        pc = char_set(r["answer_len"], r["pred_spans"])
        ctp += len(gc & pc); cfp += len(pc - gc); cfn += len(gc - pc)
        if gc:
            ious.append(len(gc & pc) / len(gc | pc) if (gc | pc) else 1.0)

    total = sum(sum(d.values()) for d in conf.values())
    acc = sum(conf[c][c] for c in LABELS) / total if total else 0.0
    per_class, recalls, f1s = {}, [], []
    for c in LABELS:
        tp = conf[c][c]
        fp = sum(conf[g][c] for g in LABELS if g != c)
        fn = sum(conf[c][p] for p in LABELS if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = dict(precision=prec, recall=rec, f1=f1, support=tp + fn)
        recalls.append(rec); f1s.append(f1)

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)

    return dict(
        n=total, parse_fail=n_parse_fail,
        fine_accuracy=acc,
        balanced_accuracy=sum(recalls) / len(recalls),
        macro_f1=sum(f1s) / len(f1s),
        binary_accuracy=(bin_tp + bin_tn) / total if total else 0.0,
        binary_hallucination=prf(bin_tp, bin_fp, bin_fn),
        per_class=per_class,
        confusion={g: dict(conf[g]) for g in LABELS},
        span_char=prf(ctp, cfp, cfn),
        span_mean_iou=sum(ious) / len(ious) if ious else 0.0,
        span_iou_rows=len(ious),
    )

def print_metrics(model, m):
    print(f"\n================ {model} ================")
    print(f"  rows={m['n']}  parse_fail={m['parse_fail']}")
    print(f"  fine acc={m['fine_accuracy']:.3f}  bal acc={m['balanced_accuracy']:.3f}  "
          f"macro-F1={m['macro_f1']:.3f}  binary acc={m['binary_accuracy']:.3f}")
    b = m["binary_hallucination"]
    print(f"  binary(hallu) P/R/F1 = {b['precision']:.3f}/{b['recall']:.3f}/{b['f1']:.3f}")
    print(f"  span char  P/R/F1 = {m['span_char']['precision']:.3f}/"
          f"{m['span_char']['recall']:.3f}/{m['span_char']['f1']:.3f}   "
          f"mean IoU={m['span_mean_iou']:.3f} over {m['span_iou_rows']} span rows")
    print("  per-class  P      R      F1     n")
    for c in LABELS:
        pc = m["per_class"][c]
        print(f"    {c:16s}{pc['precision']:.3f}  {pc['recall']:.3f}  {pc['f1']:.3f}  {pc['support']}")
    print("  confusion (gold rows -> pred cols):  " + " ".join(f"{c[:5]:>6}" for c in LABELS))
    for g in LABELS:
        print(f"    {g:16s}" + " ".join(f"{m['confusion'][g].get(p,0):6d}" for p in LABELS))


# ----------------------------------------------------------------- data ------
def load_rows(inp, fmt, split, sample_per_class, limit, seed):
    import random
    rng = random.Random(seed)
    if fmt == "hf":
        from datasets import load_from_disk, DatasetDict
        ds = load_from_disk(inp)
        ds = ds[split] if isinstance(ds, DatasetDict) else ds
        rows = [dict(ds[i]) for i in range(len(ds))]
    else:
        import pandas as pd
        df = pd.read_csv(inp, index_col=0, keep_default_na=False, dtype=str)
        rows = [dict(r, dialogue_id=r.get("dialogue_id", idx)) for idx, r in df.iterrows()]
    for r in rows:
        r["gold_type"] = r.get("type")
        r["_gold_spans_raw"] = r.get("span_labels")
    if sample_per_class:
        by = defaultdict(list)
        for r in rows: by[r["gold_type"]].append(r)
        picked = []
        for c, lst in by.items():
            rng.shuffle(lst); picked += lst[:sample_per_class]
        rows = picked
    if limit: rows = rows[:limit]
    return rows

def gold_spans_of(row):
    sp = parse_maybe_literal(row["_gold_spans_raw"]) or []
    out = []
    for s in sp:
        if isinstance(s, dict) and isinstance(s.get("start"), int) and isinstance(s.get("end"), int):
            out.append({"start": s["start"], "end": s["end"], "text": s.get("text", "")})
    return out


# ----------------------------------------------------------------- run -------
def run_model(model, rows, complete, shots, max_tool_chars, concurrency,
              out_path, resume):
    done = {}
    if resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line); done[rec["dialogue_id"]] = rec
                except Exception: pass
        print(f"  resume: {len(done)} cached")
    todo = [r for r in rows if r["dialogue_id"] not in done]
    records = list(done.values())

    fh = open(out_path, "a", encoding="utf-8")
    def work(r):
        answer = final_answer_of(parse_maybe_literal(r["conversations"]))
        try:
            pred = predict_row(r, complete, shots, max_tool_chars)
            err = None
        except Exception as e:                              # noqa: BLE001
            pred = dict(pred_type=None, pred_spans=[], pred_span_unlocated=0,
                        parse_ok=False, reason="", raw=""); err = str(e)
        return dict(dialogue_id=r["dialogue_id"], gold_type=r["gold_type"],
                    gold_spans=gold_spans_of(r), answer_len=len(answer),
                    error=err, **pred)

    if todo:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(work, r): r for r in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                records.append(rec)
                if i % 25 == 0 or i == len(todo):
                    print(f"  {model}: {i}/{len(todo)}")
    fh.close()
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="HF dataset dir or CSV (test split)")
    ap.add_argument("--models", nargs="+", required=True, help="OpenRouter model slugs")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--format", choices=["csv", "hf"])
    ap.add_argument("--sample-per-class", type=int, default=0,
                    help="balanced subset for a cheap check-up (0 = full split)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tool-chars", type=int, default=6000)
    ap.add_argument("--no-json-mode", action="store_true",
                    help="disable response_format=json_object")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key: sys.exit("ERROR: OPENROUTER_API_KEY not set")

    fmt = args.format or ("hf" if os.path.isdir(args.input) else "csv")
    rows = load_rows(args.input, fmt, args.split, args.sample_per_class,
                     args.limit, args.seed)
    print(f"loaded {len(rows)} rows  | class dist: "
          f"{dict(Counter(r['gold_type'] for r in rows))}")
    os.makedirs(args.out_dir, exist_ok=True)
    shots = BUILTIN_SHOTS

    all_metrics = {}
    for model in args.models:
        complete = make_completer(model, api_key, args.temperature, not args.no_json_mode)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        preds_path = os.path.join(args.out_dir, f"preds_{safe}.jsonl")
        records = run_model(model, rows, complete, shots, args.max_tool_chars,
                            args.concurrency, preds_path, args.resume)
        m = score(records)
        all_metrics[model] = m
        print_metrics(model, m)
        with open(os.path.join(args.out_dir, f"metrics_{safe}.json"), "w") as fh:
            json.dump(m, fh, indent=2)

    if len(all_metrics) > 1:
        print("\n================ comparison ================")
        print(f"  {'model':40s} fineAcc binAcc macroF1 spanF1")
        for mdl, m in all_metrics.items():
            print(f"  {mdl:40s} {m['fine_accuracy']:.3f}  {m['binary_accuracy']:.3f}  "
                  f"{m['macro_f1']:.3f}  {m['span_char']['f1']:.3f}")
    with open(os.path.join(args.out_dir, "metrics_all.json"), "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"\nwrote results to {args.out_dir}/")


if __name__ == "__main__":
    main()
