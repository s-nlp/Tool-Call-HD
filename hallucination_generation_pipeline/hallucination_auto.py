"""
hallucination_auto.py — Main entry point.

Split into four themed files for readability:

  schema.py     — JSON schema tools (locked schema, cascade/focused masking,
                  evaluate_hallucination).  This is the backbone of Type 1.

  singlehop.py  — Type 1 / 2 / 3 hallucination logic for single-turn QA:
                  API call methods (type1_api, type2_delete, type3_api),
                  dataset generation loops (sync + async), system prompts.

  multihop.py   — Multistep dialogue injection + pruning-based generation:
                  detect_type*_spans, _inject_last_turn, generate_multistep_type*,
                  extract_turn_as_row, prune_dialogue, generate_pruned_multistep.

  export.py     — Convert generated data to RAGTruth CSV or JSON format.
                  (_find_in_output uses 7 fallback strategies for offset matching)

Quick start:
    ha = HallucinationAuto()

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="http://host:8000/v1", api_key="dummy")

    import asyncio, json
    multistep = json.load(open("data/toolace_multistep_clean.json"))

    # Pruning-based generation (all depths, all types):
    asyncio.run(ha.generate_pruned_multistep(
        client, "Qwen/Qwen2.5-14B-Instruct",
        multistep, output_dir="pruning_gen/results",
    ))
"""

from schema    import SchemaMixin
from singlehop import SinglehopMixin
from multihop  import MultihopMixin
from export    import ExportMixin


class HallucinationAuto(SchemaMixin, SinglehopMixin, MultihopMixin, ExportMixin):
    """Automated tool-call hallucination generator.

    Generates three types of hallucinations from tool-calling dialogue data:
      Type 1 — Schema-based JSON corruption  (guided decoding, needs vLLM)
      Type 2 — Deletion-based overgeneration (no LLM)
      Type 3 — Subtle tool-constrained overgeneration (needs LLM + system column)

    See the module docstring above for a quick-start example.
    All methods are documented in their respective theme files.
    """

    # Field names that look numeric but must stay as strings (used by Type 2)
    _KEEP_AS_STRING_KEYS = {
        "id", "zip", "zip_code", "postal_code", "postal", "code",
        "phone", "phone_number", "fax", "ssn", "isbn", "ean", "upc",
    }
