#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Singlehop hallucination generation pipeline
#
# Type map:
#   undergeneration_legacy     flat leaf deletion           (no LLM)
#   undergeneration            cascade deletion, clean spans (no LLM)               ← preferred
#   answer_mismatch_legacy     schema-based hallucination   (vLLM)
#   answer_mismatch_gated      schema hallucination + undergeneration span gate (vLLM)  ← needs undergeneration first
#   answer_mismatch_targeted   span-targeted schema hallucination (vLLM) ← preferred, needs undergeneration first
#                              (unlocks only answer-quoted leaves → higher yield than gated)
#   overgeneration_legacy      tool overgeneration          (vLLM)
#   overgeneration             overgeneration + filler gate (vLLM)                  ← preferred
#
# Examples:
#   ./run.sh                                                              → default: undergeneration, answer_mismatch_gated, overgeneration
#   ./run.sh undergeneration answer_mismatch_targeted overgeneration      → use targeted instead of gated
#   ./run.sh undergeneration overgeneration                               → skip answer-mismatch entirely
#   ./run.sh answer_mismatch_legacy undergeneration_legacy overgeneration_legacy  → legacy types
#   ./run.sh undergeneration                                              → just cascade deletion (no server needed)
#   FRESH=1 ./run.sh overgeneration                                       → re-generate overgeneration from scratch
# ═══════════════════════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── CONFIG ────────────────────────────────────────────────────────────────────
PYTHON="python3"

# vLLM server
BASE_URL="http://172.17.0.1:8000/v1"
API_KEY="dummy"
MODEL="Qwen/Qwen2.5-14B-Instruct"
TIMEOUT=300

# Input dataset (converted singlehop ToolACE JSON — see README for the
# expected schema; supply your own file here)
DATASET="$SCRIPT_DIR/data/singlehop_synthetic_toolace_converted.json"
OUT_DIR="$SCRIPT_DIR/output/singlehop_new"

# Concurrent async requests per batch (keep <= --max-num-seqs on vLLM)
BATCH_SIZE=50

# Set to 1 to delete existing output files and start fresh
FRESH="${FRESH:-0}"

# Set to 1 to use the slow sync path (answer_mismatch_legacy / overgeneration_legacy
# only; no effect on the gated/targeted/preferred variants)
SYNC=0

# Optional: override the undergeneration input for answer_mismatch_gated /
# answer_mismatch_targeted generation.
# Leave empty to use the default <out-dir>/type2_1_output.json.
# Set to the filtered subset to avoid re-running already-done records, e.g.:
#   TYPE2_1_PATH="$SCRIPT_DIR/type1_2_input.json"
TYPE2_1_PATH=""
# ── END CONFIG ────────────────────────────────────────────────────────────────

# Default: all three preferred types (undergeneration must precede answer_mismatch_gated)
TYPES="${@:-undergeneration answer_mismatch_gated overgeneration}"

EXTRA=""
[ "$SYNC"  = "1" ] && EXTRA="$EXTRA --sync"
[ "$FRESH" = "1" ] && EXTRA="$EXTRA --fresh"
[ -n "$TYPE2_1_PATH" ] && EXTRA="$EXTRA --type2-1-path $TYPE2_1_PATH"

echo "════════════════════════════════════════════════════════"
echo " Singlehop hallucination generation"
echo " Types      : $TYPES"
echo " Dataset    : $DATASET"
echo " Batch size : $BATCH_SIZE"
echo " Out dir    : $OUT_DIR"
echo " Server     : $BASE_URL"
echo " Model      : $MODEL"
[ "$FRESH" = "1" ] && echo " Fresh      : YES — existing outputs will be deleted"
echo "════════════════════════════════════════════════════════"
echo ""

LOG="$OUT_DIR/run_log.txt"
mkdir -p "$OUT_DIR"
echo "Log: $LOG"
echo ""

$PYTHON "$SCRIPT_DIR/generate.py" \
    --types      $TYPES \
    --batch-size $BATCH_SIZE \
    --base-url   "$BASE_URL" \
    --api-key    "$API_KEY" \
    --model      "$MODEL" \
    --timeout    $TIMEOUT \
    --dataset    "$DATASET" \
    --out-dir    "$OUT_DIR" \
    $EXTRA 2>&1 | tee "$LOG"
