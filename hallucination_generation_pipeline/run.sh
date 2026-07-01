#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Singlehop hallucination generation pipeline
#
# Type map:
#   2    flat leaf deletion           (no LLM)
#   2.1  cascade deletion, clean spans (no LLM)               ← preferred
#   1    schema-based hallucination   (vLLM)
#   1.1  schema hallucination + 2.1 span gate (vLLM)          ← needs 2.1 first
#   1.2  span-targeted schema hallucination (vLLM)            ← preferred, needs 2.1 first
#        (unlocks only answer-quoted leaves → higher yield than 1.1)
#   3    tool overgeneration          (vLLM)
#   3.1  overgeneration + filler gate (vLLM)                  ← preferred
#
# Examples:
#   ./run.sh                  → default: 2.1 then 1.1 then 3.1
#   ./run.sh 2.1 1.2 3.1      → use 1.2 instead of 1.1
#   ./run.sh 2.1 3.1          → skip type-1 entirely
#   ./run.sh 1 2 3            → legacy types
#   ./run.sh 2.1              → just cascade deletion (no server needed)
#   FRESH=1 ./run.sh 3.1      → re-generate type 3.1 from scratch
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

# Set to 1 to use the slow sync path (types 1 / 3 only; no effect on .1 types)
SYNC=0

# Optional: override the Type 2.1 input for 1.1 / 1.2 generation.
# Leave empty to use the default <out-dir>/type2_1_output.json.
# Set to the filtered subset to avoid re-running already-done records, e.g.:
#   TYPE2_1_PATH="$SCRIPT_DIR/type1_2_input.json"
TYPE2_1_PATH=""
# ── END CONFIG ────────────────────────────────────────────────────────────────

# Default: all three preferred types (2.1 must precede 1.1)
TYPES="${@:-2.1 1.1 3.1}"

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
