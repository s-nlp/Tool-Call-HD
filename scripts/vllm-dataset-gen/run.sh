#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# SINGLEHOP generation — edit the CONFIG block below, then run:
#
#   ./scripts/run.sh          # all 3 types
#   ./scripts/run.sh 3        # type 3 only
#   ./scripts/run.sh 1 3      # types 1 and 3
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── CONFIG ───────────────────────────────────────────────────────
PYTHON="python3"

# vLLM server
BASE_URL="http://172.17.0.1:8000/v1"
API_KEY="dummy"
MODEL="Qwen/Qwen2.5-14B-Instruct"
TIMEOUT=300

# Input dataset
DATASET="/workspace/dimabsa/workingsolution/last_scrpits/small_datasets/test_generate_input.json"

# Output directory (json + jsonl written here)
OUT_DIR="$PROJECT_ROOT/singlehop_new"

# Concurrent async requests per batch (keep <= --max-num-seqs on vLLM)
BATCH_SIZE=50

# Set to 1 to delete existing output files and start fresh
FRESH=1

# Set to 1 to disable async (slow, one row at a time)
SYNC=0
# ── END CONFIG ───────────────────────────────────────────────────

TYPES="${@:-1 2 3}"

EXTRA=""
[ "$SYNC"  = "1" ] && EXTRA="$EXTRA --sync"
[ "$FRESH" = "1" ] && EXTRA="$EXTRA --fresh"

echo "════════════════════════════════════════════"
echo " SINGLEHOP generation"
echo " Types      : $TYPES"
echo " Batch size : $BATCH_SIZE"
echo " Out dir    : $OUT_DIR"
echo " Server     : $BASE_URL"
echo "════════════════════════════════════════════"
echo ""

$PYTHON generate.py \
    --types      $TYPES \
    --batch-size $BATCH_SIZE \
    --base-url   "$BASE_URL" \
    --api-key    "$API_KEY" \
    --model      "$MODEL" \
    --timeout    $TIMEOUT \
    --dataset    "$DATASET" \
    --out-dir    "$OUT_DIR" \
    $EXTRA
