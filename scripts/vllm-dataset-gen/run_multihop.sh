#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# MULTIHOP generation — edit the CONFIG block below, then run:
#
#   ./scripts/run_multihop.sh          # all 3 types
#   ./scripts/run_multihop.sh 3        # type 3 only
#   ./scripts/run_multihop.sh 1 3      # types 1 and 3
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

# Input multistep dialogues
MULTISTEP="/workspace/dimabsa/workingsolution/last_scrpits/small_datasets/test_generate_input_multistep.json"

# Output directory (pruned_type*.json + .jsonl written here)
OUT_DIR="$PROJECT_ROOT/pruneddataset"

# Concurrent async requests per batch (keep <= --max-num-seqs on vLLM)
BATCH_SIZE=50

# Set to 1 to delete existing output files and start fresh
FRESH=1
# ── END CONFIG ───────────────────────────────────────────────────

TYPES="${@:-1 2 3}"

FRESH_FLAG=""
[ "$FRESH" = "1" ] && FRESH_FLAG="--fresh"

echo "════════════════════════════════════════════"
echo " MULTIHOP generation"
echo " Types      : $TYPES"
echo " Batch size : $BATCH_SIZE"
echo " Out dir    : $OUT_DIR"
echo " Server     : $BASE_URL"
echo "════════════════════════════════════════════"
echo ""

$PYTHON generate_multihop.py \
    --types      $TYPES \
    --batch-size $BATCH_SIZE \
    --base-url   "$BASE_URL" \
    --api-key    "$API_KEY" \
    --model      "$MODEL" \
    --timeout    $TIMEOUT \
    --multistep  "$MULTISTEP" \
    --out-dir    "$OUT_DIR" \
    $FRESH_FLAG
