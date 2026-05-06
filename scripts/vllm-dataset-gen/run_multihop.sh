#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# MULTIHOP generation — pruning-based hallucination injection
#
# Input:  toolace_multistep_clean (1).json  (186 multistep dialogues)
# Output: pruneddataset/pruned_type1.json
#         pruneddataset/pruned_type2.json
#         pruneddataset/pruned_type3.json
#
# For SINGLEHOP generation use run.sh instead.
#
# Usage:
#   ./run_multihop.sh            # all 3 types
#   ./run_multihop.sh 2          # type 2 only (no LLM, instant)
#   ./run_multihop.sh 1 3        # types 1 and 3 only
# ═══════════════════════════════════════════════════════════════

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PYTHON="${PYTHON:-/Users/anatoliifrolov/anaconda3/bin/python3}"
TYPES="${@:-1 2 3}"

# ── Server settings (only needed for type 1 and type 3) ──────────
BASE_URL="http://172.17.0.1:8000/v1"
API_KEY="dummy"
MODEL="Qwen/Qwen2.5-14B-Instruct"
TIMEOUT=300
BATCH_SIZE=5

# ── Paths ─────────────────────────────────────────────────────────
MULTISTEP="toolace_multistep_clean (1).json"
OUT_DIR="pruneddataset"

echo "========================================================"
echo " MULTIHOP generation"
echo " Types  : $TYPES"
echo " Output : $OUT_DIR/"
echo "========================================================"
echo ""

$PYTHON generate_multihop.py \
    --multistep  "$MULTISTEP" \
    --base-url   "$BASE_URL" \
    --api-key    "$API_KEY" \
    --model      "$MODEL" \
    --timeout    $TIMEOUT \
    --batch-size $BATCH_SIZE \
    --out-dir    "$OUT_DIR" \
    --types      $TYPES
