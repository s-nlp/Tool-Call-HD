#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# SINGLEHOP generation — flat QA dataset → type1/2/3 outputs
#
# Input:  dataset_v3_tagged_cleaned_sys.json  (or any flat dataset)
# Output: type1_output.json, type2_output.json, type3_output.json
#
# For MULTIHOP (pruning-based) generation use run_multihop.sh
#
# Usage:
#   ./run.sh              # all 3 types
#   ./run.sh 2            # just type 2 (instant, no LLM)
#   ./run.sh 1 3          # types 1 and 3 only
# ═══════════════════════════════════════════════════════════════

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PYTHON="${PYTHON:-python3}"
TYPES="${@:-1 2 3}"

echo "Generating types: $TYPES"
echo ""

$PYTHON generate.py --types $TYPES
