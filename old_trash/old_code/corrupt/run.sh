#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Generate hallucination datasets
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
