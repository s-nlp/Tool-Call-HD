#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LLM-based hallucination generation pipeline
#
# Stage map (each stage reads the previous stage's output):
#   generate   LLM error generation             (generate_errors.py, needs server)
#   unify      raw → unified ToolHACE schema    (to_unified.py, no LLM)
#   fix        unicode/punct span repair        (fix_unicode_drift.py, no LLM)
#   judge      LLM-judge annotation audit       (judge_verify_annotations.py, needs judge server)
#   filter     drop/rescue rows by verdict      (filter_by_judge.py, no LLM)
#   export     → LettuceDetect train/dev/test   (export_lettucedetect.py, no LLM)
#   validate   CI gate on invariants            (validate_output.py, no LLM)
#
# Examples:
#   ./run.sh                          → all stages in order
#   ./run.sh unify fix judge filter   → re-run from an existing generation
#   ./run.sh generate                 → generation only
#   FRESH=1 ./run.sh judge filter     → wipe judge output and re-judge
# ═══════════════════════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── CONFIG ────────────────────────────────────────────────────────────────────
PYTHON="python3"

# Run name — namespaces every intermediate under output/<stage>/<RUN_NAME>
RUN_NAME="singlehop_synthetic"

# Input dataset for generation: HF Hub ID, save_to_disk dir, or file
# (.json/.jsonl/.csv/.parquet). See generate_errors.py header for schemas.
DATASET="$SCRIPT_DIR/data/singlehop_synthetic_toolace_converted.json"
FORMAT="auto"              # auto | flat | conversation
HISTORY_TURNS=1
CLASSES="all"              # or e.g. "correct,hallucination,overgeneration"
SAMPLES_PER_CLASS=""       # empty = all
CONCURRENCY=10

# Generation model (OpenAI-compatible: local vLLM or OpenRouter)
GEN_BASE_URL="http://172.17.0.1:8000/v1"
GEN_MODEL="openai/gpt-oss-120b"
GEN_API_KEY="dummy"

# Judge model (can differ from the generator — recommended)
JUDGE_BASE_URL="https://openrouter.ai/api/v1"
JUDGE_MODEL="openai/gpt-oss-120b"
JUDGE_API_KEY="${OPENROUTER_API_KEY:-dummy}"
JUDGE_CONCURRENCY=8

# Filter policy
APPLY_SUGGESTED_SPANS=1    # 1 = rescue span-only judge failures
ON_UNCERTAIN="drop"        # drop | keep
ON_MISSING="keep"          # keep | drop  (rows without a usable verdict)

# Export split ratios (split assigned by dialogue_id — no leakage)
DEV_RATIO=0.1
TEST_RATIO=0.1

OUT_DIR="$SCRIPT_DIR/output"

# Set to 1 to delete the target stage outputs before running
FRESH="${FRESH:-0}"
# ── END CONFIG ────────────────────────────────────────────────────────────────

GEN_OUT="$OUT_DIR/generated/$RUN_NAME"
UNI_OUT="$OUT_DIR/unified/$RUN_NAME"
FIX_OUT="$OUT_DIR/fixed/$RUN_NAME"
JUDGE_OUT="$OUT_DIR/judge/$RUN_NAME.jsonl"
FINAL_OUT="$OUT_DIR/final_dataset/$RUN_NAME"
FINAL_JSONL="$OUT_DIR/final_dataset/$RUN_NAME.jsonl"
LD_OUT="$OUT_DIR/lettucedetect_data"
REPORTS="$OUT_DIR/reports"

STAGES="${@:-generate unify fix judge filter export validate}"

echo "════════════════════════════════════════════════════════"
echo " LLM hallucination generation pipeline"
echo " Run name   : $RUN_NAME"
echo " Stages     : $STAGES"
echo " Dataset    : $DATASET"
echo " Gen server : $GEN_BASE_URL ($GEN_MODEL)"
echo " Judge      : $JUDGE_BASE_URL ($JUDGE_MODEL)"
echo " Out dir    : $OUT_DIR"
[ "$FRESH" = "1" ] && echo " Fresh      : YES — stage outputs will be deleted"
echo "════════════════════════════════════════════════════════"
echo ""

mkdir -p "$OUT_DIR" "$REPORTS"
LOG="$OUT_DIR/run_log_$RUN_NAME.txt"
echo "Log: $LOG"
echo ""

run_stage() {
    echo ""
    echo "──── stage: $1 ────────────────────────────────────────"
}

{
for STAGE in $STAGES; do
case "$STAGE" in

generate)
    run_stage generate
    [ "$FRESH" = "1" ] && rm -rf "$GEN_OUT"
    EXTRA=""
    [ -n "$SAMPLES_PER_CLASS" ] && EXTRA="--samples-per-class $SAMPLES_PER_CLASS"
    $PYTHON "$SCRIPT_DIR/generate_errors.py" \
        --input "$DATASET" \
        --output "$GEN_OUT" \
        --format "$FORMAT" \
        --history-turns "$HISTORY_TURNS" \
        --classes "$CLASSES" \
        --concurrency "$CONCURRENCY" \
        --model custom \
        --base-url "$GEN_BASE_URL" \
        --model-name "$GEN_MODEL" \
        --api-key "$GEN_API_KEY" \
        $EXTRA
    ;;

unify)
    run_stage unify
    [ "$FRESH" = "1" ] && rm -rf "$UNI_OUT"
    $PYTHON "$SCRIPT_DIR/to_unified.py" \
        --input "$GEN_OUT" \
        --output "$UNI_OUT" \
        --subset "$RUN_NAME" \
        --report "$REPORTS/${RUN_NAME}_unify.jsonl" \
        --on-broken drop
    ;;

fix)
    run_stage fix
    [ "$FRESH" = "1" ] && rm -rf "$FIX_OUT"
    $PYTHON "$SCRIPT_DIR/fix_unicode_drift.py" "$UNI_OUT" \
        --out "$FIX_OUT" \
        --report "$REPORTS/${RUN_NAME}_drift.jsonl"
    ;;

judge)
    run_stage judge
    RESUME="--resume"
    [ "$FRESH" = "1" ] && { rm -f "$JUDGE_OUT"; RESUME=""; }
    $PYTHON "$SCRIPT_DIR/judge_verify_annotations.py" \
        --hf-dataset "$FIX_OUT" \
        --model "$JUDGE_MODEL" \
        --base-url "$JUDGE_BASE_URL" \
        --api-key "$JUDGE_API_KEY" \
        --output "$JUDGE_OUT" \
        --concurrency "$JUDGE_CONCURRENCY" \
        $RESUME
    ;;

filter)
    run_stage filter
    [ "$FRESH" = "1" ] && rm -rf "$FINAL_OUT" "$FINAL_JSONL"
    EXTRA="--on-uncertain $ON_UNCERTAIN --on-missing $ON_MISSING"
    [ "$APPLY_SUGGESTED_SPANS" = "1" ] && EXTRA="$EXTRA --apply-suggested-spans"
    $PYTHON "$SCRIPT_DIR/filter_by_judge.py" \
        --dataset "$FIX_OUT" \
        --judge "$JUDGE_OUT" \
        --output "$FINAL_OUT" \
        --jsonl "$FINAL_JSONL" \
        $EXTRA
    ;;

export)
    run_stage export
    $PYTHON "$SCRIPT_DIR/export_lettucedetect.py" \
        --input "$FINAL_OUT" \
        --out-dir "$LD_OUT" \
        --prefix "toolhace_generation_$RUN_NAME" \
        --dev-ratio "$DEV_RATIO" \
        --test-ratio "$TEST_RATIO"
    ;;

validate)
    run_stage validate
    $PYTHON "$SCRIPT_DIR/validate_output.py" "$FINAL_OUT"
    ;;

*)
    echo "Unknown stage: $STAGE (valid: generate unify fix judge filter export validate)"
    exit 1
    ;;
esac
done
} 2>&1 | tee "$LOG"
