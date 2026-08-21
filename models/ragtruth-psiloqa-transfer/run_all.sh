#!/usr/bin/env bash
# Four independent one-A100 experiments:
#   q08_ragtruth, q08_psiloqa, q2_ragtruth, q2_psiloqa.
# Every experiment starts from its cached base model and evaluates on ToolHACE.
set -uo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-/home/jovyan/.mlspace/envs/fresh_vllm/bin/python}
TORCHRUN=${TORCHRUN:-$(dirname "$PYTHON")/torchrun}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

DATA_DIR=${DATA_DIR:-data}
RUN_ROOT=${RUN_ROOT:-runs}
RAGTRUTH_PARQUET="$DATA_DIR/ragtruth_train.parquet"
PSILOQA_PARQUET="$DATA_DIR/psiloqa_train.parquet"
TEST_PARQUET=${TEST_PARQUET:-}
PSILOQA_LANGUAGES=${PSILOQA_LANGUAGES:-en}
RAGTRUTH_QUALITY=${RAGTRUTH_QUALITY:-good}

BS=${BS:-2}
ACCUM=${ACCUM:-16}
MAXLEN=${MAXLEN:-6144}
LR=${LR:-1e-5}
MAP_PROCS=${MAP_PROCS:-8}
RUN_EXPERIMENTS=${RUN_EXPERIMENTS:-q08_ragtruth,q08_psiloqa,q2_ragtruth,q2_psiloqa}

Q08_CACHE=${Q08_CACHE:-/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B}
Q2_CACHE=${Q2_CACHE:-/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B}

RAGTRUTH_LIMIT=${RAGTRUTH_LIMIT:-0}
PSILOQA_LIMIT=${PSILOQA_LIMIT:-0}
SFT_LIMIT=${SFT_LIMIT:-0}
EVAL_LIMIT=${EVAL_LIMIT:-0}

resolve_snapshot() {
  local cache_root=$1 revision snapshot
  if [[ -f "$cache_root/refs/main" ]]; then
    revision=$(tr -d '\r\n' < "$cache_root/refs/main")
    snapshot="$cache_root/snapshots/$revision"
  else
    snapshot=$(find "$cache_root/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
  fi
  if [[ ! -d "$snapshot" ]]; then
    echo "cannot resolve model snapshot under $cache_root" >&2
    return 1
  fi
  printf '%s\n' "$snapshot"
}

contains_experiment() {
  [[ ",$RUN_EXPERIMENTS," == *",$1,"* ]]
}

build_source_data() {
  local source=$1 output=$2 limit=$3 meta="$DATA_DIR/${1}_convert_meta.json"
  if [[ -f "$output" && "${REBUILD_DATA:-0}" != 1 ]]; then
    return 0
  fi
  local data_args=(
    --source "$source"
    --out "$output"
    --meta "$meta"
    --psiloqa-languages "$PSILOQA_LANGUAGES"
    --ragtruth-quality "$RAGTRUTH_QUALITY"
  )
  if [[ "$source" == ragtruth && "$limit" != 0 ]]; then
    data_args+=(--max-ragtruth "$limit")
  fi
  if [[ "$source" == psiloqa && "$limit" != 0 ]]; then
    data_args+=(--max-psiloqa "$limit")
  fi
  [[ "${STREAMING_DATA:-0}" == 1 ]] && data_args+=(--streaming)
  "$PYTHON" build_train_data.py "${data_args[@]}"
}

if [[ -z "$TEST_PARQUET" || ! -f "$TEST_PARQUET" ]]; then
  echo "Set TEST_PARQUET to the ToolHACE test Parquet file." >&2
  exit 1
fi

Q08_MODEL=$(resolve_snapshot "$Q08_CACHE") || exit 1
Q2_MODEL=$(resolve_snapshot "$Q2_CACHE") || exit 1
for model in "$Q08_MODEL" "$Q2_MODEL"; do
  if [[ ! -f "$model/config.json" ]] || ! compgen -G "$model/*.safetensors" >/dev/null; then
    echo "not a standalone local model checkpoint: $model" >&2
    exit 1
  fi
done

mkdir -p "$DATA_DIR" "$RUN_ROOT"
build_source_data ragtruth "$RAGTRUTH_PARQUET" "$RAGTRUTH_LIMIT" || exit 1
build_source_data psiloqa "$PSILOQA_PARQUET" "$PSILOQA_LIMIT" || exit 1

echo "one-A100 configuration: GPU=$GPU bs=$BS accum=$ACCUM effective_batch=$((BS*ACCUM))"
echo "ragtruth_train=$RAGTRUTH_PARQUET"
echo "psiloqa_train=$PSILOQA_PARQUET"
echo "test=$TEST_PARQUET"
echo "q08_start_model=$Q08_MODEL"
echo "q2_start_model=$Q2_MODEL"
echo "experiment_order=$RUN_EXPERIMENTS"

fail=0
experiments=(
  "q08_ragtruth|$Q08_MODEL|$RAGTRUTH_PARQUET"
  "q08_psiloqa|$Q08_MODEL|$PSILOQA_PARQUET"
  "q2_ragtruth|$Q2_MODEL|$RAGTRUTH_PARQUET"
  "q2_psiloqa|$Q2_MODEL|$PSILOQA_PARQUET"
)

for spec in "${experiments[@]}"; do
  IFS='|' read -r tag model train_parquet <<< "$spec"
  contains_experiment "$tag" || continue
  out="$RUN_ROOT/$tag"
  mkdir -p "$out"
  echo
  echo "=== $tag: full SFT on $train_parquet ==="

  if [[ "${SKIP_TRAIN:-0}" != 1 ]]; then
    sft_args=(
      --model "$model"
      --train-parquet "$train_parquet"
      --out "$out/sft"
      --bs "$BS"
      --accum "$ACCUM"
      --maxlen "$MAXLEN"
      --lr "$LR"
      --map-procs "$MAP_PROCS"
    )
    [[ "$SFT_LIMIT" != 0 ]] && sft_args+=(--limit "$SFT_LIMIT")
    [[ "${GRAD_CKPT:-0}" == 1 ]] && sft_args+=(--grad-ckpt)
    "$TORCHRUN" --nproc_per_node=1 sft_json.py "${sft_args[@]}" \
      || { fail=1; continue; }
  fi

  "$PYTHON" convert_composite.py \
    --base "$model" --tuned "$out/sft/model" --out "$out/composite" \
    || { fail=1; continue; }

  echo "=== $tag: evaluate on ToolHACE ==="
  eval_args=(
    --base-model "$model"
    --checkpoint "$out/composite"
    --test-parquet "$TEST_PARQUET"
    --out "$out/eval"
    --tp 1
  )
  [[ "$EVAL_LIMIT" != 0 ]] && eval_args+=(--limit "$EVAL_LIMIT")
  "$PYTHON" eval_toolhace.py "${eval_args[@]}" || { fail=1; continue; }

  "$PYTHON" normalize_for_compute_metrics.py \
    --pred "$out/eval/verdicts.jsonl" \
    --test-parquet "$TEST_PARQUET" \
    --out "$out/eval/metrics_input.jsonl" \
    --model "$tag" \
    --checkpoint "$out/composite" \
    --setting "Full fine-tune from base" \
    || { fail=1; continue; }

  "$PYTHON" compute_metrics.py "$out/eval/metrics_input.jsonl" \
    --output-json "$out/eval/metrics.json" \
    --output-csv "$out/eval/metrics.csv" \
    --setting "Full fine-tune from base" \
    --data ToolHACE \
    --model "$tag" \
    --iou-threshold 0.75 \
    2>&1 | tee "$out/eval/metrics.txt" || fail=1
done

exit "$fail"
