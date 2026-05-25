#!/bin/bash
# e3b_8b_32b_bon: Qwen3-8B base, data from Qwen3-32B teacher
# Usage: bash run_e3b_8b_32b_bon.sh [--dry-run]

set -euo pipefail
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[dry-run]"

EXP_NAME="e3b_8b_32b_bon"
PD_DIR="/user/zhujiatong/pd-tau-bench"
LOG_DIR="$PD_DIR/logs"
VENV="/user/zhujiatong/envs/eval-4b/bin/activate"
BASE_MODEL="/user/zhujiatong/models/Qwen3-8B"

SFT_DATASET="$PD_DIR/data/sft_dataset_32b_bon/train.jsonl"
ADAPTER_DIR="$PD_DIR/outputs/models/${EXP_NAME}/final"
MERGED_DIR="$PD_DIR/outputs/models/${EXP_NAME}/merged"
OUTPUT_DIR="$PD_DIR/data/results/${EXP_NAME}"
PORT=8002

echo "[job-bootstrap] date=$(date) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
source "$VENV"
export PYTHONPATH="$PD_DIR"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
source "$PD_DIR/experiments/lib_sft_eval.sh"

if [ "$DRY_RUN" = true ]; then
    echo "SFT (8B): $BASE_MODEL on $SFT_DATASET"
    echo "Eval on retail+airline+telecom -> $OUTPUT_DIR"
    exit 0
fi

echo "=== SFT: $EXP_NAME ==="
cd "$PD_DIR"
python -m src.training.sft_train \
    --model "$BASE_MODEL" \
    --dataset "$SFT_DATASET" \
    --output "$PD_DIR/outputs/models/${EXP_NAME}" \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}_sft.log"

merge_lora "$ADAPTER_DIR" "$MERGED_DIR"
start_server "$MERGED_DIR" "$PORT" "$LOG_DIR/${EXP_NAME}_server.log"
run_eval "$EXP_NAME" "$(basename $MERGED_DIR)" "$PORT" "$OUTPUT_DIR" retail airline telecom

kill $SERVER_PID 2>/dev/null || true
echo "Done: $EXP_NAME"
