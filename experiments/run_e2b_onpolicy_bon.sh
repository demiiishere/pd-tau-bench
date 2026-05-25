#!/bin/bash
# E2b: SFT on on-policy BoN data (99 samples) -> eval
# Usage: bash run_e2b_onpolicy_bon.sh [--dry-run]

set -euo pipefail
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[dry-run]"

EXP_NAME="e2b_sft_onpolicy_bon"
PD_DIR="/user/zhujiatong/pd-tau-bench"
LOG_DIR="$PD_DIR/logs"
VENV="/user/zhujiatong/envs/eval-4b/bin/activate"
BASE_MODEL="/user/zhujiatong/models/Qwen3-4B"

DATASET="$PD_DIR/data/sft_dataset_bon3/train.jsonl"
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
    echo "SFT: $BASE_MODEL on $DATASET -> $ADAPTER_DIR"
    echo "Merge: $ADAPTER_DIR -> $MERGED_DIR"
    echo "Eval on retail+airline test -> $OUTPUT_DIR"
    exit 0
fi

echo "=== SFT: $EXP_NAME ==="
cd "$PD_DIR"
python -m src.training.sft_train \
    --model "$BASE_MODEL" \
    --dataset "$DATASET" \
    --output "$PD_DIR/outputs/models/${EXP_NAME}" \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}_sft.log"

merge_lora "$ADAPTER_DIR" "$MERGED_DIR"
start_server "$MERGED_DIR" "$PORT" "$LOG_DIR/${EXP_NAME}_server.log"
run_eval "$EXP_NAME" "$(basename $MERGED_DIR)" "$PORT" "$OUTPUT_DIR"

kill $SERVER_PID 2>/dev/null || true
echo "Done: $EXP_NAME"
