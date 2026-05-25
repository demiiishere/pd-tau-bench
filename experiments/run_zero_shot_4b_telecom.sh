#!/bin/bash
# Zero-shot Qwen3-4B eval on telecom (补齐 baseline)
# Usage: bash run_zero_shot_4b_telecom.sh [--dry-run]

set -euo pipefail
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[dry-run]"

EXP_NAME="zero_shot_4b"
PD_DIR="/user/zhujiatong/pd-tau-bench"
LOG_DIR="$PD_DIR/logs"
VENV="/user/zhujiatong/envs/eval-4b/bin/activate"
MODEL="/user/zhujiatong/models/Qwen3-4B"
OUTPUT_DIR="$PD_DIR/data/results/${EXP_NAME}"
PORT=8002

echo "[job-bootstrap] date=$(date) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
source "$VENV"
export PYTHONPATH="$PD_DIR"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
source "$PD_DIR/experiments/lib_sft_eval.sh"

if [ "$DRY_RUN" = true ]; then
    echo "Zero-shot eval: $MODEL on telecom test -> $OUTPUT_DIR"
    exit 0
fi

start_server "$MODEL" "$PORT" "$LOG_DIR/${EXP_NAME}_telecom_server.log"
run_eval "$EXP_NAME" "$(basename $MODEL)" "$PORT" "$OUTPUT_DIR" telecom

kill $SERVER_PID 2>/dev/null || true
echo "Done: zero-shot telecom"
