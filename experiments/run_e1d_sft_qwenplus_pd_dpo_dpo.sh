#!/bin/bash
# DPO + eval for e1d_sft_qwenplus_pd_dpo
# SFT 已完成，直接从 merged SFT 开始做 DPO
# Usage: bash run_e1d_sft_qwenplus_pd_dpo_dpo.sh [--dry-run]

set -euo pipefail
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[dry-run]"

EXP_NAME="e1d_sft_qwenplus_pd_dpo"
PD_DIR="/user/zhujiatong/pd-tau-bench"
LOG_DIR="$PD_DIR/logs"
VENV="/user/zhujiatong/envs/eval-4b/bin/activate"
DPO_DATASET="$PD_DIR/data/dpo_dataset/train.jsonl"

SFT_MERGED="$PD_DIR/outputs/models/${EXP_NAME}_sft/merged"
DPO_ADAPTER="$PD_DIR/outputs/models/${EXP_NAME}_dpo/final"
DPO_MERGED="$PD_DIR/outputs/models/${EXP_NAME}_dpo/merged"
OUTPUT_DIR="$PD_DIR/data/results/${EXP_NAME}"
PORT=8002

echo "[job-bootstrap] date=$(date) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
source "$VENV"
export PYTHONPATH="$PD_DIR"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
source "$PD_DIR/experiments/lib_sft_eval.sh"

if [ "$DRY_RUN" = true ]; then
    echo "DPO: $SFT_MERGED on $DPO_DATASET"
    echo "Eval -> $OUTPUT_DIR"
    exit 0
fi

echo "=== DPO stage: $EXP_NAME ==="
cd "$PD_DIR"
python -m src.training.dpo_train \
    --sft-model "$SFT_MERGED" \
    --dataset "$DPO_DATASET" \
    --output "$PD_DIR/outputs/models/${EXP_NAME}_dpo" \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}_dpo.log"

merge_lora "$DPO_ADAPTER" "$DPO_MERGED"

start_server "$DPO_MERGED" "$PORT" "$LOG_DIR/${EXP_NAME}_server.log"
run_eval "$EXP_NAME" "$(basename $DPO_MERGED)" "$PORT" "$OUTPUT_DIR"

kill $SERVER_PID 2>/dev/null || true
echo "Done: $EXP_NAME"
