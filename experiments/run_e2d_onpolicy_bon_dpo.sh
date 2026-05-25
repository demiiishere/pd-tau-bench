#!/bin/bash
# E2d: SFT on on-policy BoN (99) -> DPO on on-policy PD turn-level pairs (349)
# 对应 E1c 的 on-policy 版本
# Usage: bash run_e2d_onpolicy_bon_dpo.sh [--dry-run]

set -euo pipefail
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[dry-run]"

EXP_NAME="e2d_sft_onpolicy_bon_dpo"
PD_DIR="/user/zhujiatong/pd-tau-bench"
LOG_DIR="$PD_DIR/logs"
VENV="/user/zhujiatong/envs/eval-4b/bin/activate"
BASE_MODEL="/user/zhujiatong/models/Qwen3-4B"

SFT_DATASET="$PD_DIR/data/sft_dataset_bon3/train.jsonl"          # on-policy BoN, 99条
DPO_DATASET="$PD_DIR/data/dpo_dataset_onpolicy/train.jsonl"      # on-policy PD turn-level, 349对

SFT_ADAPTER="$PD_DIR/outputs/models/${EXP_NAME}_sft/final"
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
    echo "SFT: $BASE_MODEL on $SFT_DATASET"
    echo "DPO: $SFT_MERGED on $DPO_DATASET ($(wc -l < $DPO_DATASET) pairs)"
    echo "Eval on retail+airline test -> $OUTPUT_DIR"
    exit 0
fi

echo "=== SFT stage: $EXP_NAME ==="
cd "$PD_DIR"
python -m src.training.sft_train \
    --model "$BASE_MODEL" \
    --dataset "$SFT_DATASET" \
    --output "$PD_DIR/outputs/models/${EXP_NAME}_sft" \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}_sft.log"

merge_lora "$SFT_ADAPTER" "$SFT_MERGED"

echo "=== DPO stage: $EXP_NAME ==="
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
