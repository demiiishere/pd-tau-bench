#!/bin/bash
# E2c: SFT on on-policy PD (111) -> DPO on on-policy turn pairs (349) -> eval
# Usage: bash run_e2c_onpolicy_pd_dpo.sh [--dry-run]

set -euo pipefail
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[dry-run]"

EXP_NAME="e2c_sft_onpolicy_pd_dpo"
PD_DIR="/user/zhujiatong/pd-tau-bench"
LOG_DIR="$PD_DIR/logs"
VENV="/user/zhujiatong/envs/eval-4b/bin/activate"
BASE_MODEL="/user/zhujiatong/models/Qwen3-4B"

SFT_DATASET="$PD_DIR/data/sft_dataset_onpolicy/train.jsonl"
DPO_DATASET="$PD_DIR/data/dpo_dataset_onpolicy/train.jsonl"

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
    DPO_LINES=$(wc -l < "$DPO_DATASET" 2>/dev/null || echo 0)
    echo "SFT: $BASE_MODEL on $SFT_DATASET"
    echo "DPO: -> $DPO_DATASET ($DPO_LINES pairs)"
    echo "Eval on retail+airline test -> $OUTPUT_DIR"
    exit 0
fi

# --- SFT ---
echo "=== SFT stage: $EXP_NAME ==="
cd "$PD_DIR"
python -m src.training.sft_train \
    --model "$BASE_MODEL" \
    --dataset "$SFT_DATASET" \
    --output "$PD_DIR/outputs/models/${EXP_NAME}_sft" \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}_sft.log"

merge_lora "$SFT_ADAPTER" "$SFT_MERGED"

# --- DPO ---
echo "=== DPO stage: $EXP_NAME ==="
python -m src.training.dpo_train \
    --sft-model "$SFT_MERGED" \
    --dataset "$DPO_DATASET" \
    --output "$PD_DIR/outputs/models/${EXP_NAME}_dpo" \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}_dpo.log"

merge_lora "$DPO_ADAPTER" "$DPO_MERGED"

# --- Eval ---
start_server "$DPO_MERGED" "$PORT" "$LOG_DIR/${EXP_NAME}_server.log"
run_eval "$EXP_NAME" "$(basename $DPO_MERGED)" "$PORT" "$OUTPUT_DIR"

kill $SERVER_PID 2>/dev/null || true
echo "Done: $EXP_NAME"
