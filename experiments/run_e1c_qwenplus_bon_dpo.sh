#!/bin/bash
# E1c: SFT on Qwen-Plus BoN (295) -> DPO on Qwen-Plus turn pairs -> eval
# NOTE: requires data/dpo_dataset/train.jsonl to be non-empty (run build_dataset first)
# Usage: bash run_e1c_qwenplus_bon_dpo.sh [--dry-run]

set -euo pipefail
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[dry-run]"

EXP_NAME="e1c_sft_qwenplus_bon_dpo"
PD_DIR="/user/zhujiatong/pd-tau-bench"
LOG_DIR="$PD_DIR/logs"
VENV="/user/zhujiatong/envs/eval-4b/bin/activate"
BASE_MODEL="/user/zhujiatong/models/Qwen3-4B"

SFT_DATASET="$PD_DIR/data/sft_dataset/train_bon.jsonl"
DPO_DATASET="$PD_DIR/data/dpo_dataset/train.jsonl"

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

# Guard: DPO data must exist
DPO_LINES=$(wc -l < "$DPO_DATASET" 2>/dev/null || echo 0)
if [ "$DPO_LINES" -lt 10 ]; then
    echo "ERROR: $DPO_DATASET is empty ($DPO_LINES lines). Run build_dataset first:" >&2
    echo "  python -m src.data_generation.build_dataset" >&2
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo "SFT: $BASE_MODEL on $SFT_DATASET"
    echo "DPO: $SFT_MERGED on $DPO_DATASET ($DPO_LINES pairs)"
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
