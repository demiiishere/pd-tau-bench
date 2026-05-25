#!/bin/bash
# E1a: SFT on Qwen-Plus PD data (197 samples) -> eval
# Usage: bash run_e1a_qwenplus_pd.sh [--dry-run]

set -euo pipefail
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true && echo "[dry-run]"

# =============================================================================
# CONFIGURATION
# =============================================================================

EXP_NAME="e1a_sft_qwenplus_pd"
PD_DIR="/user/zhujiatong/pd-tau-bench"
LOG_DIR="$PD_DIR/logs"
VENV="/user/zhujiatong/envs/eval-4b/bin/activate"
BASE_MODEL="/user/zhujiatong/models/Qwen3-4B"

DATASET="$PD_DIR/data/sft_dataset/train.jsonl"
ADAPTER_DIR="$PD_DIR/outputs/models/${EXP_NAME}/final"
MERGED_DIR="$PD_DIR/outputs/models/${EXP_NAME}/merged"
OUTPUT_DIR="$PD_DIR/data/results/${EXP_NAME}"
PORT=8002

# =============================================================================
# ENVIRONMENT
# =============================================================================

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

# =============================================================================
# SFT
# =============================================================================

echo "=== SFT: $EXP_NAME ==="
cd "$PD_DIR"
python -m src.training.sft_train \
    --model "$BASE_MODEL" \
    --dataset "$DATASET" \
    --output "$PD_DIR/outputs/models/${EXP_NAME}" \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}_sft.log"

# =============================================================================
# MERGE + SERVE + EVAL
# =============================================================================

merge_lora "$ADAPTER_DIR" "$MERGED_DIR"
start_server "$MERGED_DIR" "$PORT" "$LOG_DIR/${EXP_NAME}_server.log"
run_eval "$EXP_NAME" "$(basename $MERGED_DIR)" "$PORT" "$OUTPUT_DIR"

kill $SERVER_PID 2>/dev/null || true
echo "Done: $EXP_NAME"
