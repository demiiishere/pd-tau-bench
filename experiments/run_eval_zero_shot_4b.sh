#!/bin/bash
# Zero-shot eval of Qwen3-4B on retail+airline test split
# Usage: bash run_eval_zero_shot_4b.sh [--dry-run]

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] Commands printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

EXP_NAME="eval_zero_shot_qwen3_4b"
PD_DIR="/user/zhujiatong/pd-tau-bench"
LOG_DIR="$PD_DIR/logs"
VENV="/user/zhujiatong/envs/eval-4b/bin/activate"

MODEL_PATH="/user/zhujiatong/models/Qwen3-4B"
SERVED_NAME="qwen3-4b"
PORT=8002
OUTPUT_DIR="$PD_DIR/data/results/zero_shot_4b"

# =============================================================================
# ENVIRONMENT
# =============================================================================

echo "[job-bootstrap] date=$(date) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

source "$VENV"
export no_proxy="*"
export NO_PROXY="*"
export OPENAI_API_BASE="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="fake"
export PYTHONPATH="$PD_DIR"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# =============================================================================
# START INFERENCE SERVER
# =============================================================================

echo "=== Starting inference server ==="
nohup python "$PD_DIR/scripts/serve_model.py" \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    > "$LOG_DIR/${EXP_NAME}_server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait until server is ready
echo "Waiting for server to be ready..."
for i in $(seq 1 60); do
    if curl --noproxy '*' -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 2
done

# =============================================================================
# RUN EVAL
# =============================================================================

if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] Would run eval for retail + airline"
    kill $SERVER_PID 2>/dev/null || true
    exit 0
fi

cd "$PD_DIR"

echo "=== Eval: retail test ==="
python -m src.data_generation.generate_baseline \
    --domain retail \
    --split test \
    --model "openai/${SERVED_NAME}" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 30 \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}_retail.log"

echo "=== Eval: airline test ==="
python -m src.data_generation.generate_baseline \
    --domain airline \
    --split test \
    --model "openai/${SERVED_NAME}" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 30 \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}_airline.log"

# =============================================================================
# SUMMARIZE
# =============================================================================

echo "=== Results ==="
python - << 'PYEOF'
import json
from pathlib import Path
output_dir = Path("/user/zhujiatong/pd-tau-bench/data/results/zero_shot_4b")
for domain in ["retail", "airline"]:
    files = list((output_dir / domain).glob("*_baseline.json"))
    if not files:
        print(f"{domain}: no results")
        continue
    rewards = [json.load(open(f))["final_reward"] for f in files]
    print(f"{domain}: {sum(rewards)}/{len(rewards)} = {sum(rewards)/len(rewards):.3f}")
PYEOF

kill $SERVER_PID 2>/dev/null || true
echo "Done: $EXP_NAME"
