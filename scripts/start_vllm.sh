#!/bin/bash
# Start vLLM with LoRA adapter for evaluation.
# Usage: bash scripts/start_vllm.sh [bon|pd]
#   bon  → serve sft_bon/final   (default)
#   pd   → serve sft_pd/final

set -e

VARIANT="${1:-bon}"
BASE_MODEL="/user/zhujiatong/models/Qwen3-8B"
PORT=8001
LOG_FILE="/tmp/vllm_${VARIANT}.log"

if [ "$VARIANT" = "pd" ]; then
    LORA_PATH="/user/zhujiatong/outputs_pd/sft_pd/final"
else
    LORA_PATH="/user/zhujiatong/outputs_pd/sft_bon/final"
fi

echo "Starting vLLM: model=${VARIANT}, lora=${LORA_PATH}, port=${PORT}"
echo "Log: ${LOG_FILE}"

eval "$(conda shell.bash hook)"
conda activate vllm-env

nohup vllm serve "${BASE_MODEL}" \
    --enable-lora \
    --lora-modules "finetuned=${LORA_PATH}" \
    --port "${PORT}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "${LOG_FILE}" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"
echo "Waiting for startup..."
sleep 30
echo "Last 5 lines of log:"
tail -5 "${LOG_FILE}"
echo ""
echo "To check status: tail -f ${LOG_FILE}"
echo "To verify:       curl http://localhost:${PORT}/v1/models"
echo "To stop:         kill ${VLLM_PID}"
