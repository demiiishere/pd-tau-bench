#!/bin/bash
# Start vLLM for evaluation or on-policy data generation.
# Usage: bash scripts/start_vllm.sh [base|bon|pd|dpo|pd_onpolicy|bon_onpolicy]
#   base         → serve base Qwen3-8B, no LoRA, thinking ON  (for on-policy data generation)
#   bon          → serve sft_bon/final         with LoRA, thinking OFF
#   pd           → serve sft_pd/final          with LoRA, thinking OFF
#   dpo          → serve dpo_pd/final          with LoRA, thinking OFF
#   pd_onpolicy  → serve sft_pd_onpolicy/final  with LoRA, thinking ON  (on-policy evaluation)
#   bon_onpolicy   → serve sft_bon3_onpolicy/final  with LoRA, thinking ON  (BoN ablation evaluation)
#   mixed_onpolicy → serve sft_mixed_onpolicy/final with LoRA, thinking ON  (PD+BoN mixed evaluation)

set -e

VARIANT="${1:-bon}"
BASE_MODEL="/user/zhujiatong/models/Qwen3-8B"
PORT=8001
LOG_FILE="/tmp/vllm_${VARIANT}.log"

eval "$(conda shell.bash hook)"
conda activate vllm-env

if [ "$VARIANT" = "base" ]; then
    # Base model, no LoRA — used for on-policy data generation (thinking ON)
    echo "Starting vLLM: base model (no LoRA, thinking ON), port=${PORT}"
    echo "Log: ${LOG_FILE}"
    vllm serve "${BASE_MODEL}" --served-model-name finetuned --port "${PORT}" --trust-remote-code --dtype bfloat16 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes > "${LOG_FILE}" 2>&1 &
elif [ "$VARIANT" = "pd_onpolicy" ]; then
    LORA_PATH="/user/zhujiatong/outputs_pd/sft_pd_onpolicy/final"
    echo "Starting vLLM: model=${VARIANT}, lora=${LORA_PATH}, thinking ON, port=${PORT}"
    echo "Log: ${LOG_FILE}"
    vllm serve "${BASE_MODEL}" --enable-lora --lora-modules "finetuned=${LORA_PATH}" --port "${PORT}" --trust-remote-code --dtype bfloat16 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes > "${LOG_FILE}" 2>&1 &
elif [ "$VARIANT" = "bon_onpolicy" ]; then
    LORA_PATH="/user/zhujiatong/outputs_pd/sft_bon3_onpolicy/final"
    echo "Starting vLLM: model=${VARIANT}, lora=${LORA_PATH}, thinking ON, port=${PORT}"
    echo "Log: ${LOG_FILE}"
    vllm serve "${BASE_MODEL}" --enable-lora --lora-modules "finetuned=${LORA_PATH}" --port "${PORT}" --trust-remote-code --dtype bfloat16 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes > "${LOG_FILE}" 2>&1 &
elif [ "$VARIANT" = "mixed_onpolicy" ]; then
    LORA_PATH="/user/zhujiatong/outputs_pd/sft_mixed_onpolicy/final"
    echo "Starting vLLM: model=${VARIANT}, lora=${LORA_PATH}, thinking ON, port=${PORT}"
    echo "Log: ${LOG_FILE}"
    vllm serve "${BASE_MODEL}" --enable-lora --lora-modules "finetuned=${LORA_PATH}" --port "${PORT}" --trust-remote-code --dtype bfloat16 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes > "${LOG_FILE}" 2>&1 &
else
    if [ "$VARIANT" = "pd" ]; then
        LORA_PATH="/user/zhujiatong/outputs_pd/sft_pd/final"
    elif [ "$VARIANT" = "bon" ]; then
        LORA_PATH="/user/zhujiatong/outputs_pd/sft_bon/final"
    else
        LORA_PATH="/user/zhujiatong/outputs_pd/dpo_pd/final"
    fi
    echo "Starting vLLM: model=${VARIANT}, lora=${LORA_PATH}, thinking OFF, port=${PORT}"
    echo "Log: ${LOG_FILE}"
    vllm serve "${BASE_MODEL}" --enable-lora --lora-modules "finetuned=${LORA_PATH}" --port "${PORT}" --trust-remote-code --dtype bfloat16 --override-generation-config '{"enable_thinking": false}' --enable-auto-tool-choice --tool-call-parser hermes > "${LOG_FILE}" 2>&1 &
fi

VLLM_PID=$!
disown ${VLLM_PID}   # detach from shell so it survives terminal close
echo "vLLM PID: ${VLLM_PID}"
echo "Waiting for startup..."
sleep 30
echo "Last 5 lines of log:"
tail -5 "${LOG_FILE}"
echo ""
echo "To check status: tail -f ${LOG_FILE}"
echo "To verify:       curl http://localhost:${PORT}/v1/models"
echo "To stop:         kill ${VLLM_PID}"
