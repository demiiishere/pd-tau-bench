#!/bin/bash
# Shared functions for SFT/DPO/eval job scripts.
# Source this file; do not run directly.

PD_DIR="/user/zhujiatong/pd-tau-bench"
# BASE_MODEL is set by each experiment script; do not override here

# ---------------------------------------------------------------------------
# merge_lora  <adapter_dir> <merged_dir>
#   Merges LoRA adapter into base model and saves full weights.
# ---------------------------------------------------------------------------
merge_lora() {
    local adapter_dir="$1"
    local merged_dir="$2"
    echo "=== Merging LoRA: $adapter_dir -> $merged_dir ==="
    python - << PYEOF
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "$BASE_MODEL", torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
model = PeftModel.from_pretrained(base, "$adapter_dir")
merged = model.merge_and_unload()
merged.save_pretrained("$merged_dir")
AutoTokenizer.from_pretrained("$BASE_MODEL", trust_remote_code=True).save_pretrained("$merged_dir")
print("Merged model saved to $merged_dir")
PYEOF
}

# ---------------------------------------------------------------------------
# start_server  <model_dir> <port> <log_file>
#   Starts serve_model.py and waits until ready. Sets SERVER_PID.
# ---------------------------------------------------------------------------
start_server() {
    local model_dir="$1"
    local port="$2"
    local log_file="$3"
    local served_name
    served_name=$(basename "$model_dir")

    echo "=== Starting inference server on port $port ==="
    nohup python "$PD_DIR/scripts/serve_model.py" \
        --model "$model_dir" \
        --served-model-name "$served_name" \
        --port "$port" \
        > "$log_file" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"

    for i in $(seq 1 90); do
        if curl --noproxy '*' -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: server did not start in 180s" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# run_eval  <exp_name> <served_name> <port> <output_dir> [domains...]
#   Runs generate_baseline for given domains (default: retail airline).
# ---------------------------------------------------------------------------
run_eval() {
    local exp_name="$1"
    local served_name="$2"
    local port="$3"
    local output_dir="$4"
    shift 4
    local domains=("${@:-retail airline}")
    # If no extra args, default to retail and airline
    if [ ${#domains[@]} -eq 0 ] || [ "${domains[0]}" = "retail airline" ]; then
        domains=(retail airline)
    fi

    export OPENAI_API_BASE="http://localhost:${port}/v1"
    export OPENAI_API_KEY="fake"
    export no_proxy="*"
    export NO_PROXY="*"

    mkdir -p "$output_dir"
    cd "$PD_DIR"

    for domain in "${domains[@]}"; do
        echo "=== Eval: ${domain} test ==="
        python -m src.data_generation.generate_baseline \
            --domain "$domain" \
            --split test \
            --model "openai/${served_name}" \
            --output-dir "$output_dir" \
            --max-steps 30 \
            2>&1 | tee "$PD_DIR/logs/${exp_name}_${domain}.log"
    done

    echo "=== Results: $exp_name ==="
    local domains_py
    domains_py=$(printf '"%s",' "${domains[@]}")
    domains_py="[${domains_py%,}]"
    python - << PYEOF
import json
from pathlib import Path
output_dir = Path("$output_dir")
total_r, total_n = 0, 0
for domain in ${domains_py}:
    files = list((output_dir / domain).glob("*_baseline.json"))
    if not files:
        print(f"  {domain}: no results")
        continue
    rewards = [json.load(open(f))["final_reward"] for f in files]
    print(f"  {domain}: {sum(rewards)}/{len(rewards)} = {sum(rewards)/len(rewards):.3f}")
    total_r += sum(rewards)
    total_n += len(rewards)
if total_n:
    print(f"  TOTAL:  {total_r}/{total_n} = {total_r/total_n:.3f}")
PYEOF
}
