#!/bin/bash
# Step 2: Generate PD and baseline trajectories (Phase A)
# Run from project root: bash scripts/step2_generate.sh

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pd-tau-bench
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "ERROR: DASHSCOPE_API_KEY not set"
    exit 1
fi

echo "=== Step 2a: Test with single task (K=3, H=1) ==="
python -m src.data_generation.generate_trajectories \
    --domain retail \
    --task-ids 0 \
    --K 3 --H 1 \
    --model openai/qwen-plus \
    --num-trials 1

echo ""
echo "=== Inspect first trajectory ==="
python -m src.data_generation.inspect_trajectories \
    --trajectory-file data/raw_trajectories/retail/task_0_trial_0_pd.json

echo ""
echo "=== Step 2b: Full generation (K=5, H=2) ==="
echo "This will take several hours and cost ~¥50-100 in API fees."
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -m src.data_generation.generate_trajectories \
        --domain retail airline \
        --K 5 --H 2 \
        --model openai/qwen-plus \
        --num-trials 3 \
        --max-concurrency 5
fi

echo ""
echo "=== Step 2c: Build training datasets ==="
python -m src.data_generation.build_dataset \
    --raw-dir data/raw_trajectories/ \
    --sft-output data/sft_dataset/train.jsonl \
    --dpo-output data/dpo_dataset/train.jsonl
