#!/bin/bash
# 实验A 训练种子重复 — e3b: 32B BoN-SFT (仅 SFT, temp=0 评估)
# 提交(无参数): bash /user/zhujiatong/pd-tau-bench/experiments/run_trainseed_e3b.sh
set -euo pipefail
export TS_LABEL="e3b_bonsft"
export TS_SFT_DATASET="/user/zhujiatong/pd-tau-bench/data/sft_dataset_32b_bon/train.jsonl"
export TS_DPO_DATASET=""
export TS_DOMAINS="retail airline telecom"
source "/user/zhujiatong/pd-tau-bench/experiments/lib_trainseed.sh"
run_trainseed
