#!/bin/bash
# 实验A 训练种子重复 — e3c: 32B BoN-SFT + PD-DPO (旗舰配置, temp=0 评估)
# 提交(无参数): bash /user/zhujiatong/pd-tau-bench/experiments/run_trainseed_e3c.sh
set -euo pipefail
export TS_LABEL="e3c_bondpo"
export TS_SFT_DATASET="/user/zhujiatong/pd-tau-bench/data/sft_dataset_32b_bon/train.jsonl"
export TS_DPO_DATASET="/user/zhujiatong/pd-tau-bench/data/dpo_dataset_32b/train.jsonl"
export TS_DOMAINS="retail airline telecom"
source "/user/zhujiatong/pd-tau-bench/experiments/lib_trainseed.sh"
run_trainseed
