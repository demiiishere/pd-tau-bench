#!/bin/bash
# 多种子评估作业: 零样本 Qwen3-4B 基线
# 提交方式(无参数, 与 run_e3c_32b_bon_dpo.sh 完全一致): bash experiments/run_eval_ms_zeroshot.sh
set -euo pipefail
export MS_MODEL="/user/zhujiatong/models/Qwen3-4B"
export MS_LABEL="zeroshot_4b"
export MS_DOMAINS="retail airline telecom"
source "/user/zhujiatong/pd-tau-bench/experiments/lib_multiseed.sh"
run_multiseed
