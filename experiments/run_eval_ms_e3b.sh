#!/bin/bash
# 多种子评估作业: 32B BoN-SFT (e3b)
# 提交方式(无参数): bash experiments/run_eval_ms_e3b.sh
set -euo pipefail
export MS_MODEL="outputs/models/e3b_sft_32b_bon/merged"
export MS_LABEL="e3b_32b_bonsft"
export MS_DOMAINS="retail airline telecom"
source "/user/zhujiatong/pd-tau-bench/experiments/lib_multiseed.sh"
run_multiseed
