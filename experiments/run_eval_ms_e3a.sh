#!/bin/bash
# 多种子评估作业: 32B PD-SFT (e3a)
# 提交方式(无参数): bash experiments/run_eval_ms_e3a.sh
set -euo pipefail
export MS_MODEL="outputs/models/e3a_sft_32b_pd/merged"
export MS_LABEL="e3a_32b_pdsft"
export MS_DOMAINS="retail airline telecom"
source "/user/zhujiatong/pd-tau-bench/experiments/lib_multiseed.sh"
run_multiseed
