#!/bin/bash
# 多种子评估作业: 32B PD-SFT + PD-DPO (e3d)
# 提交方式(无参数): bash experiments/run_eval_ms_e3d.sh
set -euo pipefail
export MS_MODEL="outputs/models/e3d_sft_32b_pd_dpo_dpo/merged"
export MS_LABEL="e3d_32b_pddpo"
export MS_DOMAINS="retail airline telecom"
source "/user/zhujiatong/pd-tau-bench/experiments/lib_multiseed.sh"
run_multiseed
