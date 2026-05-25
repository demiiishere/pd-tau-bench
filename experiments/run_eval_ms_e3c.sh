#!/bin/bash
# 多种子评估作业: 32B BoN-SFT + PD-DPO (e3c, 论文旗舰配置)
# 提交方式(无参数): bash experiments/run_eval_ms_e3c.sh
set -euo pipefail
export MS_MODEL="outputs/models/e3c_sft_32b_bon_dpo_dpo/merged"
export MS_LABEL="e3c_32b_bondpo"
export MS_DOMAINS="retail airline telecom"
source "/user/zhujiatong/pd-tau-bench/experiments/lib_multiseed.sh"
run_multiseed
