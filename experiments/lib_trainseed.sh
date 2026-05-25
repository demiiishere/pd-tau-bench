#!/bin/bash
# ===========================================================================
# 实验 A —— 训练种子重复(评估全程 temp=0)共享逻辑.
# source 本文件后调用 run_trainseed. 不要直接运行.
#
# 调用方(无参数作业脚本)需 export:
#   TS_LABEL        实验标签
#   TS_SFT_DATASET  SFT 数据集 —— 完整绝对路径
#   TS_DPO_DATASET  DPO 数据集 —— 完整绝对路径; 留空字符串 = 只做 SFT
# 可选(环境变量):
#   TS_DOMAINS="retail airline telecom"   SEEDS="1 2 3"   PORT=8020   DRY_RUN=1
#
# 原理: temp=0 评估是确定性的, 但训练(数据 shuffle / LoRA 初始化)受种子控制.
#       换 N 个训练种子重训, 每个仍在 temp=0 下评估 => 得到 temp=0 pass@1 的
#       mean±std, 用以判断"14.5%"是方法的稳定属性还是某次走运的训练.
# ===========================================================================

run_trainseed() {
    PD_DIR="/user/zhujiatong/pd-tau-bench"
    BASE_MODEL="/user/zhujiatong/models/Qwen3-4B"
    local VENV="/user/zhujiatong/envs/eval-4b/bin/activate"
    local LOG_DIR="/user/zhujiatong/pd-tau-bench/logs"
    local RESULTS_BASE="/user/zhujiatong/pd-tau-bench/data/results/trainseed"
    local MODELS_BASE="/user/zhujiatong/pd-tau-bench/outputs/models/trainseed"

    local label="${TS_LABEL:?lib_trainseed: 需要 export TS_LABEL}"
    local sft_data="${TS_SFT_DATASET:?lib_trainseed: 需要 export TS_SFT_DATASET}"
    local dpo_data="${TS_DPO_DATASET:-}"
    local domains="${TS_DOMAINS:-retail airline telecom}"
    local seeds="${SEEDS:-1 2 3}"
    local port="${PORT:-8020}"

    echo "[job-bootstrap] date=$(date) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "[config] label=$label"
    echo "[config] base_model=$BASE_MODEL"
    echo "[config] sft_dataset=$sft_data"
    echo "[config] dpo_dataset=${dpo_data:-<无, 仅 SFT>}"
    echo "[config] domains=$domains  seeds=[$seeds]  port=$port"
    echo "[config] eval(temp=0) -> $RESULTS_BASE/$label/seed_<S>/"

    [ -f "$sft_data" ] || { echo "ERROR: SFT 数据集不存在: $sft_data" >&2; exit 1; }
    if [ -n "$dpo_data" ] && [ ! -f "$dpo_data" ]; then
        echo "ERROR: DPO 数据集不存在: $dpo_data" >&2; exit 1
    fi

    if [ "${DRY_RUN:-0}" = "1" ]; then
        local s
        for s in $seeds; do
            echo "[dry-run] seed=$s: SFT$([ -n "$dpo_data" ] && echo '+DPO') -> merge -> temp=0 评估 -> $RESULTS_BASE/$label/seed_${s}/"
        done
        return 0
    fi

    source "$VENV"
    export PYTHONPATH="$PD_DIR"
    export no_proxy="*" NO_PROXY="*"
    mkdir -p "$LOG_DIR"
    source "$PD_DIR/experiments/lib_sft_eval.sh"
    cd "$PD_DIR"

    local S
    for S in $seeds; do
        echo "################  [$label]  训练种子 $S  ################"
        local sft_out="$MODELS_BASE/${label}/seed${S}_sft"
        local sft_merged="$sft_out/merged"
        local eval_model

        echo "=== SFT (seed=$S) ==="
        python -m src.training.sft_train \
            --model "$BASE_MODEL" \
            --dataset "$sft_data" \
            --output "$sft_out" \
            --seed "$S" \
            2>&1 | tee "$LOG_DIR/trainseed_${label}_seed${S}_sft.log"
        merge_lora "$sft_out/final" "$sft_merged"

        if [ -n "$dpo_data" ]; then
            local dpo_out="$MODELS_BASE/${label}/seed${S}_dpo"
            echo "=== DPO (seed=$S) ==="
            python -m src.training.dpo_train \
                --sft-model "$sft_merged" \
                --dataset "$dpo_data" \
                --output "$dpo_out" \
                --seed "$S" \
                2>&1 | tee "$LOG_DIR/trainseed_${label}_seed${S}_dpo.log"
            merge_lora "$dpo_out/final" "$dpo_out/merged"
            eval_model="$dpo_out/merged"
        else
            eval_model="$sft_merged"
        fi

        echo "=== 评估 temp=0 (seed=$S) ==="
        start_server "$eval_model" "$port" "$LOG_DIR/trainseed_${label}_seed${S}_server.log"
        run_eval "trainseed_${label}_seed${S}" "$(basename "$eval_model")" "$port" \
                 "$RESULTS_BASE/$label/seed_${S}" $domains
        kill "${SERVER_PID:-0}" 2>/dev/null || true
        sleep 3
    done

    echo "================  汇总: $label (训练种子 mean±std, 全程 temp=0)  ================"
    python "$PD_DIR/experiments/aggregate_multiseed.py" \
        --results-base "$RESULTS_BASE" --label "$label" \
        --domains $domains --seeds $seeds

    echo "Done: trainseed $label"
}
