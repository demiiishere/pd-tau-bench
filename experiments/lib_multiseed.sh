#!/bin/bash
# ===========================================================================
# 多种子评估共享逻辑 (类比 lib_sft_eval.sh). source 本文件, 不要直接运行.
#
# 调用方(无参数作业脚本)需设置:
#   MS_MODEL    merged 模型目录 (绝对路径, 或相对 pd-tau-bench 根目录)
#   MS_LABEL    实验标签 (结果写到 data/results/multiseed/<MS_LABEL>/)
# 可选 (环境变量, 有默认值):
#   MS_DOMAINS="retail airline telecom"   SEEDS="0 1 2 3 4"
#   TEMPERATURE=0.7   PORT=8010   MAX_STEPS=30   DRY_RUN=1
#
# 原理: 起 1 个推理服务, 在 temperature>0 下对同一模型独立重跑 N 次
#       (每次 agent 采样独立 => N 次 = N 个独立抽样, 用于估 pass@1 mean±std).
# ===========================================================================

run_multiseed() {
    local PD_DIR="/user/zhujiatong/pd-tau-bench"
    local VENV="/user/zhujiatong/envs/eval-4b/bin/activate"
    local LOG_DIR="$PD_DIR/logs"
    local RESULTS_BASE="$PD_DIR/data/results/multiseed"

    local model="${MS_MODEL:?lib_multiseed: 需要设置 MS_MODEL}"
    local label="${MS_LABEL:?lib_multiseed: 需要设置 MS_LABEL}"
    local domains="${MS_DOMAINS:-retail airline telecom}"
    local seeds="${SEEDS:-0 1 2 3 4}"
    local temperature="${TEMPERATURE:-0.7}"
    local port="${PORT:-8010}"
    local max_steps="${MAX_STEPS:-30}"

    case "$model" in /*) : ;; *) model="$PD_DIR/$model" ;; esac

    echo "[job-bootstrap] date=$(date) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "[config] model=$model"
    echo "[config] label=$label  domains=$domains"
    echo "[config] seeds=[$seeds]  temperature=$temperature  port=$port  max_steps=$max_steps"
    echo "[config] results -> $RESULTS_BASE/$label/seed_<S>/"

    [ -d "$model" ] || { echo "ERROR: 模型目录不存在: $model" >&2; exit 1; }

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry-run] 计划: 启动 1 个服务, 对每个 seed × domain 跑 generate_baseline(test 集)"
        local s d
        for s in $seeds; do for d in $domains; do
            echo "[dry-run]   seed=$s domain=$d -> $RESULTS_BASE/$label/seed_${s}/${d}/"
        done; done
        return 0
    fi

    source "$VENV"
    export PYTHONPATH="$PD_DIR"
    export no_proxy="*" NO_PROXY="*"
    mkdir -p "$LOG_DIR"
    source "$PD_DIR/experiments/lib_sft_eval.sh"

    local served; served=$(basename "$model")

    # 起 1 个服务, 所有种子查询同一个服务
    start_server "$model" "$port" "$LOG_DIR/multiseed_${label}_server.log"
    trap 'kill ${SERVER_PID:-0} 2>/dev/null || true' EXIT

    export OPENAI_API_BASE="http://localhost:${port}/v1"
    export OPENAI_API_KEY="fake"
    cd "$PD_DIR"

    local S D
    for S in $seeds; do
        for D in $domains; do
            echo "=== [$label] seed=$S domain=$D temp=$temperature ==="
            # generate_baseline 跳过已存在结果文件 => 作业可断点续跑
            python -m src.data_generation.generate_baseline \
                --domain "$D" --split test \
                --model "openai/${served}" \
                --temperature "$temperature" --max-steps "$max_steps" \
                --output-dir "$RESULTS_BASE/$label/seed_${S}" \
                2>&1 | tee "$LOG_DIR/multiseed_${label}_seed${S}_${D}.log"
        done
    done

    kill "${SERVER_PID:-0}" 2>/dev/null || true
    trap - EXIT

    echo "=== 汇总: $label ==="
    python "$PD_DIR/experiments/aggregate_multiseed.py" \
        --results-base "$RESULTS_BASE" --label "$label" \
        --domains $domains --seeds $seeds

    echo "Done: multiseed $label"
}
