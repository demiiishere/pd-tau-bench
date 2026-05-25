#!/bin/bash
# ===========================================================================
# 多种子评估: 同一模型在 temperature>0 下独立重跑 N 次, 估计 pass@1 的
#            均值 ± 标准差 (论文鲁棒性检查). 一个作业 = 一个模型 × N 个种子.
#
# 用法:
#   bash experiments/run_eval_multiseed.sh <merged_model_dir> <label> [domain ...]
#
# 例:
#   # 零样本基线 (直接用 base 4B)
#   bash experiments/run_eval_multiseed.sh \
#       /user/zhujiatong/models/Qwen3-4B  zeroshot_4b  retail airline telecom
#   # 旗舰: 32B BoN-SFT + PD-DPO
#   bash experiments/run_eval_multiseed.sh \
#       outputs/models/e3c_sft_32b_bon_dpo_dpo/merged  e3c_32b_bondpo
#
# 可调环境变量:
#   SEEDS="0 1 2 3 4"   要跑的种子(独立重跑)列表
#   TEMPERATURE=0.7     agent 采样温度(用户模拟器固定为 0, 见 generate_baseline)
#   PORT=8010           推理服务端口
#   MAX_STEPS=30        单回合最大交互步数
#   DRY_RUN=1           只打印计划, 不真正运行
#
# 提交方式: 与 run_e3c_32b_bon_dpo.sh 等现有作业脚本完全一致 —— 它是一个
#   自包含的 bash 作业脚本, 申请 1 张 GPU, 按你提交 run_e3c 的同样方式提交即可.
#   建议每个模型提交一个作业; 若平台有墙钟上限, 可用 SEEDS 拆分成多个作业,
#   最后单独跑 aggregate_multiseed.py 汇总(它解耦, 可随时单独运行).
# ===========================================================================

set -euo pipefail

MERGED_MODEL="${1:?用法: run_eval_multiseed.sh <merged_model_dir> <label> [domain ...]}"
LABEL="${2:?需要第2个参数: 实验标签(如 e3c_32b_bondpo)}"
shift 2
DOMAINS=("$@")
[ ${#DOMAINS[@]} -eq 0 ] && DOMAINS=(retail airline telecom)

SEEDS="${SEEDS:-0 1 2 3 4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
PORT="${PORT:-8010}"
MAX_STEPS="${MAX_STEPS:-30}"

PD_DIR="/user/zhujiatong/pd-tau-bench"
VENV="/user/zhujiatong/envs/eval-4b/bin/activate"
LOG_DIR="$PD_DIR/logs"
RESULTS_BASE="$PD_DIR/data/results/multiseed"

# 允许相对路径(相对 PD_DIR)
case "$MERGED_MODEL" in
    /*) : ;;
    *) MERGED_MODEL="$PD_DIR/$MERGED_MODEL" ;;
esac

echo "[job-bootstrap] date=$(date) host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[config] model=$MERGED_MODEL"
echo "[config] label=$LABEL  domains=${DOMAINS[*]}"
echo "[config] seeds=[$SEEDS]  temperature=$TEMPERATURE  port=$PORT  max_steps=$MAX_STEPS"
echo "[config] results -> $RESULTS_BASE/$LABEL/seed_<S>/"

[ -d "$MERGED_MODEL" ] || { echo "ERROR: 模型目录不存在: $MERGED_MODEL" >&2; exit 1; }

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[dry-run] 计划: 启动 1 个服务, 对每个 seed × domain 跑 generate_baseline(test 集)"
    for S in $SEEDS; do
        for D in "${DOMAINS[@]}"; do
            echo "[dry-run]   seed=$S domain=$D -> $RESULTS_BASE/$LABEL/seed_${S}/${D}/"
        done
    done
    exit 0
fi

source "$VENV"
export PYTHONPATH="$PD_DIR"
export no_proxy="*" NO_PROXY="*"
mkdir -p "$LOG_DIR"
source "$PD_DIR/experiments/lib_sft_eval.sh"

SERVED=$(basename "$MERGED_MODEL")

# 启动一个推理服务; 所有种子查询同一个服务.
# temperature>0 => agent 每次采样独立, 因此 N 次重跑是 N 次独立抽样.
start_server "$MERGED_MODEL" "$PORT" "$LOG_DIR/multiseed_${LABEL}_server.log"
trap 'kill ${SERVER_PID:-0} 2>/dev/null || true' EXIT

export OPENAI_API_BASE="http://localhost:${PORT}/v1"
export OPENAI_API_KEY="fake"

cd "$PD_DIR"
for S in $SEEDS; do
    OUTDIR="$RESULTS_BASE/$LABEL/seed_${S}"
    for D in "${DOMAINS[@]}"; do
        echo "=== [$LABEL] seed=$S domain=$D temp=$TEMPERATURE ==="
        # generate_baseline 会跳过已存在的结果文件, 因此作业可安全断点续跑.
        python -m src.data_generation.generate_baseline \
            --domain "$D" \
            --split test \
            --model "openai/${SERVED}" \
            --temperature "$TEMPERATURE" \
            --max-steps "$MAX_STEPS" \
            --output-dir "$OUTDIR" \
            2>&1 | tee "$LOG_DIR/multiseed_${LABEL}_seed${S}_${D}.log"
    done
done

kill "${SERVER_PID:-0}" 2>/dev/null || true
trap - EXIT

echo "=== 汇总: $LABEL ==="
python "$PD_DIR/experiments/aggregate_multiseed.py" \
    --results-base "$RESULTS_BASE" \
    --label "$LABEL" \
    --domains "${DOMAINS[@]}" \
    --seeds $SEEDS

echo "Done: multiseed $LABEL"
