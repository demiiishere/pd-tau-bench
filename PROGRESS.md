# PD-τ-bench 项目进度

## TL;DR

**Phase A（本地数据生成）基础工作全部完成，4 个关键问题已修复，准备跑全量数据。**
核心验证：在 retail task 0 上，Predictive Decoding (reward=**1.0**) vs 标准 greedy decoding (reward=**0.0**)。
下一步是跑全量数据生成（train split），然后租 GPU 做 Phase B 训练。

---

## 已完成的工作

### 环境搭建
- 创建 conda 环境 `pd-tau-bench`（Python 3.11）
- 克隆并安装 `tau2-bench`（sierra-research，支持 retail / airline 两个域）
- 配置 DashScope API（Qwen-Plus，OpenAI 兼容格式）

### 核心实现（Phase A）

| 模块 | 文件 | 状态 |
|------|------|------|
| 环境 fork/restore | `src/predictive_decoding/tau_bench_adapter.py` | ✅ 测试通过 |
| Train/Test task split | `configs/task_splits.json` + `load_task_split()` | ✅ 新增（seed=42）|
| Turn-level PD 主循环 | `src/predictive_decoding/core.py` | ✅ 含自适应温度、跳过重复候选 |
| Value function | `src/predictive_decoding/value_function.py` | ✅ 正常工作 |
| Fork 测试脚本 | `src/predictive_decoding/test_env_fork.py` | ✅ 3/3 通过 |
| Baseline 生成 | `src/data_generation/generate_baseline.py` | ✅ 含 --split 和时间追踪 |
| PD 轨迹生成 | `src/data_generation/generate_trajectories.py` | ✅ 含 --split 支持 |
| Best-of-N 生成 | `src/data_generation/generate_bon.py` | ✅ 新增 |
| SFT/DPO 数据集构建 | `src/data_generation/build_dataset.py` | ✅ DPO 格式已修复，含 --split |
| 轨迹质量检查 | `src/data_generation/inspect_trajectories.py` | ✅ 正常 |
| Phase B 训练框架 | `src/training/sft_train.py` + `dpo_train.py` | 框架已写，等 GPU |
| Phase B 评估框架 | `src/evaluation/eval_on_tau_bench.py` + `analysis.py` | 框架已写，等 GPU |

### Session 2 修复（2026-03-03）

**修复了 4 个全量数据生成前的关键问题：**

#### 1. Train/Test Split（防止数据污染）
- 新增 `configs/task_splits.json`（seed=42，retail 80/34，airline 35/15）
- 所有生成脚本加 `--split train|test|all` 参数，默认 `train`
- 训练数据只用 train split，test split 留给评估

#### 2. Token/时间追踪
- `run_pd_episode()` 和 `run_baseline_episode()` 现在输出 `wall_time_s`、`api_calls_approx`
- PD 还输出 `pd_steps_count`、`pd_steps_skipped_count`
- 用于 Phase B 的成本效率分析（Pareto frontier）

#### 3. Best-of-N（BoN）基线
- 新增 `src/data_generation/generate_bon.py`
- BoN：对每个任务采样 N 个 temperature=0.8 的 episode，取最好的（oracle reward）
- 这是 **最关键的对比基线 E2**：和 PD 使用相同 inference budget，但无 lookahead
- 输出 oracle_reward、avg_reward、success_count，以及 best_conversation（用于 SFT）

#### 4. DPO 数据格式修复
- 旧格式（错误）：`"chosen": "[TOOL] exchange_delivered_order_items({...})"` — 字符串，丢失结构
- 新格式（正确）：
  ```json
  {
    "chosen": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
    }
  }
  ```
- 完整保留 tool_calls 结构，可直接用于 TRL DPO trainer

#### 5. 优化：自适应温度 + 跳过重复候选（from potential_optimization.md）
- **自适应温度**：如果前几个候选太相似（avg_sim > 0.9），自动提高 temperature（最高 1.2），增加多样性
- **跳过重复候选**：如果超过一半候选对相似度 > 0.95，认为该步骤确定性太高，直接用 greedy（节省 K×H 次 API 调用）

---

## 验证结果（1 个 retail task）

```
Task 0（用户要求同时换两件商品：机械键盘 + 智能温控器）

Baseline (greedy):   final_reward = 0.0   ← 任务失败
PD (K=3, H=2):       final_reward = 1.0   ← 任务成功 ✓
```

PD 在步骤 6（展示换货选项）选择了格式更清晰的候选，让用户确认更准确，最终 agent 正确执行了换货。

---

## 一个可读的验证例子

### 关键决策点（Step 6）

PD 对 3 个候选做了 2 步 foresight：
```
候选 0 (score=0.695): "Item ID: 7706410293 - Clicky switches, no backlight..."
候选 1 (score=0.820) ★ 被选中: "Clicky switches, no backlight, full size: $269.16 (item ID: 7706410293)"
候选 2 (score=0.695): 和候选 0 基本相同
```

候选 1 把价格和 item ID 放在同一行，用户确认更直接 → agent 正确调用 `exchange_delivered_order_items` → reward=1.0。

---

## Todo（待完成）

### Phase A — 全量数据生成（下一步立刻要做）

```bash
conda activate pd-tau-bench
cd ~/pd-tau-bench
export DASHSCOPE_API_KEY=你的key

# 1. 生成 PD 轨迹（train split，默认）
python -m src.data_generation.generate_trajectories \
    --domain retail airline --K 5 --H 2 \
    --model openai/qwen-plus --num-trials 3 --max-concurrency 5

# 2. 生成 BoN 基线（同样 train split）
python -m src.data_generation.generate_bon \
    --domain retail airline --N 5 \
    --model openai/qwen-plus --max-concurrency 5

# 3. 构造训练数据集
python -m src.data_generation.build_dataset
```

- [ ] 预期产出：~240 SFT(PD) + ~960 DPO + ~240 SFT(BoN)，耗时 6-10h，花费 ¥50-100
- [ ] 运行完后检查 score gap 分布（平均 gap > 0.05 即可）

### Phase B — 模型训练（需租 GPU）

- [ ] 租 AutoDL / Featurize 的 A100 40G（约 ¥3-5/小时）
- [ ] 跑 4 组实验：

  | 实验 | 模型 | 训练数据 | 对应基线 |
  |------|------|---------|----|
  | E1 | Qwen3-8B（无微调） | — | zero-shot |
  | E2 | Qwen3-8B + SFT | BoN 成功轨迹 | BoN oracle |
  | E3 | Qwen3-8B + SFT | PD 成功轨迹 | PD selection |
  | E4 | Qwen3-8B + SFT + DPO | PD 数据 + DPO 偏好对 | PD+DPO |

- [ ] 评估所有模型，报告 pass@1（test split）

### 潜在优化（供参考，见 potential_optimization.md）

- **LLM-as-Judge**：在 value function 里加 Qwen 打分（成本 +30-50%，效果最好）- 等看全量数据的 score gap 分布再决定
- **Task filtering**：先跑完全量 baseline，按 reward/turn 数筛高潜力任务，对它们用更大 K/H
- **更多实验**：H=3 vs H=2，K=8 vs K=5，temperature=1.0 vs 0.8

---

## 文件结构

```
pd-tau-bench/
├── tau2-bench/                    # tau2-bench 源码（已安装）
├── configs/
│   ├── generation_config.yaml     # 生成参数配置
│   └── task_splits.json           # ← 新增：train/test split（seed=42）
├── src/
│   ├── predictive_decoding/
│   │   ├── core.py                # PD 主循环（含自适应温度、skip-identical）
│   │   ├── tau_bench_adapter.py   # fork/restore + load_task_split()
│   │   ├── value_function.py      # 打分函数
│   │   └── test_env_fork.py       # fork 测试
│   ├── data_generation/
│   │   ├── generate_trajectories.py  # PD 轨迹生成（支持 --split）
│   │   ├── generate_baseline.py      # baseline 生成（支持 --split，含 time tracking）
│   │   ├── generate_bon.py           # ← 新增：BoN 基线生成
│   │   ├── build_dataset.py          # SFT/DPO 构建（DPO 格式修复，支持 --split）
│   │   └── inspect_trajectories.py   # 轨迹质量检查
│   ├── training/                  # Phase B：SFT + DPO（框架已写）
│   └── evaluation/                # Phase B：评估 + 分析（框架已写）
├── data/
│   ├── raw_trajectories/retail/   # 已生成 1 条 PD + 1 条 baseline
│   ├── sft_dataset/               # 已生成 1 条 SFT 样本
│   └── dpo_dataset/               # 已生成 1 条 DPO 对（旧格式，需重建）
├── potential_optimization.md      # 优化方案参考（LLM-judge, task filtering, adaptive temp）
├── CLAUDE_CONTEXT.md              # 给 Claude 的技术备忘录
├── PROGRESS.md                    # 本文件
└── scripts/
    ├── step1_setup.sh
    └── step2_generate.sh
```
