# PD-τ-bench 项目进度

## TL;DR

**已完成 Phase A（本地数据生成）的全部基础工作。**
核心验证：在 retail task 0 上，Predictive Decoding (reward=**1.0**) vs 标准 greedy decoding (reward=**0.0**)。
下一步是跑全量数据生成，然后租 GPU 做 Phase B 训练。

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
| Turn-level PD 主循环 | `src/predictive_decoding/core.py` | ✅ 测试通过 |
| Value function | `src/predictive_decoding/value_function.py` | ✅ 正常工作 |
| Fork 测试脚本 | `src/predictive_decoding/test_env_fork.py` | ✅ 3/3 通过 |
| Baseline 生成 | `src/data_generation/generate_baseline.py` | ✅ 测试通过 |
| PD 轨迹生成 | `src/data_generation/generate_trajectories.py` | ✅ 测试通过 |
| SFT/DPO 数据集构建 | `src/data_generation/build_dataset.py` | ✅ 测试通过 |
| 轨迹质量检查 | `src/data_generation/inspect_trajectories.py` | ✅ 正常 |
| Phase B 训练框架 | `src/training/sft_train.py` + `dpo_train.py` | 框架已写，等 GPU |
| Phase B 评估框架 | `src/evaluation/eval_on_tau_bench.py` + `analysis.py` | 框架已写，等 GPU |

### 验证结果（1 个 retail task）

```
Task 0（用户要求同时换两件商品：机械键盘 + 智能温控器）

Baseline (greedy):   final_reward = 0.0   ← 任务失败
PD (K=3, H=2):       final_reward = 1.0   ← 任务成功 ✓
```

PD 在 9 个决策步骤中，步骤 6 和步骤 8 产生了有意义的候选区分：
- Step 6：gap = 0.125（PD 选择了更清晰的商品展示方式）
- Step 8：gap = 0.090
- 产出 1 个有效 DPO 偏好对（gap > 0.1）

---

## 一个可读的验证例子

### 任务描述
用户 Yusuf Rossi 要求将订单 #W2378156 中的两件商品换货：
- 机械键盘（linear → clicky，保留 RGB 背光和 full-size）
- 智能温控器（Apple HomeKit → Google Home 兼容）

### Baseline 的失败过程

```
[user]      Hi! I'd like to exchange keyboard and thermostat from order #W2378156...
[assistant] Let me verify your identity first. What's your email?
[user]      I don't remember my email, but I'm Yusuf Rossi, zip 19122
[assistant] [tool] find_user_id_by_name_zip → "yusuf_rossi_9620"
[assistant] [tool] get_order_details → order found
[assistant] Great! I've found the available options...
[user]      I'll go with keyboard #7706410293 and thermostat #7747408585. Proceed.
[assistant] Let me calculate the price difference...   ← 说要计算但没有执行换货
[user]      Yes I confirm, please proceed.
...
[终止]  user_stop，final_reward = 0.0  ← DB 状态未变，换货未完成
```

Baseline 在用户确认后，agent 说了"Let me calculate..."但没有调用 `exchange_delivered_order_items`，任务失败。

### PD 的成功过程

在**步骤 6**（agent 需要向用户展示可用换货选项），PD 对 3 个候选做了 2 步 foresight：

```
候选 0 (score=0.695): "Great! I've found the available options..."
  → 格式：Item ID: 7706410293 - Clicky switches, no backlight...

候选 1 (score=0.820) ★ 被选中:  "Great! I've found available options..."
  → 格式：Clicky switches, no backlight, full size: $269.16 (item ID: 7706410293) ✓

候选 2 (score=0.695): 和候选 0 基本相同
```

PD 的 foresight 发现：候选 1 的展示格式更清晰（价格和 item ID 在同一行），导致用户 2 步后能更准确地确认，进而 agent 正确调用了 `exchange_delivered_order_items`。

```
[user]      Hi! I'd like to exchange keyboard and thermostat...
[assistant] [tool] find_user_id_by_name_zip → "yusuf_rossi_9620"
[assistant] [tool] get_order_details → order found
[assistant] [tool] get_product_details (keyboard)
[assistant] [tool] get_product_details (thermostat)
[assistant] ★ [候选1] Great! Here are available options: Clicky, full size, $269.16...
[user]      Yes, proceed with keyboard #7706410293 and thermostat #7747408585.
[assistant] [tool] exchange_delivered_order_items → exchange processed ✓
[assistant] Your exchange has been successfully processed!
[user]      Great, thanks! ###STOP###
[终止]  user_stop，final_reward = 1.0  ← DB 状态匹配，换货完成 ✓
```

---

## Todo（待完成）

### Phase A — 全量数据生成（下一步立刻要做）

- [ ] **全量运行**（耗时约 6-10 小时，费用约 ¥50-100）
  ```bash
  conda activate pd-tau-bench
  cd ~/pd-tau-bench
  export DASHSCOPE_API_KEY=你的key
  python -m src.data_generation.generate_trajectories \
      --domain retail airline --K 5 --H 2 \
      --model openai/qwen-plus --num-trials 3 --max-concurrency 5
  ```
- [ ] 运行完后构造训练数据集：
  ```bash
  python -m src.data_generation.build_dataset
  ```
- [ ] 检查数据量（预期：~300 条 SFT + ~1200 条 DPO）

### Phase B — 模型训练（需租 GPU）

- [ ] 租 AutoDL / Featurize 的 A100 40G（约 ¥3-5/小时）
- [ ] 把 `data/` 目录和 `src/` 代码传到 GPU 机器
- [ ] 跑 4 组实验：
  | 实验 | 模型 | 训练数据 |
  |------|------|---------|
  | E1 | Qwen3-8B（无微调） | — |
  | E2 | Qwen3-8B + SFT | baseline 成功轨迹 |
  | E3 | Qwen3-8B + SFT | PD 成功轨迹 |
  | E4 | Qwen3-8B + SFT + DPO | PD 数据 + DPO 偏好对 |
- [ ] 评估所有模型，报告 pass@1 和 pass@k

### 潜在优化方向

- **Value function**：当前平均 score gap 偏低（~0.024）。对于没有 `env_assertions` 的任务，action overlap 区分度有限。可以尝试用 LLM 打分（但会增加 API 成本）。
- **任务筛选**：顺序性强的任务（每步只有一个正确 tool call）几乎没有 PD 收益；关注 reward_basis 包含 COMMUNICATE 的任务（更多文本决策点）。
- **Temperature 调参**：当前 candidate_temperature=0.8 在某些步骤仍然产出重复候选，可试 1.0。

---

## 文件结构

```
pd-tau-bench/
├── tau2-bench/                    # tau2-bench 源码（已安装）
├── src/
│   ├── predictive_decoding/
│   │   ├── core.py                # PD 主循环
│   │   ├── tau_bench_adapter.py   # fork/restore + 环境创建
│   │   ├── value_function.py      # 打分函数
│   │   └── test_env_fork.py       # fork 测试
│   ├── data_generation/
│   │   ├── generate_trajectories.py  # PD 轨迹生成（主入口）
│   │   ├── generate_baseline.py      # baseline 生成
│   │   ├── build_dataset.py          # SFT/DPO 数据集构建
│   │   └── inspect_trajectories.py   # 轨迹质量检查
│   ├── training/                  # Phase B：SFT + DPO（框架已写）
│   └── evaluation/                # Phase B：评估 + 分析（框架已写）
├── data/
│   ├── raw_trajectories/retail/   # 已生成 1 条 PD + 1 条 baseline
│   ├── sft_dataset/               # 已生成 1 条 SFT 样本
│   └── dpo_dataset/               # 已生成 1 条 DPO 对
├── configs/generation_config.yaml
└── scripts/
    ├── step1_setup.sh
    └── step2_generate.sh
```
