# PD-τ-bench 项目进度

## TL;DR

**Phase A 完成，训练策略已确定，准备上服务器跑 Phase B。**
核心策略：小数据体制 (r=8, q/v-only LoRA, 1 epoch) + 通用数据混合 + 新增 E2+ (BoN+episode-DPO) 与 E4 (PD+turn-DPO) 对比。

| | retail | airline | 合计 |
|--|--|--|--|
| SFT (PD) | 133 | 64 | **197** |
| SFT (BoN) | 196 | 99 | **295** |
| SFT (baseline) | 41 | 20 | 61 |
| DPO pairs | 185 | 128 | **313** |

PD pass@1：retail 55.4%，airline 62.7%（均高于同 domain BoN pass@1）

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
| SFT/DPO 数据集构建 | `src/data_generation/build_dataset.py` | ✅ 多域修复，数据集已生成 |
| 轨迹质量检查 | `src/data_generation/inspect_trajectories.py` | ✅ 正常 |
| Phase B 训练框架 | `src/training/sft_train.py` + `dpo_train.py` | ✅ 脚本已修复，等 GPU |
| Phase B 评估框架 | `src/evaluation/eval_on_tau_bench.py` + `analysis.py` | ✅ 框架已写，等 GPU |

### Session 5 更新（2026-03-04）— Phase B 训练策略确定

#### 1. 训练脚本超参修订（小数据体制）

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| LoRA r | 16 | **8** | 可训练参数少 → 过拟合风险低 |
| target_modules | q/k/v/o | **q/v only** | 进一步减少参数量 |
| lora_dropout | 0.05 | **0.1** | 更强正则化 |
| SFT epochs | 3 | **1** | 多 epoch = 反复背诵 |
| SFT lr | 2e-5 | **5e-5** | 少数据需要更大步长 |
| grad_accum | 8 | **4** | 更多梯度更新步 |
| DPO beta | 0.1 | **0.3** | 防小数据下模型漂移 |
| DPO lr | 5e-6 | **1e-6** | DPO 超参非常敏感 |
| DPO max_prompt_len | 4096 | **6144** | 更充分的历史上下文 |

#### 2. 通用数据混合策略

`sft_train.py` 新增 `--general-data` 参数，支持混入通用 function calling 数据（推荐 glaive-function-calling-v2 3k 条）。三组 SFT 实验 (E1/E2/E3) 必须用完全相同的通用数据，唯一变量是域内数据。

#### 3. 实验矩阵扩展：新增 E2+

新增 E2+（BoN-SFT + 训练 episode-level DPO）作为 E4 的消融对照。
核心 claim：E4 > E2+ 说明**turn-level 偏好信号优于 episode-level**。

| 编号 | 训练数据 | 目的 |
|------|---------|------|
| E0 | — | zero-shot baseline |
| E1 | standard ~61 条 | standard SFT baseline |
| E2 | BoN ~295 条 | BoN SFT baseline |
| **E2+** | BoN + episode-DPO | BoN+DPO baseline（新增） |
| E3 | PD ~197 条 | PD SFT |
| **E4** | PD + turn-DPO | **我们的方法** |

#### 4. build_dataset.py 新增 BoN episode-level DPO

新增 `build_bon_dpo_dataset()`：对每个有成功/失败对的 task，抽取一对 BoN episode，
用各自第一个 agent action 构建偏好对。输出到 `data/dpo_dataset/train_bon_episode.jsonl`。

```bash
# 重新 build（新增输出 train_bon_episode.jsonl）
python -m src.data_generation.build_dataset
```

---

### Session 4 更新（2026-03-04）— Phase A 收尾，数据集就绪

#### 1. 全量数据生成完成

| 域 | PD 文件 | BoN 文件 | Baseline |
|----|---------|---------|---------|
| retail | 240（80×3）✅ | 400（80×5）✅ | 81 ✅ |
| airline | 102（35×3，差3）✅ | 174（35×5，差1）✅ | 33 ✅ |

#### 2. 数据质量报告

**PD (K=5, H=2)**

| 指标 | retail | airline |
|------|--------|---------|
| Pass@1 | 55.4% | 62.7% |
| Oracle (pass@any) | 82.5% | 85.7% |
| 全失败 tasks | 14/80 | 5/35 |
| 真实 PD 决策比例 (gap>0.05) | 27% | 35% |
| 真实 PD gap avg/median | 0.108 / 0.083 | 0.120 / 0.100 |
| Token/episode avg | 963k (92% OH) | 935k (92% OH) |

**BoN (N=5)**

| 指标 | retail | airline |
|------|--------|---------|
| Pass@1 (avg) | 48.5% | 57.7% |
| Oracle | 76.2% | 88.6% |
| 全失败 tasks | 19/80 | 4/35 |
| Token/sample avg | ~89k | ~83k |

PD pass@1 在两个域均高于 BoN pass@1（retail +7pp，airline +5pp）。

#### 3. build_dataset.py 三个 bug 修复

**Bug 1：BoN oracle_reward 字段名错误**
- 旧代码用 `final_reward` 筛选 BoN 成功轨迹，但 BoN summary 文件用 `oracle_reward`
- 结果：SFT(BoN) 始终为 0 条
- 修复：按 source_filter 选择对应字段名

**Bug 2：跨 domain task ID 泄漏**
- 旧代码合并 retail+airline 的 allowed_task_ids（共 88 个）
- retail 测试集的某些 task ID 恰好与 airline 训练集重叠，导致测试集数据混入训练集
- 修复：每个 domain 独立处理，各自用自己的 allowed_task_ids

**Bug 3：BoN 只用 oracle best，浪费数据**
- 旧代码从 `_bon_summary.json` 只取 best_conversation（1条/task）→ 61 条
- 修复：改用所有 `_bon_n*.json` 成功的个人 sample → 196 条（retail）

#### 4. 正确的 build_dataset 命令

```bash
# 必须这样调用，否则有跨域 ID 泄漏问题
conda run -n pd-tau-bench python3.11 -m src.data_generation.build_dataset
# 默认已经是 --domains retail airline，分别处理再合并，无需额外参数
```

**最终数据集（`data/sft_dataset/` + `data/dpo_dataset/`）：**

| 文件 | 条数 | 用于 |
|------|------|------|
| `train.jsonl` | 197 | E3: SFT on PD |
| `train_bon.jsonl` | 295 | E2: SFT on BoN |
| `train_baseline.jsonl` | 61 | 参考 |
| `dpo_dataset/train.jsonl` | 313 | E4: DPO (gap≥0.10) |

---

### Session 3 更新（2026-03-03）

#### 1. Value Function 大改进（True PD Rate: 6% → 17%）

**根本问题**：H=2 的 foresight 太短，所有候选在前 2 步都调同一个 tool call，导致 score gap ≈ 0（94% 的步骤 gap=0）。

**解决方案**：5 个信号，全部在 foresight delta（trajectory[foresight_start_idx:]）上计算，而不是全量轨迹：

```
score = 0.35 × delta_progress    # 新增：foresight 中调了多少期望工具
      + 0.25 × foresight_health  # 新增：tool 调用错误/冗余惩罚
      + 0.15 × user_sentiment    # 新增：用户回复中的正/负面关键词
      + 0.15 × termination       # 保留（权重降低）
      + 0.10 × env_assertions    # 保留（权重降低）
```

- `_compute_delta_progress()`：base=0.5（中性），调到期望工具 → 最高 1.0
- `_compute_foresight_health()`：无错误 → 1.0，每个错误/冗余扣分
- `_compute_user_sentiment()`：正面词 → 0.75，负面词 → 0.25，中性 → 0.5

**效果**：True PD rate 从 6% 提升到 17%（gap > 0.05 的步骤比例）。

#### 2. BoN 生成完成（80/80 retail tasks）

- oracle_reward@5 = 0.76（76% 任务至少 1 次成功）
- avg pass@1 = 0.49
- 每 task (N=5) token 总计：avg=447k, median=390k
- 每个 sample 平均：89k tokens
- 关键对比：PD pass@1（57%）> BoN pass@1（49%），但 PD 每 episode 消耗 ~970k tokens（91% 是 overhead）

#### 3. PD 数据删除并重新生成

旧数据（有 greedy_fb bug + 旧 value function）全部删除，使用新代码（value_function.py 改进 + greedy fallback 修复）重新生成。
重新生成命令（PID 96488 在后台运行中）：
```bash
DASHSCOPE_API_KEY=xxx nohup python3.11 -m src.data_generation.generate_trajectories \
    --domain retail --K 5 --H 2 --num-trials 3 \
    --model openai/qwen-plus --max-concurrency 5 > logs/pd_regen.log 2>&1 &
```

#### 4. 论文策略分析

- **E1**（zero-shot）+ **E2**（SFT BoN）+ **E3**（SFT PD）= 可发表的最小组合
- **E4**（SFT+DPO）：DPO 数据质量偏弱（~390 对，67% gap<0.10），预期提升仅 0-3pp，风险高
- Phase B（GPU 训练）是关键缺失环节，应尽快完成

#### 5. GPU 策略：RTX 5090 (32GB) 无需量化

- Qwen3-8B bf16 权重 ≈ 16GB
- LoRA 参数 + 优化器 ≈ 2GB
- activations（开启 gradient_checkpointing）≈ 2-4GB
- 总计 ~20GB，32GB 完全够用，**无需量化**
- 仍需 `gradient_checkpointing=True` 以安全处理 8192 seq_len

#### 6. 训练脚本修复

`sft_train.py` 修复内容：
- 添加 `gradient_checkpointing=True` + `gradient_checkpointing_kwargs={"use_reentrant": False}`
- 添加 `trust_remote_code=True` + `model.config.use_cache = False`
- 正确处理 `messages` 列：用 `tokenizer.apply_chat_template()` 转为文本，再用 `dataset_text_field="text"`
- 添加 `--source` 过滤参数（pd/bon/baseline）
- 添加 eval split（默认 5%）
- 添加 `lr_scheduler_type="cosine"`，保存 tokenizer

`dpo_train.py` 修复内容：
- 移除独立 `ref_model`（PEFT 模式下 TRL 自动用禁用 LoRA 的同一模型作参考，省 16GB VRAM）
- 添加 `gradient_checkpointing=True`、`trust_remote_code=True`
- 正确预处理 DPO 格式：`prompt + chosen/rejected` 用 chat_template 转成字符串对
- 添加 `--min-score-gap` 过滤参数（默认 0.05）
- 添加 eval split

---

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

### Phase A — ✅ 完成

数据集已就绪：`data/sft_dataset/` + `data/dpo_dataset/`

### Phase B — 模型训练（当前阶段，上服务器）

服务器配置见 `CLAUDE_CONTEXT.md` → "服务器环境配置" 小节。

**训练策略**：见 `CLAUDE_CONTEXT.md` → "Phase B 训练策略" 小节（r=8, q/v only, 1 epoch, 通用数据混合）。

- [ ] 上传数据集到服务器（`data/sft_dataset/` + `data/dpo_dataset/`）
- [ ] 配置服务器环境（conda、依赖、Qwen3-8B 模型下载）
- [ ] 重新 build dataset（新增 `train_bon_episode.jsonl`）
- [ ] （可选）准备通用数据 glaive-function-calling-v2 ~3k 条 → `data/general/glaive_3k.jsonl`
- [ ] 跑预实验（先跑 E2 debug 训练流程，确认无 OOM / loss 正常）
- [ ] 跑正式实验（按优先级，**所有 SFT 用相同 LoRA 和超参**）：

  | 实验 | 说明 | 优先级 |
  |------|------|------|
  | E0 | zero-shot（直接评估，无训练）| ★★★ |
  | E2 | SFT on BoN 295 条 | ★★★ |
  | E3 | SFT on PD 197 条 | ★★★ |
  | E1 | SFT on standard 61 条 | ★★ |
  | E2+ | E2 → DPO on BoN episode-level pairs | ★★ |
  | E4 | E3 → DPO on PD turn-level pairs | ★★ |

- [ ] 评估所有模型（retail+airline test split，`eval_on_tau_bench.py`），报告 pass@1
- [ ] **核心对比**：E2+ vs E4（turn-level vs episode-level DPO），E2 vs E3（PD vs BoN SFT）

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
│   ├── raw_trajectories/
│   │   ├── retail/    # 240 PD + 400 BoN + 81 baseline
│   │   └── airline/   # 102 PD + 174 BoN + 33 baseline
│   ├── sft_dataset/   # train.jsonl(197) + train_bon.jsonl(295) + train_baseline.jsonl(61)
│   └── dpo_dataset/   # train.jsonl(313 pairs, gap≥0.10)
├── potential_optimization.md      # 优化方案参考（LLM-judge, task filtering, adaptive temp）
├── CLAUDE_CONTEXT.md              # 给 Claude 的技术备忘录
├── PROGRESS.md                    # 本文件
└── scripts/
    ├── step1_setup.sh
    └── step2_generate.sh
```
