# PD-τ-bench 项目进度

## TL;DR

**Phase B：E0/E2/E3/E4 评估完成（thinking ON 为正式结果）。**
评估策略：服务器跑 vLLM（无需外网），本地 Mac 通过 VS Code PORTS 端口转发连接 vLLM + 直接调 DashScope API。

### 当前评估结果（test split，3 trials）

#### Thinking ON（推荐，性能更好）

| 实验 | retail pass@1 | retail pass@3 | retail oracle | airline pass@1 | airline pass@3 | airline oracle |
|------|--------------|--------------|--------------|---------------|---------------|---------------|
| E0 zero-shot | 0.3235 | 0.1765 | 0.5000 | 0.1778 | 0.0 | 0.4000 |
| E2 BoN SFT | 0.3137 | 0.0588 | 0.5588 | 0.0889 | 0.0 | 0.2000 |
| E3 PD SFT | 0.3137 | 0.1471 | 0.4706 | 0.1111 | 0.0 | 0.2000 |

#### Thinking OFF（`enable_thinking=False` 训练 + 评估）

| 实验 | retail pass@1 | retail pass@3 | retail oracle | airline pass@1 | airline pass@3 | airline oracle |
|------|--------------|--------------|--------------|---------------|---------------|---------------|
| E0 zero-shot | 0.3137 | 0.0882 | 0.5000 | 0.0889 | 0.0 | 0.1333 |
| E2 BoN SFT | 0.2353 | 0.0882 | 0.3824 | 0.0889 | 0.0 | 0.2667 |
| E3 PD SFT | 0.2451 | 0.1176 | 0.4118 | 0.0667 | 0.0 | 0.2000 |
| E4 DPO (on E3) | 0.2549 | 0.0882 | 0.4412 | 0.0444 | 0.0 | 0.1333 |

**注意**：E4 基于 thinking OFF 的 SFT 模型训练，评估也在 thinking OFF 下进行，应与同列（thinking OFF）比较。
**结论：thinking ON 全面优于 thinking OFF，应使用 thinking ON 作为正式结果。**

**观察**：
- **Zero-shot 优于所有 SFT（thinking ON 列）**：Qwen3-8B 本身 function calling 能力强，小数据 SFT（197-295 条）不足以超越 zero-shot，反而轻微损坏通用推理能力（catastrophic forgetting 轻量版）
- **E3 retail pass@3（0.1471）> E2（0.0588）**：PD 轨迹质量更高，一致性更好，但仍低于 zero-shot（0.1765）
- **Airline 普遍很差**：zero-shot oracle 也只有 0.4，airline 本身对 Qwen3-8B 就难；SFT 数据量更少（64 条），提升空间不足
- **E4 DPO 基本没有改善**（与 E3 thinking OFF 比：retail +1pp，airline -2pp）：根本原因是 thinking 不匹配——SFT/DPO 训练时 `enable_thinking=False`，但最优性能需要 thinking ON；小数据 DPO（313 对）在已经退化的 SFT checkpoint 上很难有正向提升
- **核心问题（thinking mismatch）**：数据生成用的是 thinking ON 的 Qwen-Plus，SFT 训练时却用 `enable_thinking=False` → 模型学到了一个没有 thinking 步骤的行为克隆，inference 时如果开 thinking 则格式不匹配
- **下一步建议**：用 thinking ON 重新跑 E2/E3 SFT 训练（训练时不加 `enable_thinking=False`），并在 eval 时也用 thinking ON（去掉 `extra_body` 中的 disable），这样才能充分利用 Qwen3-8B 的推理能力

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

### Session 7 更新（2026-03-05）— 评估环境调试完成，完整工作流确立

#### 完整评估工作流（每次评估按此步骤执行）

**Step 1：在 VS Code 里连服务器并启动 vLLM**

```bash
# 在 VS Code 里打开 devspace-zhujiatong-pad-alter-193739，
# 然后在 VS Code 终端里执行：
source /user/zhujiatong/miniconda3/bin/activate   # 每次新终端必须先执行
cd /user/zhujiatong/pd-tau-bench
git pull
conda activate vllm-env
bash scripts/start_vllm.sh bon   # bon | pd | dpo
# 等待约 30s，看到 "Application startup complete" 即可
curl http://localhost:8001/v1/models   # 验证：应返回含 "finetuned" 的 JSON
```

**Step 2：在 VS Code 里开端口转发**

底部面板 → **PORTS（端口）** 选项卡 → **Forward a Port** → 输入 `8001`。
注意 VS Code 可能映射到不同的本地端口（如 59827），以 PORTS 面板显示的为准。

**Step 3：在本地 Mac 验证端口转发**

```bash
curl http://localhost:8001/v1/models   # 若端口映射到其他号，换成对应号
```

**Step 4：在本地 Mac 跑评估**

```bash
conda activate pd-tau-bench
cd /Users/zhujiatong/pd-tau-bench

# E0: 零样本 baseline（vLLM 直接 serve base model，不加 LoRA）
# 服务器启动时用：vllm serve /user/zhujiatong/models/Qwen3-8B --port 8001 ...（不带 --enable-lora）
python3.11 -m src.evaluation.eval_on_tau_bench \
    --domain retail \
    --split test \
    --agent-model "openai//user/zhujiatong/models/Qwen3-8B" \
    --vllm-url http://localhost:8001/v1 \
    --user-model openai/qwen-plus \
    --num-trials 3 \
    --output-dir outputs/results/zero_shot

# E2: BoN SFT（服务器先跑 bash scripts/start_vllm.sh bon）
python3.11 -m src.evaluation.eval_on_tau_bench \
    --domain retail \
    --split test \
    --agent-model openai/finetuned \
    --vllm-url http://localhost:8001/v1 \
    --user-model openai/qwen-plus \
    --num-trials 3 \
    --output-dir outputs/results/sft_bon
# airline 同理，换 --domain airline --output-dir outputs/results/sft_bon_airline

# E3: PD SFT（服务器先跑 bash scripts/start_vllm.sh pd）
python3.11 -m src.evaluation.eval_on_tau_bench \
    --domain retail \
    --split test \
    --agent-model openai/finetuned \
    --vllm-url http://localhost:8001/v1 \
    --user-model openai/qwen-plus \
    --num-trials 3 \
    --output-dir outputs/results/sft_pd

# E4: DPO（服务器先跑 bash scripts/start_vllm.sh dpo）
python3.11 -m src.evaluation.eval_on_tau_bench \
    --domain retail airline \
    --split test \
    --agent-model openai/finetuned \
    --vllm-url http://localhost:8001/v1 \
    --user-model openai/qwen-plus \
    --num-trials 3 \
    --output-dir outputs/results/sft_pd_dpo
```

#### 本次调试记录的问题与修复

1. **SSH 隧道方案失败**：Teleport ProxyCommand 不支持 `-N -L` 端口转发 → **改用 VS Code PORTS 面板端口转发**（VS Code 的连接本身就是 SSH，自带端口转发能力）

2. **脚本路径错误**：`scripts/start_vllm.sh` 中 `conda activate` 在非交互 bash 里报错（需要先 init）
   → 修复：改为 `eval "$(conda shell.bash hook)" && conda activate vllm-env`

3. **评估脚本路由冲突（最终修复）**：
   - 旧方案：`OPENAI_API_BASE=DashScope`（全局），agent 用 per-call `api_base=vLLM` override → 502（edge case 不生效）
   - 新方案（inverse pattern）：`OPENAI_API_BASE=vLLM`（全局，agent 走这里），DashScope 凭据通过 `user_model_args` 的 per-call `api_base`/`api_key` 传入 → 路由完全可控

4. **`generate_baseline.py` 新增 `user_model_args` 参数**：配合上述路由修复，让 user simulator 能指定 DashScope 凭据

5. **新增 `scripts/start_vllm.sh`**：封装 vLLM 启动命令，支持 `bon`/`pd` 两个变体，避免每次手打长命令

#### 服务器信息（公司 Teleport 服务器，非 AutoDL）

- SSH host：`devspace-zhujiatong-pad-alter-193739`（VS Code SSH config 里）
- 项目路径：`/user/zhujiatong/pd-tau-bench/`
- 模型路径：`/user/zhujiatong/models/Qwen3-8B`（base）
- LoRA 路径：`/user/zhujiatong/outputs_pd/sft_bon/final/`、`/user/zhujiatong/outputs_pd/sft_pd/final/`
- 训练 conda 环境：`pd-qwen3-8b`；推理 conda 环境：`vllm-env`
- 每次新终端需先执行：`source /user/zhujiatong/miniconda3/bin/activate`
- **无公网**：仅可用 HF mirror（`https://hf-mirror.com`）、清华/USTC 镜像源
- GPU：2× H100 80GB（当前 vLLM 只用单卡）

---

### Session 6 更新（2026-03-04）— E2/E3 SFT 训练完成，评估脚本修复

#### 1. SFT 训练结果（服务器 A100，1 epoch，LoRA r=8，alpha=16，dropout=0.1，q_proj+v_proj）

| 实验 | optimizer steps | train_loss(最终) | eval_loss | token_accuracy | 训练时间 |
|------|----------------|-----------------|-----------|---------------|---------|
| E2 (BoN, 295条) | 70 | 0.870 | 0.8485 | 0.7801 | ~168s |
| E3 (PD,  197条) | 47 | 0.841 | 0.7797 | 0.8038 | ~112s |

**关键观察**：PD 模型用更少的 step 达到更低 eval_loss 和更高 token_accuracy，说明 PD 轨迹质量更高——模型学起来更容易，预示 tau-bench 实际评估 PD 应优于 BoN。

**模型路径（本地）**：
- E2: `outputs_pd/sft_bon/final/`（adapter only，base: `/user/zhujiatong/models/Qwen3-8B`）
- E3: `outputs_pd/sft_pd/final/`

#### 2. 评估脚本修复（`src/evaluation/eval_on_tau_bench.py` 3 个 bug）

- **Bug 1（最严重）**：没用 `load_task_split()`，会跑全部任务而非 test split → 已修复
- **Bug 2**：pass@k 算的是 `any`（oracle），应为 `all` → 已修复；新增 oracle 字段保留两者
- **Bug 3**：模型路由冲突，agent 和 user simulator 都被路由到 vLLM → 已修复，改为 `agent_model_args` 传 `api_base`/`api_key` per-call override

`generate_baseline.py` 新增 `agent_model_args` 参数支持上述 per-call override。

#### 3. 评估策略：本地 Mac 通过 VS Code 端口转发

服务器无外网，DashScope API 无法从服务器调用（且 Teleport SSH 不支持 `-N -L` 直连隧道）。
确立的方案：VS Code 连服务器 → PORTS 面板转发 8001 → 本地 Mac 跑评估。详见 Session 7。

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

  | 实验 | 说明 | 优先级 | 状态 |
  |------|------|------|------|
  | E0 | zero-shot（直接评估，无训练）| ★★★ | retail ✅ / airline ✅ |
  | E2 | SFT on BoN 295 条 | ★★★ | retail ✅ / airline ✅ |
  | E3 | SFT on PD 197 条 | ★★★ | retail ✅ / airline ✅ |
  | E1 | SFT on standard 61 条 | ★★ | ⬜ |
  | E2+ | E2 → DPO on BoN episode-level pairs | ★★ | ⬜ |
  | E4 | E3 → DPO on PD turn-level pairs | ★★ | retail ✅ / airline ✅（thinking OFF，效果不佳）|

- [ ] 评估所有模型（retail+airline test split，`eval_on_tau_bench.py`），报告 pass@1
- [ ] **核心对比**：E2+ vs E4（turn-level vs episode-level DPO），E2 vs E3（PD vs BoN SFT）

### Phase C — On-policy 自蒸馏（当前阶段）

**核心思路**：用 Qwen3-8B 自己生成训练数据（thinking ON），迭代 fine-tune 自己，消除 distribution mismatch。不与 BoN 比较，专注于自蒸馏是否能让模型迭代提升。

#### 自蒸馏迭代框架

```
Base Qwen3-8B
    ↓ (PD生成高质量轨迹, thinking ON)
Round 0 数据 → SFT → Qwen3-8B-v1
    ↓ (v1用PD生成更好的轨迹)
Round 1 数据 → SFT → Qwen3-8B-v2  ...
```

**评估重点**：test split pass@1 是否随轮次上升（与 base 模型比较，user simulator 也用 Qwen3-8B）。

#### Round 0 数据生成结果（train split，115 tasks）

**PD（K=5, H=2）**

| 指标 | 值 |
|-----|---|
| Oracle（任意1次成功） | 55/115 = **47.8%** |
| Avg pass@1 | **32.2%** |
| 成功轨迹数（SFT数据）| **111条** |

**BoN N=3 消融（同样3次试验，无value function）**

| 指标 | 值 |
|-----|---|
| Oracle（任意1次成功） | 48/115 = **41.7%** |
| Avg pass@1 | **28.8%** |
| 成功轨迹数（SFT数据）| ~100条 |

PD oracle 比 BoN 高 +6.1pp，说明 value function 确有引导作用；PD pass@1 高 +3.4pp。

#### Round 0 SFT 训练

训练 5 epochs，111条成功轨迹，thinking ON 数据，约 200s。

```
train_loss: 1.141 → 收敛
mean_token_accuracy: 0.7247
```

#### Round 0 评估结果（test split，3 trials，user=Qwen3-8B）

| 模型 | retail pass@1 | retail pass@3 | retail oracle | airline pass@1 | airline pass@3 | airline oracle |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| Base Qwen3-8B | 0.2157 | 0.1176 | 0.3529 | **0.1556** | 0.0667 | **0.2667** |
| PD SFT（111条） | **0.2647** | **0.1176** | 0.4412 | 0.1333 | 0.0667 | 0.2000 |
| BoN SFT（~100条） | 0.2353 | 0.0588 | **0.4706** | **0.1778** | 0.0667 | **0.2667** |
| Mixed SFT（PD+BoN） | 0.1961 | 0.0882 | 0.3529 | 0.0667 | 0.0667 | 0.0667 |

**注意**：此处 user simulator 是 Qwen3-8B，与 Phase B 评估（user=Qwen-Plus）不可直接比较。

#### 消融分析结论

**数据生成 vs 评估结果的关联：**

| domain | PD oracle | BoN oracle | 差值 | PD SFT | BoN SFT | 优胜者 |
|--------|-----------|------------|------|--------|---------|--------|
| retail | 更高 | 较低 | +6.2pp | **0.2647** | 0.2353 | **PD** |
| airline | 较高 | 相近 | +5.7pp | 0.1333 | **0.1778** | **BoN** |

**解读**：
- retail：PD oracle 明显更高 → value function 有效引导 → PD 轨迹质量高 → SFT 效果好
- airline：两者 oracle 相近，但 BoN 轨迹更干净（无 value function 噪声）→ BoN SFT 更好
- **结论**：PD 数据质量优势体现在 oracle 率差距大时；BoN 轨迹更一致干净，适合 value function 噪声较大的场景

#### 混合训练结果（❌ 失败）

将 PD(111条) + BoN(~100条) 数据合并后训练，结果比单独训练更差：
- retail pass@1: 0.1961（低于 base 0.2157）
- airline pass@1: 0.0667（灾难性遗忘，oracle 也只有 0.0667）

**原因**：PD 和 BoN 轨迹风格冲突（PD = step-level 最优选，BoN = episode 整体成功但步骤未必最优），混合后训练信号矛盾。且 PD 数据以 retail 为主，混合后模型 airline 分布被严重稀释。

**结论：简单合并无效，放弃混合策略。**

#### 完整执行流程（全部在服务器上，无需联网）

**环境说明**
- 终端 A：`vllm-env`，负责 vLLM serving
- 终端 B：`pd-qwen3-8b`，负责数据生成和训练
- 每次新终端先执行：`source /user/zhujiatong/miniconda3/bin/activate`

**Step 1：启动 vLLM**

```bash
# 数据生成用 base 模型（thinking ON）
bash scripts/start_vllm.sh base

# 评估 fine-tuned 模型（thinking ON）
bash scripts/start_vllm.sh pd_onpolicy

# 验证（no_proxy 绕过服务器代理）
no_proxy=localhost curl http://localhost:8001/v1/models
# 两个 variant 都已设置 --served-model-name finetuned，返回 "finetuned"
```

**Step 2：PD 数据生成**

```bash
python -m src.data_generation.generate_trajectories_onpolicy --domain retail airline --K 5 --H 2 --num-trials 3 --max-concurrency 3 --output-dir data/raw_trajectories_onpolicy
```

支持断点续传（已有 `task_{id}_trial_{n}_pd.json` 文件自动跳过）。

**Step 3：构建数据集**

```bash
python -m src.data_generation.build_dataset --raw-dir data/raw_trajectories_onpolicy --sft-output data/sft_dataset_onpolicy/train.jsonl --dpo-output data/dpo_dataset_onpolicy/train.jsonl
```

`build_dataset.py` 支持 `--source pd|baseline|bon`（默认 pd）。

**Step 4：SFT 训练（先关掉 vLLM）**

```bash
python -m src.training.sft_train_onpolicy --model /user/zhujiatong/models/Qwen3-8B --dataset data/sft_dataset_onpolicy/train.jsonl --output /user/zhujiatong/outputs_pd/sft_pd_onpolicy --eval-fraction 0
```

注：`sft_train_onpolicy.py` 已设 5 epochs，max_length=16384（容纳 thinking tokens）。

**Step 5：评估（在服务器上，无需本地 Mac）**

```bash
# 服务器启动 fine-tuned 模型
bash scripts/start_vllm.sh pd_onpolicy

# 评估（--local 表示 user model 也用本地 vLLM，--enable-thinking 开 thinking）
python -m src.evaluation.eval_on_tau_bench --domain retail airline --split test --agent-model openai/finetuned --vllm-url http://localhost:8001/v1 --user-model openai/finetuned --local --enable-thinking --num-trials 3 --output-dir outputs/results/sft_pd_onpolicy_round0 --max-concurrency 5

# Base 模型对照（换 base variant，去掉 --enable-thinking 可选）
bash scripts/start_vllm.sh base
python -m src.evaluation.eval_on_tau_bench --domain retail airline --split test --agent-model openai/finetuned --vllm-url http://localhost:8001/v1 --user-model openai/finetuned --local --enable-thinking --num-trials 3 --output-dir outputs/results/base_qwen3_local --max-concurrency 5
```

#### 实验进度

| 实验 | 说明 | 状态 |
|------|------|------|
| Round 0 PD 数据生成 | 115 tasks, 3 trials, K=5 H=2 | ✅ 完成（111条成功） |
| Round 0 BoN N=3 消融 | 115 tasks, N=3 | ✅ 完成（~100条成功） |
| Round 0 PD SFT | 111条，5 epochs | ✅ 完成 |
| Round 0 BoN SFT | ~100条，5 epochs | ✅ 完成 |
| Round 0 eval（PD vs BoN vs Base） | test split，user=Qwen3-8B | ✅ 完成（见上表） |
| 混合训练（PD+BoN） | ~211条混合数据 | ✅ 完成（airline 崩溃，策略放弃） |
| Round 1 自蒸馏 | 用 sft_pd_onpolicy 生成 Round 1 数据 | ⬜ 待做 |

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
    ├── step2_generate.sh
    └── start_vllm.sh              # ← 新增：服务器启动 vLLM（bon/pd/dpo 参数）
```
