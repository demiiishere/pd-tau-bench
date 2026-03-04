# Claude 工作上下文文档

> 给下一个 Claude 实例在继续工作前阅读的技术备忘录。
>
> **本地 Mac（Phase A 数据生成）**
> 项目路径：`/Users/zhujiatong/pd-tau-bench/`
> Conda 环境：`pd-tau-bench`（用 `python3.11`，没有 `python` 软链接）
> API Key：存在用户的 shell 环境变量 `DASHSCOPE_API_KEY` 里（每次对话重新设置）
>
> **GPU 服务器（Phase B 训练+评估）**
> AutoDL A100-PCIE-40GB，项目在 `/root/autodl-tmp/pd-tau-bench/`
> 环境配置见下方"服务器环境配置"小节

---

## 项目是什么

在 τ-bench 上实现 Turn-level Predictive Decoding (PD)：
- 每个 agent 决策点采样 K 个候选回复，对每个候选向前模拟 H 步，打分，选最优的
- 用这个机制生成高质量轨迹数据（SFT + DPO 格式）
- Phase B：用这些数据 SFT/DPO 微调 Qwen3-8B，验证超过 baseline

**Phase A（✅ 完成：数据已生成，数据集已构建）**
**Phase B（🔄 进行中：上服务器做预实验）**

---

## tau2-bench 关键架构（必读）

### 核心类和调用关系

```
Orchestrator
  ├── agent: LLMAgent            # 调用 LLM 生成 agent 回复
  ├── user: UserSimulator        # 调用 LLM 模拟用户
  ├── environment: Environment   # 管理 DB 状态，执行 tool call
  │    └── tools: RetailTools    # 包含 tools.db (RetailDB, Pydantic model)
  ├── agent_state: LLMAgentState # messages list（不含 incoming msg）
  ├── user_state: UserState      # messages list
  ├── trajectory: list[Message]  # 全部消息历史
  ├── from_role / to_role        # 当前消息流向
  └── message                   # 当前待处理消息
```

### step() 的状态机逻辑

```python
# AGENT/ENV → USER：让 user simulator 生成回复
# USER/ENV → AGENT：让 agent 生成回复  ← PD 的拦截点
# AGENT/USER → ENV：执行 tool call，修改 DB
```

### 任务加载

```python
from tau2.domains.retail.environment import get_tasks, get_environment
tasks = get_tasks(task_split_name="base")  # retail=114, airline=50
env = get_environment()  # 每次需要新的 env（DB 是独立的 Pydantic model）
```

### 模型调用（litellm 封装）

```python
from tau2.utils.llm_utils import generate
msg = generate(model="openai/qwen-plus", messages=[...], tools=[...], temperature=0.8)
# 返回 AssistantMessage（可能包含 tool_calls）
```

---

## 我们的 PD 实现

### Fork/Restore 机制

```python
# 保存状态（在 agent 决策前）
state = save_orchestrator_state(orch)
# → deepcopy: agent_state, user_state, trajectory, tools.db, user_db

# 恢复状态（foresight 结束后，或者注入选定候选前）
restore_orchestrator_state(orch, state)
```

**关键**：DB 是 Pydantic model（RetailDB），完全 deepcopy-safe。不需要额外处理。

### PD 主循环 (`core.py`)

```python
while not orch.done:
    if orch.to_role == Role.AGENT:
        # PD 决策点
        saved_state = save_orchestrator_state(orch)
        candidates = _generate_candidates(orch, incoming_message, K, temp=0.8)

        # 优化：如果所有候选近乎相同（sim > 0.95），跳过 PD 节省 API
        if _candidates_are_identical(candidates):
            inject best=candidates[0], skip foresight
            continue

        for candidate in candidates:
            restore_orchestrator_state(orch, saved_state)
            _inject_agent_response(orch, incoming_message, candidate)
            for _ in range(H):        # foresight rollout（greedy）
                orch.step()
            score, fs = compute_value(orch, task)

        best = candidates[argmax(scores)]
        restore_orchestrator_state(orch, saved_state)
        _inject_agent_response(orch, incoming_message, best)  # 注入到真实环境
    else:
        orch.step()  # 非 agent 步骤直接执行
```

### Value Function (`value_function.py`)

**Session 3 全面改进**，所有信号在 foresight delta（`trajectory[foresight_start_idx:]`）上计算：

```python
score = 0.35 × delta_progress     # foresight 中新调了多少期望工具
      + 0.25 × foresight_health   # tool 调用错误/冗余惩罚
      + 0.15 × user_sentiment     # 用户回复正/负面关键词
      + 0.15 × termination        # AGENT_STOP=1.0, USER_STOP=0.7, 运行中=0.4
      + 0.10 × env_assertions     # 满足多少 env 断言（或 action_overlap）

delta_progress:
  - 1.0: 期望工具在 foresight 前已全部完成
  - 0.5 + 0.5*(matched/remaining): 有新期望工具被调用
  - 0.5: 无新工具调用（中性，不惩罚文本回复步骤）

foresight_health:
  - 1.0: 无错误无冗余
  - 0.6: penalty <= 1   (1 次错误或 2 次冗余)
  - 0.3: penalty <= 2
  - 0.0: penalty > 2
  - 0.7: 无 tool call（文本步骤，中性）

user_sentiment（关键词匹配最后一条 user 消息）:
  - 0.75: 正面词 (yes, ok, correct, thanks...)
  - 0.50: 无 user 消息或中性
  - 0.25: 负面词 (no, wrong, confused, cancel...)
```

**效果**：True PD rate（gap > 0.05 的步骤比例）从 6% 提升到 17%。
**核心改动**：`compute_value()` 现在接受 `foresight_start_idx` 参数，`core.py` 中传递 `foresight_start_idx=traj_len_before`。

---

## 已验证的结果

```
retail task 0: PD(K=3, H=2) reward=1.0 vs baseline reward=0.0

关键决策点（step 6）：
  候选1 score=0.820 ← 被选中：格式更清晰的商品展示
  候选0/2 score=0.695 ← 被拒：格式略差，导致用户后续确认不够直接
```

fork/restore 测试：3/3 通过。SFT/DPO 数据格式：正确。

---

## 最新修改（Session 4 — 2026-03-04）

### 1. Phase A 数据生成全部完成

| 域 | PD | BoN | Baseline |
|----|----|-----|---------|
| retail | 240 ✅ | 400 ✅ | 81 ✅ |
| airline | 102 ✅ | 174 ✅ | 33 ✅ |

### 2. 最终数据集（已构建完毕）

```
data/sft_dataset/train.jsonl          197 条  (retail 133 + airline 64)   → E3
data/sft_dataset/train_bon.jsonl      295 条  (retail 196 + airline 99)   → E2
data/sft_dataset/train_baseline.jsonl  61 条  (retail 41  + airline 20)   → 参考
data/dpo_dataset/train.jsonl          313 对  (retail 185 + airline 128)  → E4
```

### 3. build_dataset.py 三个 bug 修复

1. **BoN 字段名**：summary 文件用 `oracle_reward`，旧代码查 `final_reward` → 全部被过滤（0条）
2. **跨域 ID 泄漏**：retail/airline task ID 有 27 个重叠，合并 allowed_ids 导致 retail 测试集泄漏进训练集。修复：每个 domain 独立筛选
3. **BoN 只取 oracle best**：旧代码 1条/task → 61条。修复：用所有 `*_bon_n*.json` 成功样本 → 295条

**正确调用方式**（默认参数即可，内部已分域处理）：
```bash
conda run -n pd-tau-bench python3.11 -m src.data_generation.build_dataset
```

### 4. 数据质量汇总

| 指标 | retail PD | retail BoN | airline PD | airline BoN |
|------|-----------|-----------|------------|-------------|
| Pass@1 | 55.4% | 48.5% | 62.7% | 57.7% |
| Oracle | 82.5% | 76.2% | 85.7% | 88.6% |
| 全失败 tasks | 14 | 19 | 5 | 4 |
| 真实 PD 决策 (gap>0.05) | 27% | — | 35% | — |
| Token/episode | 963k | 89k | 935k | 83k |

---

## 历史修改（Session 3 — 2026-03-03）

### 1. Value Function 大改进

见上方 "Value Function" 小节。核心：5 个 delta-based 信号，`foresight_start_idx` 参数。

### 2. BoN 完成 + PD 重新生成

- BoN 80/80 retail tasks 完成：oracle@5=76%，pass@1=49%，每 task 447k tokens
- 旧 PD 数据全部删除，用新 value function 重新生成（后台运行中）
- PD pass@1=57% > BoN pass@1=49%（公平比较，均为单轨迹）

### 3. GPU 策略：RTX 5090 (32GB)，无需量化

- Qwen3-8B bf16 ≈ 16GB，LoRA+优化器 ≈ 2GB，activations ≈ 2-4GB → 总计 ~20GB
- 仍需 `gradient_checkpointing=True`（处理 8192 token 序列）

### 4. 训练脚本修复（`src/training/`）

**sft_train.py 改动**：
- `trust_remote_code=True`，`model.config.use_cache = False`
- `tokenizer.apply_chat_template()` 处理带 tool_calls 的 messages 列
- `gradient_checkpointing=True`，`gradient_checkpointing_kwargs={"use_reentrant": False}`
- `dataset_text_field="text"`，`lr_scheduler_type="cosine"`
- `--source` 过滤参数，eval split（5%），保存 tokenizer

**dpo_train.py 改动**：
- `ref_model=None`（PEFT 模式，省 16GB VRAM）
- DPO 数据预处理：`prompt_text + chosen_full[len(prompt_text):]`
- 同样的 gradient_checkpointing 设置
- `--min-score-gap` 过滤（默认 0.05）

### 5. 论文策略

- E1+E2+E3 = 可发表最小组合
- E4（DPO）：~390 对，67% gap<0.10，预期提升 0-3pp，低优先级
- **关键路径**：完成 PD 生成 → build_dataset → 租 GPU → Phase B 训练 → 评估

---

## 历史修改（Session 2）

### 1. Train/Test Split（`configs/task_splits.json`）

```json
{
  "seed": 42,
  "retail": {"train": [80个task_id], "test": [34个task_id]},
  "airline": {"train": [35个task_id], "test": [15个task_id]}
}
```

加载方式：
```python
from src.predictive_decoding.tau_bench_adapter import load_task_split
train_ids = load_task_split("retail", "train")  # list[str]
# split="all" returns None (use all tasks)
```

**生成脚本均已加 `--split train|test|all` 参数（默认 train）。**

### 2. Token/Time Tracking

所有 episode 的输出 JSON 现在包含：
```json
{
  "wall_time_s": 45.2,
  "api_calls_approx": 120,
  "pd_steps_count": 9,
  "pd_steps_skipped_count": 3
}
```

- `wall_time_s`：实际运行时间
- `api_calls_approx`：候选生成 + foresight step 数之和（估算）
- `pd_steps_skipped_count`：因候选近似相同而跳过 PD 的步骤数

### 3. Best-of-N（BoN）基线

新文件：`src/data_generation/generate_bon.py`

```bash
python -m src.data_generation.generate_bon \
    --domain retail airline --N 5 \
    --model openai/qwen-plus --max-concurrency 5
```

产出文件格式：
- `task_{id}_bon_n{i}.json`：第 i 个样本的完整轨迹
- `task_{id}_bon_summary.json`：包含 oracle_reward, avg_reward, success_count, best_conversation

**BoN 是关键对比基线（E2）：和 PD 使用相同的 inference budget，但无 lookahead。**

### 4. DPO 数据格式修复

**之前的问题**：`_candidate_to_text()` 把 tool_calls 转成 `"[TOOL] name(args)"` 字符串，丢失了结构。

**现在的格式**：`_candidate_to_chatml()` 保留完整 OpenAI-compatible 格式：
```json
{
  "chosen": {
    "role": "assistant",
    "content": null,
    "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
  },
  "rejected": {...}
}
```

这个格式可以直接给 TRL 的 DPO trainer 使用。

`build_dataset.py` 同时也支持了 `--split` 参数（默认 train）。

### 5. 优化：自适应温度 + 跳过重复候选

参考 `potential_optimization.md` 中的 Opt 3。

**自适应温度**（`_adaptive_temperature()`）：
- 前 2 个候选采样后，计算平均文本相似度
- avg_sim > 0.9 → 温度 +0.3（最高 1.2）
- avg_sim > 0.8 → 温度 +0.15
- 促进候选多样性，无额外 API cost

**跳过重复候选**（`_candidates_are_identical()`）：
- 如果超过一半的候选对相似度 > 0.95，跳过 foresight（用 greedy 第一个候选）
- 节省 K×H 次 API 调用（对于高确定性步骤如固定 tool call）

---

## 服务器环境配置（AutoDL A100 40GB，Phase B）

> 本节给在 GPU 服务器上首次开启的 Claude 实例看。本地 Mac 不需要这些步骤。

### 0. AutoDL 开机镜像选择

开实例时选：**PyTorch 2.x / CUDA 12.8（或 12.4）/ Python 3.10（或 3.11）**。
优先选 12.8（驱动更新，支持更新的 wheel）；12.4 也完全够用。不要选 TensorFlow 镜像。

### 1. 上传项目文件

```bash
# 本地执行：把整个项目压缩后上传（排除不必要的大文件）
tar --exclude='./data/raw_trajectories' \
    --exclude='./.git' \
    --exclude='./tau2-bench/.git' \
    -czf pd-tau-bench.tar.gz .

# 用 AutoDL 的文件上传功能，或：
scp -P <port> pd-tau-bench.tar.gz root@<ip>:/root/autodl-tmp/

# 服务器上解压
cd /root/autodl-tmp
tar -xzf pd-tau-bench.tar.gz -C pd-tau-bench
cd pd-tau-bench
```

> **注意**：AutoDL 的 `/root/autodl-tmp` 是高速 SSD，数据和模型都放这里。
> `/root` 本身空间很小，不要在那里存大文件。

### 2. 上传数据集

把本地 `data/sft_dataset/` 和 `data/dpo_dataset/` 上传到服务器同路径：

```bash
# 本地执行
scp -P <port> -r data/sft_dataset data/dpo_dataset root@<ip>:/root/autodl-tmp/pd-tau-bench/data/
```

### 3. 安装 Python 依赖

```bash
# 创建 conda 环境（如果镜像已有 python 3.11 也可以直接用）
conda create -n pd-tau-bench python=3.11 -y
conda activate pd-tau-bench

# 用清华镜像加速 pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 训练依赖
# CUDA 12.8（AutoDL 推荐选项）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# 如果 cu128 wheels 还没发布，cu124 在 12.8 驱动上也能跑（12.x 向后兼容）
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.45 trl>=0.9 peft>=0.12 datasets>=2.20 accelerate>=0.33
pip install bitsandbytes loguru  # bitsandbytes 备用，当前脚本不用量化

# 评估依赖（serving 微调后的模型）
pip install vllm

# tau2-bench（评估时需要，tau2 是我们用的 benchmark 框架）
cd tau2-bench
pip install -e .
cd ..
```

### 4. 下载 Qwen3-8B 模型（国内用镜像）

```bash
# 设置 HuggingFace 镜像（国内服务器必须）
export HF_ENDPOINT=https://hf-mirror.com

# 下载到 autodl-tmp（大约 16GB）
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-8B \
    --local-dir /root/autodl-tmp/models/Qwen3-8B \
    --local-dir-use-symlinks False
```

### 5. 设置环境变量

```bash
# DASHSCOPE_API_KEY 用于评估时的 user simulator（调 qwen-plus）
export DASHSCOPE_API_KEY=sk-xxx

# HF 镜像（每次开机需要重设，或写入 ~/.bashrc）
export HF_ENDPOINT=https://hf-mirror.com

# 可以写入 ~/.bashrc 让它持久化：
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo 'export DASHSCOPE_API_KEY=sk-xxx' >> ~/.bashrc
```

### 6. 运行训练

```bash
cd /root/autodl-tmp/pd-tau-bench
conda activate pd-tau-bench

# E3: SFT on PD trajectories（约 2-3 小时）
python -m src.training.sft_train \
    --model /root/autodl-tmp/models/Qwen3-8B \
    --dataset data/sft_dataset/train.jsonl \
    --output /root/autodl-tmp/outputs/sft_pd

# E2: SFT on BoN trajectories
python -m src.training.sft_train \
    --model /root/autodl-tmp/models/Qwen3-8B \
    --dataset data/sft_dataset/train_bon.jsonl \
    --output /root/autodl-tmp/outputs/sft_bon

# E4 (低优先级): DPO after SFT
python -m src.training.dpo_train \
    --sft-model /root/autodl-tmp/outputs/sft_pd/final \
    --dataset data/dpo_dataset/train.jsonl \
    --output /root/autodl-tmp/outputs/dpo_pd \
    --min-score-gap 0.10
```

### 7. 运行评估

**服务器无外网，评估在本地 Mac 上跑**（服务器只负责 vLLM serving）。

```bash
# 服务器：后台启动 vLLM（无需外网）
nohup vllm serve /root/autodl-tmp/outputs/sft_bon/final \
    --served-model-name finetuned \
    --port 8001 --trust-remote-code --dtype bfloat16 \
    > vllm.log 2>&1 &

# 本地 Mac：建 SSH 隧道（AutoDL 查看实例信息获取 ip 和 port）
ssh -L 8001:localhost:8001 root@<server_ip> -p <port>

# 本地 Mac：跑评估（agent 走 SSH 隧道→vLLM，user simulator 走 DashScope）
python -m src.evaluation.eval_on_tau_bench \
    --domain retail airline \
    --split test \
    --agent-model finetuned \
    --vllm-url http://localhost:8001/v1 \
    --user-model openai/qwen-plus \
    --num-trials 3 \
    --output-dir outputs/results/sft_bon
```

评估脚本已修复 3 个 bug（见 PROGRESS.md Session 6）。输出 `summary.json` 含 pass@1、pass@k（all succeed）、oracle（any succeed）。

### 常见坑（服务器专属）

- **vLLM 启动慢**：首次加载模型需要 2-5 分钟，等 `Uvicorn running on...` 出现再跑评估
- **autodl-tmp 掉电不持久**：关机前把重要产出（模型、结果）同步到对象存储或下载到本地
- **CUDA OOM**：如果 SFT 时 OOM，先检查是否有其他进程占用 GPU：`nvidia-smi`
- **HF 下载失败**：确认 `HF_ENDPOINT` 已设置，或手动指定 `--endpoint https://hf-mirror.com`

---

## Phase B 训练策略（Session 5 决策）

### 核心原则：小数据体制下的训练

197 条域内 SFT 数据处于 8B 模型训练的下限。训练策略不对会导致过拟合或学不到东西。
关键参考：Lima (2023) 用 1000 条精选数据就调出接近 GPT-4 的对话质量。
τ-bench 的任务格式高度统一，模型学的是"调整决策偏好"而不是"新技能"，所以数据需求更少。

### LoRA 配置（已更新到训练脚本）

```python
# 小数据体制：rank 低，只调 Q/V，dropout 高
LoraConfig(
    r=8,               # 不用 16/32 —— 可训练参数越少，越不容易过拟合
    lora_alpha=16,     # alpha/r = 2
    target_modules=["q_proj", "v_proj"],  # 只调 Q 和 V，跳过 K/O/MLP
    lora_dropout=0.1,  # 加大 dropout 防过拟合
)
```

### SFT 超参（已更新）

```python
SFTConfig(
    num_train_epochs=1,           # 多 epoch = 反复背诵；1 个就够
    learning_rate=5e-5,           # 比默认偏高，少数据需要更大步长
    gradient_accumulation_steps=4,# 有效 batch=4（之前是 8）
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
)
```

### DPO 超参（已更新）

```python
DPOConfig(
    num_train_epochs=1,
    learning_rate=1e-6,   # 比 SFT 小 50 倍，DPO 超参非常敏感
    beta=0.3,             # 偏大 KL 惩罚（原论文 0.1），防止小数据下模型漂移太远
    gradient_accumulation_steps=4,
)
```

### 通用数据混合（关键正则化手段）

**这是把 ~200 条域内数据用好的核心手段。** 单独用 197 条训，模型很可能过拟合到 τ-bench 的特定用语。
混入 ~3000 条通用 function calling 数据（如 glaive-function-calling-v2）作为正则化：

```bash
# 下载 glaive 数据（约 110k 条，取前 3000）
# huggingface-cli download glaiveai/glaive-function-calling-v2 --local-dir ...
# 预处理成 {"messages": [...]} 格式，取 3000 条存为 data/general/glaive_3k.jsonl

# 训练时加 --general-data（三组实验用完全相同的通用数据）
python -m src.training.sft_train \
    --model ... --dataset data/sft_dataset/train.jsonl \
    --general-data data/general/glaive_3k.jsonl \
    --output ...
```

**重要**：E1/E2/E3 必须用完全相同的通用数据（3000 条），唯一变量是 ~200 条域内数据来源。

### 实验矩阵（更新后）

| 编号 | 训练数据 | 方法 | 域内数据量 | 说明 |
|------|---------|------|----------|------|
| E0 | — | zero-shot | 0 | baseline |
| E1 | standard成功轨迹 | SFT | ~61 条 | 对照 |
| E2 | BoN 成功轨迹 | SFT | ~295 条 | BoN baseline |
| E2+ | BoN + episode-level DPO | SFT→DPO | 同 E2 + BoN 偏好对 | BoN+DPO baseline |
| E3 | PD 成功轨迹 | SFT | ~197 条 | PD-SFT |
| E4 | PD + turn-level DPO | SFT→DPO | 同 E3 + 313 对 | **我们的方法** |

所有 SFT 实验均混入相同的 3000 条通用数据。

**核心对比链**：
- E2 → E2+：episode-level DPO 有没有用
- E2+ → E4：**turn-level DPO vs episode-level DPO**（论文最强 claim）
- E3 → E4：DPO 偏好数据的额外贡献

### E2+ 数据构建（已实现）

```bash
# 重新构建所有数据集（同时生成 BoN episode-level DPO 对）
python -m src.data_generation.build_dataset
# 额外输出：data/dpo_dataset/train_bon_episode.jsonl
```

E2+ DPO 训练：
```bash
python -m src.training.dpo_train \
    --sft-model outputs/sft_bon/final \
    --dataset data/dpo_dataset/train_bon_episode.jsonl \
    --output outputs/dpo_bon_episode
```

### 数据量风险缓解

如果域内数据仍嫌不足，优先考虑：
1. **增加 PD trials**：当前 3 trials/task → 改 5 trials，PD SFT 预计从 197 涨到 ~300
2. **合并 PD+BoN 成功轨迹**：两者都是高质量成功轨迹，合并约 ~490 条
3. **调整 split 比例**：70/30 → 85/15（更多训练任务 → 更多成功轨迹）

---

## 下一步工作（Phase B，当前阶段）

### 1. 上传文件到服务器

```bash
# 本地：打包项目（排除原始轨迹，只需数据集）
tar --exclude='./data/raw_trajectories' \
    --exclude='./.git' \
    --exclude='./tau2-bench/.git' \
    -czf pd-tau-bench.tar.gz .

# 上传数据集（约 50MB）
scp -P <port> -r data/sft_dataset data/dpo_dataset root@<ip>:/root/autodl-tmp/pd-tau-bench/data/
```

### 2. 服务器训练（含通用数据混合，优先级顺序）

```bash
cd /root/autodl-tmp/pd-tau-bench
conda activate pd-tau-bench

# 可选：先重新构建数据集（如果需要 BoN episode-level DPO 对 E2+）
python -m src.data_generation.build_dataset
# 新增输出：data/dpo_dataset/train_bon_episode.jsonl

# （可选）下载并准备通用数据（各实验共享，glaive ~3k 条）
# 见 CLAUDE_CONTEXT.md "Phase B 训练策略" 小节

# E2: SFT on BoN（295 条，先跑做 debug，最快）
python -m src.training.sft_train \
    --model /root/autodl-tmp/models/Qwen3-8B \
    --dataset data/sft_dataset/train_bon.jsonl \
    --output /root/autodl-tmp/outputs/sft_bon
    # [+ --general-data data/general/glaive_3k.jsonl 如果已准备]

# E3: SFT on PD（197 条，核心实验）
python -m src.training.sft_train \
    --model /root/autodl-tmp/models/Qwen3-8B \
    --dataset data/sft_dataset/train.jsonl \
    --output /root/autodl-tmp/outputs/sft_pd

# E2+: DPO on BoN episode-level pairs（BoN+DPO baseline，与 E4 对比）
python -m src.training.dpo_train \
    --sft-model /root/autodl-tmp/outputs/sft_bon/final \
    --dataset data/dpo_dataset/train_bon_episode.jsonl \
    --output /root/autodl-tmp/outputs/dpo_bon_episode

# E4: DPO on PD turn-level pairs（我们的方法）
python -m src.training.dpo_train \
    --sft-model /root/autodl-tmp/outputs/sft_pd/final \
    --dataset data/dpo_dataset/train.jsonl \
    --output /root/autodl-tmp/outputs/dpo_pd \
    --min-score-gap 0.10
```

### 3. 评估

```bash
# 终端 1: serve 模型
vllm serve /root/autodl-tmp/outputs/sft_pd/final \
    --port 8001 --trust-remote-code --dtype bfloat16

# 终端 2: 评估（retail + airline test split）
python -m src.evaluation.eval_on_tau_bench \
    --domain retail airline --split test \
    --model-url http://localhost:8001/v1 \
    --user-model openai/qwen-plus \
    --num-trials 3 \
    --output results/sft_pd_eval.json
```

---

## 踩过的坑

1. **conda 环境里没有 `python` 软链接**，要用 `python3.11`
2. **litellm 提示 "model isn't mapped"**：只是缺 qwen-plus 定价数据，不影响功能，可忽略
3. **Fork test 的第三个断言**：不能断言两次独立 LLM 调用产生相同输出（即使 temperature=0）。改为断言 restore 后的起始状态相同。
4. **H=1 完全没有区分度**：因为 foresight 只走一步，所有候选要么都调同一个 tool，要么都是文本但 action_overlap=0。必须用 H=2。
5. **evaluate_simulation 需要 SimulationRun**：不能直接传 trajectory，要包成 SimulationRun 对象。
6. **环境 fork 时 user_tools 可能为 None**：retail domain 没有 user tools，代码里已处理（try/except）。
7. **DPO 格式**：旧代码用 `_candidate_to_text()` 转成字符串，丢失 tool_call 结构，已修复为 `_candidate_to_chatml()`。
8. **PD 退步 bug（已修复）**：当所有候选 score gap=0 时，PD 仍然注入一个 temperature=0.8 的候选，而不是 greedy（temperature=0.0）的输出，导致在确定性步骤引入噪声，某些 baseline=1.0 的任务 PD 反而变成 0.0。
   - 修复：在 `_pd_step()` 中，若 `max_score - min_score < 0.05`，自动 fallback 到 greedy（重新生成 1 个 temperature=0.0 的候选）
   - 受影响的旧数据：task_8（全部 reward=0）已删除重跑；tasks 15、16 有部分成功 trial 可用
   - 旧数据里 reward=1.0 的文件不受影响（仍可用于 SFT）；DPO 数据不受影响（gap=0 的步骤本来就被过滤）
9. **Value function gap=0 问题（Session 3 改进）**：H=2 的 foresight 太短，94% 步骤所有候选调同一 tool → score 完全相同。根本原因是旧信号（env_assertions, action_overlap）不区分 foresight delta。新的 delta-based 信号（delta_progress, health, sentiment）直接在 foresight 窗口内计算，有效区分候选。
10. **DASHSCOPE_API_KEY 在 kill 进程后丢失**：kill 后台进程时环境变量丢失。重新启动时用 `DASHSCOPE_API_KEY=xxx nohup python3.11 ...` 直接传入，不依赖 export。
11. **SFTTrainer 的 messages 列处理**：TRL 的 SFTTrainer 不能直接处理包含 tool_calls 的 messages 列（版本依赖问题）。正确做法：先用 `tokenizer.apply_chat_template()` 转为文本，再用 `dataset_text_field="text"`。
12. **DPO ref_model 内存**：同时加载 model 和 ref_model 需要 32GB（16GB×2），在 32GB 显卡上无法容纳激活值。使用 `ref_model=None` + PEFT 配置，TRL 自动用禁用 LoRA 的基模型作参考，节省 16GB。A100 40GB 可以显式加载 ref_model，但 `ref_model=None` 在任何显卡上都是更优方案。
13. **build_dataset.py 必须分域处理**：retail 和 airline 的 task ID 有 27 个重叠（0-49 范围内）。如果合并 allowed_ids 再过滤，retail 测试集的某些 task 会因为恰好与 airline 训练集 ID 相同而被错误纳入。修复：main() 现在按 domain 循环，每个 domain 独立构建后 append 合并。
14. **BoN SFT 数据不要只用 oracle best**：旧代码从 `_bon_summary.json` 的 `best_conversation` 只取 1 条/task，导致 retail 仅 61 条。所有成功的 `_bon_n*.json` 都可以用，retail 有 196 条。修复后 E2 和 E3 数据量更可比（295 vs 197）。

---

## 关键文件快速导航

```
configs/task_splits.json                           train/test split（seed=42）
src/predictive_decoding/core.py          L33   run_pd_episode() 主函数
src/predictive_decoding/core.py          L113  _pd_step() PD 决策点
src/predictive_decoding/core.py          L178  _generate_candidates() 候选生成（含自适应温度）
src/predictive_decoding/core.py          L214  _adaptive_temperature() 自适应温度
src/predictive_decoding/core.py          L237  _candidates_are_identical() 跳过重复候选
src/predictive_decoding/core.py          L254  _evaluate_candidate() foresight（返回 score + steps）
src/predictive_decoding/core.py          L285  _inject_agent_response() 注入候选
src/predictive_decoding/tau_bench_adapter.py   L28  get_tasks()
src/predictive_decoding/tau_bench_adapter.py   L37  load_task_split() ← 新增
src/predictive_decoding/tau_bench_adapter.py   L114 save/restore_orchestrator_state()
src/predictive_decoding/value_function.py           compute_value()
src/data_generation/generate_trajectories.py  批量生成 PD 轨迹（支持 --split）
src/data_generation/generate_baseline.py      baseline 生成（支持 --split，含时间追踪）
src/data_generation/generate_bon.py           BoN 生成 ← 新增
src/data_generation/build_dataset.py          SFT/DPO 构建（DPO 格式已修复，支持 --split）
```
