# Claude 工作上下文文档

> 给下一个 Claude 实例在继续工作前阅读的技术备忘录。
> 项目路径：`/Users/zhujiatong/pd-tau-bench/`
> Conda 环境：`pd-tau-bench`（用 `python3.11`，没有 `python` 软链接）
> API Key：存在用户的 shell 环境变量 `DASHSCOPE_API_KEY` 里（每次对话重新设置）

---

## 项目是什么

在 τ-bench 上实现 Turn-level Predictive Decoding (PD)：
- 每个 agent 决策点采样 K 个候选回复，对每个候选向前模拟 H 步，打分，选最优的
- 用这个机制生成高质量轨迹数据（SFT + DPO 格式）
- Phase B：用这些数据 SFT/DPO 微调 Qwen3-8B，验证超过 baseline

**Phase A（本地，基础实现完成，新增 BoN + train/test split + 优化，待跑全量数据）**
**Phase B（租 GPU，尚未开始）**

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

```
score = 0.5 × assertion_score + 0.3 × termination_score + 0.2 × error_score

assertion_score:
  - 如果 task 有 env_assertions → 检查当前 env state 满足几个
  - 否则 → action_overlap（已完成的 expected tool calls 比例）

termination_score:
  - AGENT_STOP → 1.0
  - USER_STOP  → 0.7
  - 未完成     → 0.4
  - 报错       → 0.0

error_score:
  - 0 错误 → 1.0
  - ≤2 错误 → 0.5
  - ...
```

**已知限制**：大多数步骤 score gap ≈ 0，因为所有候选调同一个 tool call。只有在文本决策点（需要 COMMUNICATE 的步骤）才有明显区分。H=2 比 H=1 好很多（H=1 完全没有区分度）。

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

## 最新修改（Session 2）

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

## 下一步工作（按优先级）

### 1. 全量数据生成（立刻可以跑，Train split 只）

```bash
conda activate pd-tau-bench
cd /Users/zhujiatong/pd-tau-bench
export DASHSCOPE_API_KEY=xxx

# PD 轨迹（使用 train split，默认）
python -m src.data_generation.generate_trajectories \
    --domain retail airline --K 5 --H 2 \
    --model openai/qwen-plus --num-trials 3 \
    --max-concurrency 5
# split 默认为 train，无需额外指定

# BoN 基线（同样使用 train split）
python -m src.data_generation.generate_bon \
    --domain retail airline --N 5 \
    --model openai/qwen-plus --max-concurrency 5

# 构建训练数据集
python -m src.data_generation.build_dataset
```

预期产出：~240 SFT(PD) + ~960 DPO + ~240 SFT(BoN)，耗时 6-10h，花费 ¥50-100。

### 2. Phase B（租 GPU 后）

- A100 40G，AutoDL 或 Featurize，~¥3-5/h
- 4 组实验：E1(zero-shot), E2(SFT BoN), E3(SFT PD), E4(SFT+DPO PD)
- 评估命令在 `scripts/step4_eval.sh` 里

---

## 踩过的坑

1. **conda 环境里没有 `python` 软链接**，要用 `python3.11`
2. **litellm 提示 "model isn't mapped"**：只是缺 qwen-plus 定价数据，不影响功能，可忽略
3. **Fork test 的第三个断言**：不能断言两次独立 LLM 调用产生相同输出（即使 temperature=0）。改为断言 restore 后的起始状态相同。
4. **H=1 完全没有区分度**：因为 foresight 只走一步，所有候选要么都调同一个 tool，要么都是文本但 action_overlap=0。必须用 H=2。
5. **evaluate_simulation 需要 SimulationRun**：不能直接传 trajectory，要包成 SimulationRun 对象。
6. **环境 fork 时 user_tools 可能为 None**：retail domain 没有 user tools，代码里已处理（try/except）。
7. **DPO 格式**：旧代码用 `_candidate_to_text()` 转成字符串，丢失 tool_call 结构，已修复为 `_candidate_to_chatml()`。

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
