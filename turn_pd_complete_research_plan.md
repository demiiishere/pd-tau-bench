# Non-Myopic Data Generation for Tool-Calling Agents via Predictive Decoding

## Complete Research Plan

---

## 1. Research Story（整体叙事）

### 1.1 Problem

LLM-based agents 在需要多轮对话 + 工具调用的真实场景中可靠性很低。τ-bench 的实验数据显示，即使是 GPT-4o 这种顶级模型，在 retail 客服任务上单次成功率（pass^1）也不到 50%，连续 8 次都成功（pass^8）更是低于 25%。

问题的根源是：agent 在自回归生成时只看当前步的局部概率，不考虑未来后果。在多轮对话中，一个早期的微小失误（漏问一个关键信息、用了不够清晰的表述、选错了 tool）会在后续步骤中放大，最终导致整个任务失败。这种"短视"现象已经在数学推理任务中被量化（Ma et al., 2024 发现约 60% 的 Llama3 推理轨迹存在明显短视），但在 agent tool-calling 场景中尚未被系统研究。

### 1.2 Observation

目前提升 agent 能力的主流方式是用强模型（teacher）生成成功轨迹，再用这些轨迹 SFT 训练小模型（student）。但 **teacher 本身生成轨迹时也是短视的**——它用的也是标准自回归解码。这意味着 SFT 数据的质量存在系统性的天花板。

一个关键但被忽视的问题是：**如果我们能让 teacher 在生成轨迹时做出非短视的决策，SFT 数据质量会不会更高？训练出来的 student 模型会不会更可靠？**

### 1.3 Our Approach

我们提出 **Turn-level Predictive Decoding (Turn-PD)**，把 Predictive Decoding（Ma et al., 2024）从 token-level 推理方法改造为 turn-level 的 agent 数据生成方法：

- 在 agent 的每个决策点（每个 assistant turn），采样 K 个候选 response
- 对每个候选，在环境中向前模拟 H 轮完整的对话（agent + user + tool execution）
- 用 value function 对每个 foresight 轨迹打分
- 选择 value 最高的候选作为当前步的 action

关键创新点：

1. **Turn-level 而非 Token-level**：原始 PD 在每个 token 上做 foresight，但 agent 任务的最小有意义决策单元是一个完整的 turn（可能包含文本 + tool call）。Turn-level PD 在正确的粒度上做搜索。

2. **天然产出 SFT + DPO 双数据**：PD 过程中每个决策点的 K 个候选和它们的 value score 天然构成了偏好对（高分 chosen vs 低分 rejected）。一次数据生成同时产出 SFT 数据（成功轨迹）和 DPO 偏好数据（turn-level preference pairs）。

3. **搜索成本在数据生成阶段摊销**：PD 在推理时很贵（每步 K×H 次生成），但我们只在数据生成时用一次。训练出来的模型推理时用标准解码，零额外推理成本。

### 1.4 为什么这个方法比别的更好

和三类方法对比：

**vs. Inference-time search（MCTS、beam search、Tree-of-Thought 等）**：
这些方法每次推理都要付出搜索成本。我们把搜索成本一次性"烘焙"进数据和模型里，推理时零额外开销。对于高 QPS 的客服场景，这个区别很重要。

**vs. Best-of-N filtering**：
Best-of-N 在 episode 级别做筛选——生成 N 条完整轨迹，只保留成功的。但它不知道成功轨迹中哪些步骤是关键的、失败轨迹中哪些步骤出了错。Turn-PD 在每个步骤上做搜索，不仅成功率更高，还能提供 step-level 的偏好信号。

**vs. RL（GRPO、RLHF 等）**：
RL 需要在线交互和复杂的训练基础设施。Turn-PD 是一个纯离线的数据生成方法，产出标准的 SFT/DPO 数据，可以用任何现有的训练框架处理。门槛更低，且和 RL 不冲突——PD 数据可以作为 RL 的初始化。

### 1.5 Impact

- **实用层面**：提供了一个简单可行的方法来提升 tool-calling agent 的训练数据质量，只需要 API 调用和标准 SFT/DPO 训练。
- **方法论层面**：证明了"用 inference-time search 优化数据生成"这条路在 agent 领域同样有效，且 turn-level 是正确的搜索粒度。
- **数据层面**：PD 天然产生 turn-level 偏好数据，这在现有数据生成文献中是独特的。

---

## 2. Related Work（需要覆盖的领域）

### 2.1 LLM Agent Benchmarks
- **τ-bench / τ²-bench**（Yao et al., 2024; Barres et al., 2025）：我们的主要评测平台。强调 pass^k 可靠性指标。
- **ToolBench**（Qin et al., 2023）、**API-Bank**（Li et al., 2023）：其他 tool-calling benchmarks，可做泛化性验证。
- **SWE-bench**（Jimenez et al., 2024）：代码 agent benchmark，和 terminal 任务相关但不是我们的重点。

### 2.2 Non-Myopic / Lookahead Decoding
- **Predictive Decoding**（Ma et al., 2024, ICLR 2025）：我们直接基于的工作。Token-level MPC-based decoding。
- **φ-Decoding**（Xu et al., 2025, ACL 2025）：PD 的加速版，adaptive foresight。
- **Genius**（Ma et al., 2025, ACL 2025）：PD 用于 RL 采样。和我们的方向互补——他们用 PD 做 RL 采样，我们用 PD 做 SFT/DPO 数据生成。
- **MCTS for LLM**（Xie et al., 2024; Hao et al., 2023）：树搜索方法，计算成本更高。
- **Contrastive Decoding**（O'Brien & Lewis, 2024）：另一类改进 decoding 的方法，不做 foresight。

### 2.3 Learning from Search / Distilling Search into Policy
- **STaR**（Zelikman et al., 2022）：用 rationale 自我改进。
- **ReST / ReST^EM**（Gulcehre et al., 2023; Singh et al., 2024）：生成 → 过滤 → 训练 的迭代框架。
- **V-STaR**（Hosseini et al., 2024）：同时训练 verifier 和 generator。
- **Scaling LLM Test-Time Compute**（Snell et al., 2024）：分析了 inference-time compute 的最优分配。
- 我们的定位：和 ReST 类似的"生成→过滤→训练"框架，但用 PD 替代 standard sampling 做生成，且在 turn-level 而非 episode-level 做搜索。

### 2.4 Agent Training Data Generation
- **AgentTrek**（Xu et al., 2024）：为 web agent 自动生成训练轨迹。
- **Agent-FLAN**（Chen et al., 2024）：agent 能力的指令微调数据。
- **Nemotron-Terminal**（Pi et al., 2026）：terminal agent 的 SFT 数据工程。虽然领域不同，但 data engineering 思路可以对比。
- **Nemotron-Agentic**（NVIDIA, 2025）：多轮 tool-calling 轨迹数据集生成，使用 LLM judge 做质量过滤。

### 2.5 Preference Learning for Agents
- **DPO**（Rafailov et al., 2023）：我们用的偏好学习算法。
- **Step-DPO**（Lai et al., 2024）：在推理步骤级别做 DPO，和我们的 turn-level DPO 类似但用于数学推理。
- **RLHF for agents**：多数现有工作在 episode-level 给 reward，我们的 turn-level preference signal 是更细粒度的。

---

## 3. 实验设计

### 3.1 评估 Benchmark

**⚠️ 关键问题：训练数据和评估数据不能重叠**

我们的 SFT/DPO 训练数据是在 τ-bench 的任务上生成的。如果评估也在同一批任务上做，reviewer 会质疑数据泄漏（data contamination）。

虽然 τ-bench 的评估是动态交互式的（user simulator 每次回复有随机性，对话路径每次不同，不可能"背答案"），但为了彻底消除质疑，**必须做 task-level 的 train/test split**：

```
τ-bench retail (~115 tasks):
  → Train split: ~80 tasks  （只用这些任务生成 PD/baseline/BoN 轨迹做训练数据）
  → Test split:  ~35 tasks  （只用这些任务做评估，训练时完全不碰）

τ-bench airline (~50 tasks):
  → Train split: ~35 tasks
  → Test split:  ~15 tasks
```

**Claude Code 实现注意**：在数据生成脚本中必须实现这个 split 逻辑。建议按 task_id 做固定 seed 的 random split（70/30），保存 split 文件确保可复现。评估时严格只用 test split 的任务。

**主评估（Primary）**：

| Benchmark | 域 | 训练任务 | 测试任务 | 指标 | 说明 |
|-----------|-----|---------|---------|------|------|
| τ-bench retail | 零售客服 | ~80 | ~35 | pass^1, pass^5 | 主场景，test split 严格 held-out |
| τ-bench airline | 航空客服 | ~35 | ~15 | pass^1, pass^5 | 第二域，验证跨域效果 |

**OOD 泛化评估（Out-of-Distribution，必须做）**：

| Benchmark | 用途 | 备注 |
|-----------|------|------|
| **BFCL v3** (Berkeley Function Calling Leaderboard) | 纯 function calling 准确率 | 完全没有在 τ-bench 上训练过的域和 API schema，能证明泛化性 + 通用能力没退化 |
| **τ²-bench telecom** | 第三域（电信），dual-control 模式 | 训练时完全没接触过的域，验证跨域迁移 |

BFCL 是**必须做**的 OOD 评估。理由：如果你只在 τ-bench 的 held-out test set 上报告结果，reviewer 仍然可以说"模型只是过拟合到了 τ-bench 的 API schema 和客服场景"。在 BFCL 上也评估能堵住这个质疑。telecom 域如果时间够也做一下，不够的话 BFCL 是底线。

### 3.2 底座模型选择

| 模型 | 参数量 | 为什么选 |
|------|--------|---------|
| **Qwen3-8B**（主实验） | 8B | Teacher 用 Qwen API，student 也用 Qwen 分布最匹配；原生支持 function calling |
| **Llama-3.1-8B-Instruct**（泛化实验） | 8B | 证明方法不依赖特定模型家族；社区最广泛使用的底座之一 |

两个底座都做完整实验，但 Qwen3-8B 作为主实验报告，Llama 作为泛化性验证。

### 3.3 训练数据方案

**来源 1：τ-bench 域内数据（我们的 PD 方法生成）**

基于现有 train split 任务直接生成的轨迹（当前实际产出量）：

| 数据 | 来源 | 实际量 |
|------|------|--------|
| PD-SFT | PD 成功轨迹 (3 trials/task) | 197 条 (retail 133 + airline 64) |
| PD-DPO | PD turn-level 偏好对 (gap≥0.10) | 313 条 (retail 185 + airline 128) |
| Baseline-SFT | 标准 decoding 成功轨迹 | 61 条 (retail 41 + airline 20) |
| BoN-SFT | Best-of-N (N=5) 所有成功样本 | 295 条 (retail 196 + airline 99) |
| BoN-episode-DPO | BoN episode-level 偏好对（成功 vs 失败，用于 E2+） | `train_bon_episode.jsonl`（已实现） |

**来源 2：τ-bench 任务扩增（增加域内数据量的关键手段）**

197 条 SFT + 313 条 DPO 偏少。τ-bench 的框架天然支持合成新任务来扩增数据。有三种方式，按性价比排序：

**方式 A：同任务多轨迹（最简单）**

同一个 task 增加 trials 数（从 3 提到 5-8），用不同 temperature / 不同 user simulator seed 跑出不同对话路径。零额外工程量，但任务多样性不增加。预期产出：SFT 从 ~197 涨到 ~350。

**方式 B：扰动现有任务生成变体（推荐，性价比最高）**

拿 train split 的现有任务，用 LLM 生成变体。操作方式：

```python
perturbation_prompt = """
你是一个 τ-bench 任务设计师。以下是一个 retail 客服任务的原始定义：

用户身份：{original_user_identity}
用户意图：{original_user_intent}
目标数据库状态：{original_goal_state}

请生成 3 个变体任务，每个变体修改以下一个维度：
1. 换一个不同的商品组合（但保持相同的操作类型：换货/退货/查询）
2. 增加一个复杂度（比如用户中途改主意、用户提供了错误信息需要纠正）
3. 换一个不同的操作类型（如果原来是换货，变成退货+重新下单）

对每个变体，输出：
- user_instruction: 给 user simulator 的指令
- goal_assertions: 预期的数据库目标状态断言
- 确保变体和原任务足够不同，但仍在 retail 域的规则范围内
"""
```

每个原始任务生成 3 个变体 → train split 80 个任务变成 ~320 个任务。然后在这些变体上跑 PD 生成轨迹。

⚠️ **关键约束**：
- 只对 train split 的任务做扩增
- 生成的变体需要人工 spot-check 一些，确保 goal_assertions 合理
- 变体不能和 test split 的任务实质相同（用 LLM 做相似度检查过滤）

**方式 C：完全合成新任务（工程量最大）**

从零生成全新的用户场景和对应的目标状态。τ-bench 的原始论文在 Stage 3 就是这么标注任务的，可以自动化：

```
1. 基于域规则和 API 定义，让 LLM 设计一个新的用户场景
2. 让 LLM 标注 goal state（目标数据库状态）
3. 用 teacher model 跑一遍验证任务可解
4. 通过验证的任务加入任务池
```

这种方式能产出最多样化的任务，但需要验证任务质量（有些 LLM 生成的 goal state 可能和域规则矛盾）。

**建议策略**：先用 A（加 trials 到 5）快速把数据量翻倍，再用 B（变体生成）进一步扩到 300-500 个任务。方式 C 作为后备。

**扩增后的预期数据量**：

| 数据 | 扩增前 | 方式 A 后 | 方式 A+B 后 |
|------|--------|-----------|-------------|
| PD-SFT | ~197 | ~350 | ~700-1000 |
| PD-DPO | ~313 | ~550 | ~1000-1500 |

这个量级配合通用数据混合，训练就比较稳了。

**来源 3：通用 function calling SFT 数据（防止能力退化）**

如果只用 τ-bench 的域内数据做 SFT，模型可能会过拟合到 τ-bench 的特定格式和领域（零售、航空客服），通用 function calling 能力反而下降。为了对冲这个风险，混合一些通用数据：

| 数据集 | 来源 | 大小 | 用途 |
|--------|------|------|------|
| **glaive-function-calling-v2** | HuggingFace (glaiveai) | ~110K | 通用 function calling 对话 |
| **Nemotron-Agentic-v1** (tool-calling subset) | HuggingFace (nvidia) | ~数万条 | 高质量多轮 tool-calling 轨迹 |
| **xlam-function-calling-60k** | HuggingFace (Salesforce) | 60K | 多样化 function calling |

不需要全部用。如决定混合，建议从 glaive 或 xlam 中随机采样 **~3000 条**。

**实现方式（可选）**：通过 `sft_train.py` 的 `--general-data` 参数传入，默认不启用。若启用，三组实验（E1/E2/E3）必须传入完全相同的文件，唯一变量仍是域内数据来源。

### 3.4 核心实验矩阵

#### 主实验（Qwen3-8B 底座）

| 编号 | 训练方法 | 训练数据 | 域内数据量 | 目的 |
|------|---------|---------|----------|------|
| **E0** | 无微调 | — | 0 | 零样本 baseline |
| **E1** | SFT | 标准 decoding 成功轨迹 | ~61 条 | Standard SFT baseline |
| **E2** | SFT | Best-of-N 成功轨迹 (N=5) | ~295 条 | BoN SFT baseline |
| **E2+** | SFT → DPO | BoN-SFT + BoN episode-level 偏好对 | 同 E2 + episode DPO | BoN+DPO baseline（关键消融） |
| **E3** | SFT | PD 成功轨迹 | ~197 条 | PD-SFT（我们的方法 v1） |
| **E4** | SFT → DPO | PD-SFT + PD turn-level 偏好对 | 同 E3 + 313 对 | PD-SFT-DPO（我们的方法，最终版） |

所有 SFT 实验（E1/E2/E3）均使用相同的训练设置（LoRA 超参、epoch 数等），唯一变量是域内数据来源。可选：通过 `--general-data` 统一混入通用 function calling 数据做正则化，若加则三组实验必须用相同的量（推荐 glaive-function-calling-v2 ~3000 条）。

**核心对比链**：
- E2 → E2+：episode-level DPO 对 BoN 有没有帮助
- **E2+ → E4：turn-level DPO vs episode-level DPO**（论文最强 claim）
- E2 → E3：PD 轨迹质量是否优于 BoN 轨迹（成本相近）
- E3 → E4：turn-level DPO 偏好数据的额外贡献

#### Best-of-N Baseline 的生成方法

```
对每个 τ-bench 任务：
1. 用 teacher model (Qwen-Plus) + temperature=0.8 生成 N=5 条完整轨迹
2. 只保留 final_reward=1 的轨迹
3. 如果多条成功，随机选一条（或全部保留做数据增广）
```

成本和 PD 粗略可比（PD 每步 K 次采样 + H 步 foresight；BoN 每 episode N 次完整采样），这使得对比公平。

#### 泛化性实验（Llama-3.1-8B-Instruct 底座）

| 编号 | 训练方法 | 训练数据 |
|------|---------|---------|
| **L0** | 无微调 | — |
| **L3** | SFT | PD 成功轨迹 |
| **L4** | SFT → DPO | PD-SFT + PD-DPO |

只做 3 组，目的是验证"PD 数据对不同底座都有效"。

### 3.5 Ablation Studies

| Ablation | 变量 | 实验组 | 目的 |
|----------|------|--------|------|
| **A1: PD 参数敏感性** | K（候选数） | K=3 / 5 / 8 | 候选多样性 vs 成本 trade-off |
| **A2: Foresight 深度** | H（前瞻轮数） | H=1 / 2 / 3 | 更深的 foresight 是否持续改善 |
| **A3: SFT vs DPO** | 训练方法 | E3 vs E4 | DPO 偏好数据的额外贡献 |
| **A4: 数据混合比例** | PD 数据占比 | 10% / 30% / 50% / 100% PD | 和通用数据混合的最优比例 |
| **A5: Value function 设计** | 打分方式 | 纯环境 / 环境+LLM-judge / 纯 LLM-judge | 哪种 value function 最有效 |
| **A6: Turn-PD vs Episode BoN** | 搜索粒度 | Turn-level PD vs Episode-level BoN（成本匹配） | 核心对比：相同预算下 turn-level 搜索是否优于 episode-level 过滤 |
| **A7: Teacher model 强度** | Teacher 模型 | Qwen-Plus / Qwen-Max / Qwen-Turbo | 更强的 teacher 是否带来更好的 PD 数据 |

**最关键的 ablation 是 A3 和 A6**：
- A3 回答"DPO 偏好数据到底有没有用"
- A6 回答"turn-level 搜索到底比 episode-level 过滤好在哪"

### 3.6 分析实验

除了数字结果，需要做以下定性/定量分析：

**分析 1：PD 在哪类任务上收益最大**

按任务特征（轮数、tool call 数量、reward_basis 类型）分组，报告每组的 PD 收益。假设：多轮 + 有 COMMUNICATE 评估的任务收益最大。

**分析 2：PD 在对话的第几轮收益最大**

统计每个 turn 的平均 candidate score gap。假设：早期 turn（前 1-3 轮）的 gap 最大，因为早期决策的影响范围最广。这能直接支持"PD 缓解早期短视"的故事。

**分析 3：典型 case study（3-5 个）**

手工挑选最有代表性的例子，展示：
- PD 和 baseline 在哪一步分歧
- PD 的 foresight 看到了什么 baseline 没看到的
- 这个分歧如何最终导致任务成功/失败

（你已经有了 task 0 的 case study，全量跑完后再挑几个不同类型的。）

**分析 4：DPO 偏好对的质量**

统计 DPO 数据中 chosen vs rejected 的 score gap 分布。如果 gap 太小（<0.05），说明偏好信号太弱；如果分布有明显的双峰，说明有一些高质量偏好对和一些噪声对。

**分析 5：计算成本分析**

做一个 cost-performance Pareto 图：
- X 轴：总 API token 消耗（或总 API 成本）
- Y 轴：τ-bench pass^1
- 标注每个方法的点：Standard, BoN-3, BoN-5, BoN-10, PD(K=3,H=1), PD(K=5,H=2), PD(K=8,H=3)

如果 PD 在 Pareto frontier 上，说明它在成本效率上也有优势。

### 3.7 数据生成时记录的统计量（已实现）

成本分析（分析 5）和 Turn-PD vs BoN 的公平对比（ablation A6）依赖 token 消耗和时间记录。**已在数据生成阶段实现**，每条 PD 轨迹 JSON 包含：

```json
{
  "tokens": {
    "episode":  {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...},
    "overhead": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...},
    "total":    {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
  },
  "wall_time_s": 45.2,
  "pd_steps_count": 9,
  "pd_steps_skipped_count": 3,
  "pd_steps_greedy_fb_count": 2
}
```

- **episode**：最终轨迹（成为训练数据的部分）消耗的 token
- **overhead**：PD 探索（候选生成 + foresight）消耗的额外 token，不进入训练数据
- **total**：episode + overhead，代表 PD 的真实总成本

token 数从每次 API 响应的 `msg.usage.prompt_tokens` / `completion_tokens` 中实时累加，非估算。Baseline 和 BoN 同样记录 `wall_time_s` 和 token 统计，支持公平成本对比。

最终需要汇总的表格（用于论文）：

```
| Method            | Total Tokens | Overhead Tokens | Wall Time | Pass@1 |
|-------------------|-------------|-----------------|-----------|--------|
| Standard (greedy) | ...         | 0               | ...       | ...    |
| BoN (N=5)         | ...         | N/A             | ...       | ...    |
| PD (K=5, H=2)     | ...         | ... (~92%)      | ...       | ...    |
```

---

## 4. 评估指标

| 指标 | 定义 | 重要性 |
|------|------|--------|
| **pass^1** | 单次成功率 | 高——基本能力 |
| **pass^k (k=3,5)** | 连续 k 次全部成功的概率 | 最高——可靠性，τ-bench 的核心指标 |
| **avg reward** | 多次试验的平均 reward | 中——连续指标，比 pass 更稳定 |
| **BFCL accuracy** | Berkeley Function Calling Leaderboard 准确率 | 中——验证通用能力没退化 |
| **API cost** | 数据生成的总 token 消耗 | 中——实用性指标 |

**报告方式**：每个实验跑 5 个 trial（τ-bench 的 user simulator 有随机性），报告均值 ± 标准差。

---

## 5. 预期结果与风险

### 5.1 乐观预期

| 对比 | 预期提升 | 依据 |
|------|---------|------|
| E3 (PD-SFT) vs E1 (Standard-SFT) | pass^1 +5-10% | PD 轨迹质量更高 |
| E3 (PD-SFT) vs E2 (BoN-SFT) | pass^1 +3-5% | Turn-level 搜索比 episode-level 过滤更精细 |
| E4 (PD-DPO) vs E3 (PD-SFT) | pass^1 +2-4%, pass^5 +5-10% | DPO 偏好数据提升可靠性 |
| E4 vs E0 (零样本) | pass^1 +15-25% | SFT+DPO 的综合效果 |

### 5.2 风险与应对

| 风险 | 可能性 | 应对 |
|------|--------|------|
| PD 和 BoN 差距不显著 | 中 | 强调 DPO 数据的独特价值（BoN 产不出 turn-level 偏好对）；强调 pass^k 而非 pass^1 |
| 数据量太少导致 SFT 不稳定 | 中 | 混合通用 function calling 数据（E5/E6）；增加 num_trials |
| Qwen user simulator 和 GPT-4o user simulator 行为差异大 | 低-中 | 最终评估时用 GPT-4o 做 user simulator（花一点钱）；在论文中讨论这个限制 |
| τ-bench 任务太少，结果方差大 | 中 | 多 trial 评估 + 报告置信区间；在 retail 和 airline 两个域上分别报告 |
| BFCL 上能力退化 | 低 | 混合通用数据（E5/E6）来对冲 |

---

## 6. 论文结构建议

```
Title: Turn-Level Predictive Decoding for Non-Myopic Agent Data Generation

Abstract

1. Introduction
   - Agent 可靠性问题（τ-bench 数据）
   - 短视是核心原因
   - 我们的方法：Turn-PD for data generation
   - 贡献总结

2. Background & Related Work
   - 2.1 LLM Agents and Tool Calling
   - 2.2 Predictive Decoding and Non-Myopic Generation
   - 2.3 Learning from Search: STaR, ReST, V-STaR
   - 2.4 Preference Learning for Agents

3. Method
   - 3.1 Problem Setup: τ-bench 环境定义
   - 3.2 Turn-Level Predictive Decoding
   - 3.3 Value Function Design
   - 3.4 从 PD 到训练数据：SFT + DPO 数据构造

4. Experiments
   - 4.1 Setup（模型、数据、指标）
   - 4.2 Main Results（E0-E6 对比表）
   - 4.3 Cross-Model Generalization（Llama 实验）
   - 4.4 Ablation Studies（A1-A7）
   - 4.5 Cost-Performance Analysis

5. Analysis
   - 5.1 Where Does Turn-PD Help Most?（按任务类型和对话轮次分析）
   - 5.2 Case Studies
   - 5.3 DPO 偏好数据质量分析

6. Discussion
   - 和 inference-time search 的对比
   - 和 RL 的互补关系
   - 局限性（user simulator 差异、数据量、域的局限性）

7. Conclusion
```

---

## 7. 投稿目标

| 会议 | Deadline（大致） | 适合程度 | 原因 |
|------|-----------------|---------|------|
| **NAACL 2026** | 约 2026 年 6 月 | 高 | NLP + agent 方向，接受中等规模的工作 |
| **EMNLP 2026** | 约 2026 年 6 月 | 高 | 同上，可能更适合 |
| **ICLR 2027** | 约 2026 年 10 月 | 中-高 | 更看重方法论创新，需要实验更扎实 |
| **NeurIPS 2026** | 约 2026 年 5 月 | 中 | 竞争激烈，需要更强的实验 |
| **COLM 2026** | 待定 | 高 | 语言模型专门会议，非常对口 |

建议先冲 NAACL/EMNLP 2026（如果赶得上 deadline），比较 realistic。

---

## 8. 完整执行 Checklist

### Phase A：数据生成（本地）✅ 完成

```
=== 全部完成 ===
[x] τ²-bench 环境搭建
[x] 环境 fork/restore 机制
[x] Turn-level PD 核心实现（含自适应温度、跳过重复候选、greedy fallback）
[x] Value function（delta-based 5 信号：delta_progress, health, sentiment, termination, env_assertions）
[x] 单任务验证通过（task 0: PD=1.0 vs baseline=0.0）
[x] task-level train/test split（retail 80/34, airline 35/15, seed=42, configs/task_splits.json）
[x] token/时间追踪（tokens.episode/overhead/total + wall_time_s，从 API usage 字段实时累加）
[x] 全量 PD 轨迹生成（retail 240 + airline 102，K=5, H=2, 3 trials）
[x] 全量 BoN 轨迹生成（retail 400 + airline 174，N=5）
[x] 全量 baseline 轨迹生成（retail 81 + airline 33）
[x] 构造 SFT 数据集：train.jsonl(197) + train_bon.jsonl(295) + train_baseline.jsonl(61)
[x] 构造 DPO 数据集：train.jsonl(313 PD turn-level pairs) + train_bon_episode.jsonl(BoN episode pairs)
[x] 数据质量检查（pass@1: retail PD 55.4% > BoN 48.5%，airline PD 62.7% > BoN 57.7%）

=== Phase A-2: 任务扩增（可选，当前数据量不足时再做） ===
[ ] 方式 A：增加 PD trials 到 5-8（PD-SFT 预计从 197 涨到 ~330）
[ ] 方式 B：用 LLM 对 train split 任务生成变体（每任务 3 个变体）
[ ] 对变体任务 spot-check 质量，过滤与 test split 相似的变体
[ ] 在变体任务上跑 PD + baseline 生成轨迹
```

### Phase B：训练（租 GPU）

```
[ ] 租 GPU 服务器（AutoDL A100-40G 或同等）
[ ] 配置服务器环境（conda, transformers, trl, peft, vllm）
[ ] 下载 Qwen3-8B 模型（hf-mirror.com，约 16GB）
[ ] 上传数据集（data/sft_dataset/ + data/dpo_dataset/）

优先级顺序：
[ ] E0: 评估 Qwen3-8B 零样本 baseline（在 test split 上，无需训练）
[ ] E2: SFT on BoN 295 条 → 评估 retail+airline test split（先跑做 debug）
[ ] E3: SFT on PD 197 条 → 评估 test split（核心实验）
[ ] E1: SFT on standard 61 条 → 评估 test split
[ ] E2+: E2 → DPO on BoN episode-level pairs → 评估 test split
[ ] E4: E3 → DPO on PD turn-level 313 对 → 评估 test split（我们的方法）

核心对比：E2+ vs E4（turn-level vs episode-level DPO）
[ ] L0/L3/L4: Llama 底座泛化实验（视时间）
[ ] 关键 ablation: A3 (SFT vs DPO) 和 A6 (Turn-PD vs BoN，成本匹配)
[ ] 其余 ablation 按优先级做
[ ] BFCL 评估（所有模型，必须做）
[ ] τ²-bench telecom 评估（如果时间允许）
```

### Phase C：分析和写作

```
[ ] 按任务类型分组分析 PD 收益
[ ] 按对话轮次分析 score gap
[ ] 挑选 3-5 个 case study
[ ] DPO 数据质量统计
[ ] Cost-performance Pareto 图（基于 Phase A 记录的 token 数据）
[ ] 论文初稿
```
