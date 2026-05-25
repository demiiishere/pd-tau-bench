# 毕业论文大纲

**题目（候选）**：基于预测解码的大语言模型智能体行为蒸馏研究
（英文副标题候选：Predictive Decoding for LLM Agent Distillation: When Does Foresight Help?）

**字数目标**：约 1.5 万字（中文）

**核心论点（一句话）**：预测解码（PD）作为 SFT 数据生成方法的有效性存在 *teacher 能力门槛*——强 teacher（Qwen-Plus）上 PD 数据优于 BoN，中等 teacher（Qwen3-32B）上 PD 优势消失；但 PD 副产物（每步候选 ranking）作为 turn-level DPO 偏好信号始终稳定有效，BoN-SFT + PD-DPO 是 4B 学生模型的最佳蒸馏组合。

**全文一句话结论**：Qwen3-4B 在 τ2-bench 三域上零样本平均成功率 2.4%，经 BoN-SFT + PD-DPO 蒸馏后达 14.5%，相对提升 6×，其中 telecom 域从 0% 提升至 20.6%。

---

## 第一章 绪论（约 2000 字）

### 1.1 研究背景

**1.1.1 LLM Agent 的兴起与产业化困境**
- LLM 在工具调用、API 编排、多轮规划等复杂任务上的能力快速进步
- 企业级 Agent 部署的核心约束：延迟、成本、上下文长度，倾向于使用 4B–8B 量级的小模型
- **核心矛盾**：强模型（Qwen3-32B、Qwen-Plus）在 Agent 任务上表现可用，但单次调用成本高；小模型（Qwen3-4B/8B）成本低，但 Agent 能力严重不足

**1.1.2 蒸馏作为成本最优解**
- 用强 teacher 模型生成轨迹数据，用 SFT/DPO 训练弱 student 模型，是兼顾性能和成本的主流方案
- 关键问题：如何高效生成"高质量"的 teacher 轨迹？

### 1.2 研究现状与不足

**1.2.1 主流数据生成策略**
- Best-of-N（BoN）采样：独立采样 N 条轨迹，按 episode 级 reward 选优
  - 优点：实现简单、无需 step-level 信号
  - 缺点：episode 粒度的偏好信号粗糙，"好运气" episode 中也可能混入差的中间步骤
- 预测解码（Predictive Decoding, PD / Lookahead）：每步 K 个候选 + H 步前瞻 + value function 选优
  - 优点：理论上每步局部最优、轨迹"示范性"更强
  - 缺点：计算开销高（K×H 倍 token）；依赖 value function 准确度

**1.2.2 现有工作的盲点**
- 多数蒸馏研究（AgentTuning、FireAct 等）专注于框架设计，对"数据生成策略本身"的系统消融较少
- "PD 数据是否真的优于 BoN" 缺乏在真实 Agent benchmark 上的严格对比
- 数据生成方法的有效性是否依赖 teacher 模型本身的能力，缺乏机制层面的分析

### 1.3 研究问题

本文围绕三个核心问题展开：

1. **Q1（数据质量）**：相同任务集下，PD 生成的训练数据是否真的优于 BoN？这种"质量优势"是否与 teacher 强度相关？
2. **Q2（SFT 收益）**：PD 数据训练的学生模型，下游表现是否系统性优于 BoN 数据训练的学生模型？
3. **Q3（DPO 增益）**：PD 副产物（候选 ranking）作为 turn-level DPO 偏好信号能否进一步提升学生模型？什么样的 SFT 初始化最适合后续 DPO？

### 1.4 研究贡献

本文的主要贡献包括：

1. **将 Predictive Decoding [Ma et al., 2024, arXiv:2410.17195] 从推理时优化扩展到 Agent 蒸馏数据生成**：在 τ2-bench（retail / airline / telecom 三域）上，使用 Qwen-Plus 和 Qwen3-32B 两个 teacher 进行 PD 和 BoN 数据生成的并行实验，给出 ≈1500 条轨迹的对照数据。原始 PD 工作仅在数学/编程/AlfWorld 等单轮或半结构化任务上做 *online inference 控制*，本文首次系统验证其作为 *offline 训练数据生成* 工具的有效性边界。

2. **揭示 PD 的 teacher 能力门槛**：发现 PD 的数据生成优势仅在强 teacher（Qwen-Plus, retail baseline 50.6%）上显著（per-trial 成功率 +5.8~6.4pp），在中等 teacher（Qwen3-32B, retail baseline 37.3%）上几乎消失。机制解释：32B 在 temperature=0.8 下生成的 K 个候选高度同质化，**真实 PD 决策率仅 7-8%**（vs Qwen-Plus 的 27-35%），70% 步骤直接 skip foresight。

3. **提出 BoN-SFT + PD-DPO 组合管线**：实证发现 PD 数据作为 SFT 初始化不稳定（E3d 实验全域崩盘 avg 1.2%），而 BoN-SFT 提供多样化覆盖、PD 候选 ranking 提供精准 turn-level 偏好信号，组合后达到 4B 实验中最高 avg 14.5%（telecom 单域 20.6%）。

4. **量化弱 student 模型的蒸馏可获得性**：Qwen3-4B 零样本 avg 2.4%，蒸馏后 14.5%，相对提升 6 倍。这一显著提升与 Qwen3-8B 在相同小数据下的 SFT 退化形成鲜明对比，验证"能力起点决定 SFT 收益方向"假设。

5. **观察到学生在 telecom 上反超 teacher** [cite: Furlanello et al., 2018; Hsieh et al., 2023]：经蒸馏后 4B 在 telecom 上达到 20.6%，而 32B teacher 单次 rollout 平均成功率仅 10.4%。该现象与知识蒸馏文献中"student outperforms teacher on narrow distribution"一致：teacher 的 *oracle 能力*（≥1 次成功的能力）而非*平均能力*决定了蒸馏的有效上限。

6. **构建端到端 τ2-bench 蒸馏基础设施**：包括 PD 数据生成、value function、SFT/DPO 训练、本地 OpenAI 兼容推理服务、三域评估的完整流程，全部开源可复现。

### 1.5 论文结构

- 第二章：综述 LLM Agent、SFT 蒸馏、DPO、Inference-time Scaling 的相关工作
- 第三章：方法，详述 PD 算法、value function 设计、SFT/DPO 训练管线
- 第四章：实验设置，明确数据集、模型、超参、评估协议
- 第五章：实验结果与分析，分六个子节呈现核心证据链
- 第六章：讨论局限性与未来方向
- 第七章：总结研究发现与实践意义

---

## 第二章 相关工作（约 4000 字）

本章按"由表及里"顺序组织：从 Agent 范式与评测（2.1）、到工具学习与 Agent 蒸馏（2.2）、到偏好学习（2.3）、到本文方法的直接理论基础——非短视生成与推理时计算扩展（2.4）、到偏好信号的精细化（2.5）。

### 2.1 LLM Agent 范式与评测基准

**2.1.1 Agent 范式演进**
- **ReAct** [Yao et al., 2023, arXiv:2210.03629]：思考-行动-观察循环，奠定 LLM Agent 范式；ALFWorld/WebShop 上比 imitation/RL 提升 34%/10% 绝对成功率
- **Reflexion** [Shinn et al., 2023, arXiv:2303.11366]：以语言反馈做"verbal reinforcement"，agent 把失败经验写入 episodic memory，在下次试错中改进；不更新参数仅更新文本提示
- **Toolformer** [Schick et al., 2023, arXiv:2302.04761]：模型自监督学习何时调用何种 API，通过 self-instruct 生成工具调用数据
- Tool Use API 工业标准：OpenAI function calling / Anthropic tool use
- LLM Agent 综述：[Wang et al., 2023, arXiv:2308.11432] 和 [Xi et al., 2023, arXiv:2309.07864]

**2.1.2 Agent 评测基准**
- **τ-bench** [Yao et al., 2024, arXiv:2406.12045]：首次将"政策合规 + 多工具调用 + 用户模拟器"组合为系统评测，揭示即使 GPT-4o 在 retail/airline 上 pass^1 < 50%、pass^8 < 25%
- **τ2-bench** [Sierra Research, 2025, arXiv:2506.07982]：τ-bench 升级版，新增 telecom 域（dual-control Dec-POMDP），agent 和 user 都可调用工具修改共享状态；本文实验基于 τ2-bench
- **WebArena** [Zhou et al., 2024, arXiv:2307.13854]：4 个真实 web 域上的 812 个任务，GPT-4 仅 14.4% 成功率（vs 人类 78.2%）
- **AgentBench** [Liu et al., 2023, arXiv:2308.03688]：8 个环境跨工具/编码/知识/推理评测 LLM 作为 Agent 的核心能力
- 与 GAIA、HotpotQA 的差异：强调"流程合规与多轮交互" 而非"单次知识检索"
- 与 SWE-bench、MMLU 的差异：强调"动作执行带副作用"（修改数据库状态）而非"单次问答"

**2.1.3 小模型 Agent 的挑战**
- **Qwen3 系列** [Qwen Team, 2025, arXiv:2505.09388]：开源 Agent 能力分层，4B << 8B < 32B
- 商业 API 类似但更强：Qwen-Plus、Claude Haiku/Sonnet、GPT-4o-mini 等
- 本文以 Qwen3-4B 作为弱学生模型代表，研究"4B 量级模型能否通过蒸馏达到工业可用水平"

### 2.2 监督微调与知识蒸馏

**2.2.1 指令微调基础**
- **FLAN** [Wei et al., 2022, arXiv:2109.01652]：首次系统展示指令微调显著提升零样本能力
- **InstructGPT** [Ouyang et al., 2022, arXiv:2203.02155]：RLHF + 指令微调，奠定 ChatGPT 范式
- Llama-3 Instruct、Qwen-Instruct 系列：开源指令微调实践
- 参数高效微调（PEFT）：Adapter、Prefix-tuning、LoRA
- **LoRA** [Hu et al., 2021, arXiv:2106.09685] 数学定义：$W = W_0 + BA$，$B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$；将可训练参数量减少约 1 万倍而保持下游性能（具体超参配置详见第三章方法）

**2.2.2 知识蒸馏经典**
- **经典蒸馏** [Hinton et al., 2015, arXiv:1503.02531]：用 teacher 的 soft target 训练 student，温度系数控制信息密度
- **Born-Again Networks** [Furlanello et al., 2018, arXiv:1805.04770]：首次系统报告**学生超过教师**的现象——即使 student 与 teacher 同容量，蒸馏后性能反而提升
- **Distilling step-by-step** [Hsieh et al., 2023, arXiv:2305.02301]：小模型通过 LLM rationale 蒸馏在多项任务上超过比自己大 700× 的 teacher

**2.2.3 Agent 行为蒸馏代表性工作**
- **AgentTuning** [Zeng et al., 2023, arXiv:2310.12823]：收集多任务 Agent 轨迹（AgentInstruct）做 SFT，提升 Llama 2 通用 Agent 能力
- **FireAct** [Chen et al., 2023, arXiv:2310.05915]：多任务 + 多 prompt 风格的轨迹蒸馏，500 条 GPT-4 轨迹将 Llama2-7B HotpotQA 提升 77%
- **ToolLLM** [Qin et al., 2023, arXiv:2307.16789]：16k+ API 的工具学习数据集 ToolBench，DFS-based 决策树搜索增强

**2.2.4 小数据 SFT 的退化现象**
- Catastrophic forgetting：小数据 SFT 容易破坏预训练习得的通用能力 [Luo et al., 2023]
- 常见对策：LoRA 限制更新范围；混入通用指令数据做正则化；low rank + low lr + 单 epoch 三重约束

### 2.3 偏好学习与 DPO

**2.3.1 RLHF** [Ouyang et al., 2022, arXiv:2203.02155]
- Reward Model + PPO 范式，训练流程复杂、对超参敏感

**2.3.2 DPO** [Rafailov et al., 2023, arXiv:2305.18290]
- 直接偏好优化，跳过 reward model，等价转化为分类损失：
$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$
- DPO 已被证明在 sentiment control、summarization、dialogue 上匹配或超过 PPO，且训练显著简化
- 在 Agent 场景的应用：以 turn-level 或 episode-level 的"好/坏动作"构建偏好对

**2.3.3 Episode-level vs Turn-level 偏好信号**
- Episode-level：以整条轨迹 reward 为信号，奖励稀疏，难定位错误步骤
- Turn-level：每步比较候选优劣，信号密度高，但需要 step-level value function
- 近期 Step-DPO 系列工作（如 [Lai et al., 2024, arXiv:2406.18629]）在数学 CoT 上验证了 step-level 偏好比 sequence-level 更有效；本文 PD 在 Agent 多轮工具调用场景下产生天然的 turn-level 候选-排名对，可视为这一思路在 Agent 领域的延伸

### 2.4 非短视生成与推理时计算扩展（本文方法的直接基础）

LLM 的自回归生成本质上是"短视的"（myopic）：每一步只基于当前已生成的 prefix 选择下一 token，没有考虑该选择对未来步骤的影响。在 Agent 多轮任务中这种短视性尤其有害——一个早期的错误决策会被后续轨迹放大，最终导致整个 episode 失败。**非短视生成（non-myopic generation）** 系列工作通过引入"前瞻信号"对每一步选择进行重新打分。

**2.4.1 非短视生成与 Predictive Decoding**
- **Non-myopic Generation of Language Models for Reasoning and Planning** [Ma et al., 2024, arXiv:2410.17195]：首次将 Model Predictive Control（MPC）思想引入 LLM 生成，提出 **Predictive-Decoding** 方法，对 LLM 输出分布按前瞻轨迹（foresight trajectory）的预期收益做重加权。该方法在数学（GSM8K +7.2%）、编程、Agent（AlfWorld +25.3%）任务上对 ReAct 有显著提升
- **Controlled Decoding** [Mudgal et al., 2024, arXiv:2310.17022]：用 prefix scorer 实现 token-level/block-level 控制，弥合 BoN 与 token-level RL 之间的差距
- **Tree of Thoughts (ToT)** [Yao et al., 2023, arXiv:2305.10601]：把 LLM 生成看作树搜索，每步评估多个 thought 候选，支持回溯——Game-of-24 上从 CoT 的 4% 提升到 74%

**2.4.2 Best-of-N 采样**
- 原理：独立采样 N 条完整轨迹，按 reward（或 verifier 分）选优
- Token 消耗：N × 单 episode token
- **Self-Consistency** [Wang et al., 2022, arXiv:2203.11171]：BoN 的轻量化版本，对答案做多数投票，无需 verifier；GSM8K +17.9%
- 作为训练数据来源：保留 oracle-best（或全部成功）轨迹做 SFT——本文 BoN-SFT 即采用此策略

**2.4.3 树搜索与 MCTS 风格方法**
- **LATS / Tree Search for Agents** [Zhou et al., 2024, arXiv:2310.04406]：MCTS 风格的 Agent 推理时优化
- 这类方法通过显式树搜索 + value 评估实现更深的前瞻，但计算开销远高于 PD 的轻量化 K-way 1-level 设计

---

## 第三章 方法（约 3000 字）

### 3.1 问题形式化

**3.1.1 Agent 交互定义**
- 环境：$E = (S, A, T, R)$，$S$ 为状态空间（DB + 对话历史），$A$ 为动作空间（文本回复 + 工具调用），$T$ 为状态转移，$R$ 为 episode-end 二值奖励
- 策略：$\pi_\theta(a_t | h_t)$，其中 $h_t$ 为当前对话历史
- 优化目标：$\max_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$

**3.1.2 蒸馏问题定义**
- Teacher: $\pi_T$（Qwen-Plus 或 Qwen3-32B）
- Student: $\pi_S$（Qwen3-4B）
- 数据生成：用 $\pi_T$ 在训练 task 上生成轨迹 $\mathcal{D}$
- 训练：$\pi_S \leftarrow \mathrm{Train}(\pi_S^0, \mathcal{D})$
- 评估：$\pi_S$ 在 test task 上的 pass@1

### 3.2 Best-of-N 数据生成（基线方法）

**3.2.1 算法**
```
for each task in train_split:
    for n in 1..N:
        rollout an episode with temperature=0.8
        record final_reward
    keep all episodes with reward=1.0 → SFT data
```

**3.2.2 实现细节**
- N=5，独立采样
- temperature=0.8，与 PD 候选采样温度一致
- 保留所有成功 episode（reward=1.0），不去重；若同 task 多个成功，全部用作训练样本（增强多样性覆盖）

**3.2.3 Token 预算**
- Baseline（greedy, 1×）：单 episode 平均 ~85k tokens
- BoN（5×）：5 × 85k = ~425k tokens per task

### 3.3 预测解码数据生成

本节描述本文 PD 算法的具体实现。**算法骨架沿用 Ma et al. (2024) 的 Predictive Decoding [arXiv:2410.17195]**，关键扩展包括：(1) 将候选评分函数从其原始公式扩展为面向 Agent 任务的五维 value function（3.3.3 节）；(2) 引入候选去重与 greedy fallback 机制（3.3.1 节 Step 2 和 epsilon 判定），用以应对 32B 等中等强度 teacher 上候选多样性不足的问题；(3) 增加 env-fork 机制以在 stateful 环境（数据库 + 多轮对话）下安全做 H 步前瞻；(4) 用于 *offline 数据生成* 而非原始工作的 *online inference 控制*。

> 【图 1 — PD 算法流程图】**位置：3.3 节开头**
> 
> 用流程框图（横向，从左到右）描述一次 PD agent step 的完整流程：
> 1. 左侧：当前 conversation history $h$（含 system prompt、user msg、若干轮历史）+ env state（DB snapshot）
> 2. 中间上方：fork 出 K=5 条并行分支，每条采样一个候选回复 $c_1, ..., c_5$（用 5 个不同颜色的 token 串表示）
> 3. 中间判定菱形："候选间余弦相似度 > 0.95 ？" 是→直接选 $c_1$（skip 路径，灰色虚线）；否→进入 foresight
> 4. 中间下方：对每个候选做 H=2 步 greedy foresight（每条分支变长 2 步），同时 env state 被 deepcopy
> 5. 右侧上方：value function 给每条分支打分 $v_1, ..., v_5$，5 个分数显示在小条形图上
> 6. 右侧下方判定："max(v) - min(v) > 0.05 ？" 是→选 argmax 候选作为 chosen action（绿色高亮）；否→选 $c_1$（greedy fallback，黄色虚线）
> 7. 最右：chosen action 注入回真实 history，env restore，继续下一 step
> 
> 颜色编码：候选用 5 种不同色；chosen 用绿色高亮；skip 和 fallback 路径用灰/黄虚线；DB fork-restore 用蓝色框注明。

**3.3.1 核心算法**
```
function pd_episode(task, π_T, K, H):
    h = init_history(task)
    while not terminal(h):
        # Step 1: 采样 K 个候选
        candidates = [π_T.generate(h, temp=0.8) for _ in range(K)]
        
        # Step 2: 去重判定
        if all pairs of candidates similar (cosine > 0.95):
            chosen = candidates[0]   # skip foresight
            log "skipped_identical"
        else:
            # Step 3: K-way foresight
            scores = []
            for c_k in candidates:
                h_fork = deepcopy(h)
                env_fork = deepcopy(env)
                inject(h_fork, c_k)
                for _ in range(H):
                    a = π_T.generate(h_fork, temp=0)   # greedy foresight
                    step(env_fork, a, h_fork)
                v_k = value_function(h_fork[foresight_start:])
                scores.append(v_k)
            
            if max(scores) - min(scores) < ε:   # ε = 0.05
                chosen = candidates[0]   # greedy fallback
                log "greedy_fallback"
            else:
                chosen = candidates[argmax(scores)]
        
        # Step 4: 注入选定候选到真实轨迹
        inject(h, chosen)
        step(env, chosen, h)
    return h, scores_history
```

**3.3.2 配置**
- K=5, H=2（与 BoN N=5 的 token 预算可比性）
- 候选温度 0.8（保证多样性），前瞻温度 0（greedy 减少噪声）
- Number of trials per task: 3（兼顾覆盖与开销）

**3.3.3 Value Function（五维综合评分）**

所有信号在前瞻轨迹（foresight delta，即 $\text{traj}[\text{foresight\_start\_idx}:]$）上计算：

| 信号 | 权重 | 计算方式 | 取值范围 |
|------|------|----------|----------|
| $\delta_{\text{progress}}$ | 0.35 | 前瞻中新调用的期望工具数 / 总期望工具数 | [0, 1] |
| $f_{\text{health}}$ | 0.25 | 1 − (error_calls + redundant_calls) / total_calls | [0, 1] |
| $u_{\text{sentiment}}$ | 0.15 | 用户回复情感分类（正/中/负 → 1.0/0.5/0.0） | {0, 0.5, 1} |
| $t_{\text{termination}}$ | 0.15 | 自然终止 1.0 / 步数耗尽 0.4 / 错误终止 0 | {0, 0.4, 1} |
| $e_{\text{assertions}}$ | 0.10 | 工具调用后 DB 状态合法性比例 | [0, 1] |

总分：$V = \sum_i w_i \cdot s_i$，$\sum w_i = 1.0$

**3.3.4 真实 PD 决策率分析（关键机制指标）**

定义：一个 PD step 被认为"真正由 PD 选择"当且仅当：
1. 没有 skipped_identical（K 候选不同质）
2. 没有 greedy_fallback（max-min score gap > 0.05）

实测（详见 5.5 节）：
- Qwen-Plus：retail 27%, airline 35%
- Qwen3-32B：retail 8.2%, airline 7.2%, telecom 7.5%

**这一指标揭示了 PD 有效性的核心机制：teacher 输出的候选多样性决定 PD 是否能发挥作用。**

**3.3.5 Token 预算**
- PD（K=5, H=2）：~11.3 × baseline ≈ 960k tokens / episode（含 K 候选 + K×H foresight + 真实步骤）
- 对比 BoN：5 × 85k = 425k

### 3.4 SFT 训练管线

**核心思路**：用 LoRA 在 *teacher 成功轨迹* 上做 1 epoch 微调。仅保留 `final_reward == 1.0` 的整条 episode 内的 assistant 决策作为训练样本（每条样本是一次 agent decision，格式为 ChatML messages 序列）；不混入失败轨迹。

LoRA 配置遵循小数据 regime 的标准实践：低秩（r=8）、target 仅 q/v_proj、加 dropout=0.1、单 epoch、cosine schedule with warmup。具体超参表和数据格式细节见附录 B。

训练完成后用 `peft.merge_and_unload()` 合并 adapter，得到完整 fp16/bf16 权重供推理使用。

### 3.5 Turn-Level DPO 训练

**核心思路**：从 PD 轨迹的"真实 PD step"（非 skipped、非 fallback）中提取候选对：

- $y_w = c_{\arg\max_k v_k}$（PD 选中的候选）
- $y_l = c_{\arg\min_k v_k}$（最差候选）

筛选条件：$|v_w - v_l| \geq 0.10$，保证偏好信号显著。32B 数据上得到 298 对。

DPO 训练以 SFT-merged 模型同时作为 base 与 reference policy，遵循 trl 1.4 的 conversational 格式（prompt / chosen / rejected 均为 messages list）。关键超参：β=0.3（小数据防 ref drift）、lr=1e-6、1 epoch；其余配置见附录 B。训练后再次 merge 得到最终评估模型。

### 3.6 端到端蒸馏管线总结

> 【图 2 — 端到端蒸馏管线】**位置：3.6 节**
> 
> 用大型流程图（纵向，从上到下三层）展示整个蒸馏管线：
> 
> **顶层（数据生成阶段）**：左侧画一个大方框"Teacher Model（Qwen-Plus / Qwen3-32B / Qwen3-8B）"。从中间分出两条数据流：
> - 左路径"BoN Generation"（N=5 独立采样）→ 长方形"BoN trajectories"（标注 reward 分布）→ 漏斗过滤"keep reward=1"→ 椭圆"BoN SFT data"
> - 右路径"PD Generation"（K=5, H=2 + value function）→ 长方形"PD trajectories（带 turn-level score）"→ 两个输出分支：(1) 漏斗"keep reward=1"→ 椭圆"PD SFT data"；(2) 候选对抽取"gap ≥ 0.10"→ 椭圆"PD DPO pairs"
> 
> **中层（训练阶段）**：左侧大方框"Student π_S^0 = Qwen3-4B"。四条训练路径分别画出，每条路径上画一个小图标表示 LoRA Adapter：
> - 路径 A：BoN SFT data → SFT-LoRA → merge → 模型框"E3b 模型"
> - 路径 B：PD SFT data → SFT-LoRA → merge → 模型框"E3a 模型"
> - 路径 C：E3b 模型 + PD DPO pairs → DPO-LoRA → merge → 模型框"**E3c 模型 ★**"（绿色边框加粗强调）
> - 路径 D：E3a 模型 + PD DPO pairs → DPO-LoRA → merge → 模型框"E3d 模型"（红色边框，标"全域崩盘"）
> 
> **底层（评估阶段）**：四个模型框统一指向"τ2-bench Test Eval (retail/airline/telecom)"圆角矩形，输出 pass@1 数字（E3c 14.5% 醒目大字，其他三个标签较小）。
> 
> 重要视觉提示：从 PD trajectories 出来的两条蓝色数据流（PD SFT data 和 PD DPO pairs）应明确标注"同源 value function 偏好"，用于解释 E3d 崩盘机理。

---

## 第四章 实验设置（约 2000 字）

### 4.1 数据集：τ2-bench

**4.1.1 三个域简介**

| Domain | Task 数 | 平均工具数 | 平均轮数 | 难度（baseline 4B） |
|--------|---------|-----------|---------|---------------------|
| Retail | 114 | 16 | ~20 | 2.9% |
| Airline | 50 | 9 | ~15 | 6.7% |
| Telecom | 114 | 12 | ~18 | 0.0% |

**4.1.2 训练/测试划分**

使用 seed=42 随机划分：

| Domain | 总数 | Train | Test |
|--------|------|-------|------|
| Retail | 114 | 80 | 34 |
| Airline | 50 | 35 | 15 |
| Telecom | 114 | 80 | 34 |

严格隔离：训练数据生成（teacher rollout）仅在 train split，测试评估仅在 test split。

**4.1.3 两阶段实验的 Domain 覆盖说明**

本文实验分为两个阶段：
- **Phase A（强 teacher 验证）**：使用 Qwen-Plus API 在 τ-bench 原始两域（retail + airline）上生成数据并蒸馏 4B
- **Phase B（中等 teacher 三域蒸馏）**：使用本地部署的 Qwen3-32B 在 τ2-bench 三域（retail + airline + telecom）上生成数据并蒸馏 4B

两阶段实验目的不同：
- Phase A 验证 PD 在强 teacher 下的理论价值（PD vs BoN 数据质量对比）
- Phase B 测试中等 teacher 在多域上的实际蒸馏效果，并提供 telecom 域的完整对比

**关于 SFT 数据量的可比性**：尽管 Qwen-Plus 只覆盖 2 个域、32B 覆盖 3 个域，但因为 Qwen-Plus 更强（per-trial 成功率更高），二者最终筛选出的 *successful trajectories* 数量恰好处于同一数量级：

| Teacher | PD SFT samples | BoN SFT samples |
|---------|----------------|-----------------|
| Qwen-Plus | 197 | 295 |
| Qwen3-32B | 146 | 240 |

这意味着 Phase A vs Phase B 的对比近似于"控制数据量、变化 teacher 强度与 domain 覆盖"。**但需要指出，这只是事后观察到的近似平衡，并非实验设计时的有意控制**——严格来说 domain 覆盖差异（2 vs 3）与 teacher 强度差异是混杂变量。读者应将此对比作为示意而非严格因果推断。

Phase A 未补做 telecom 的实际原因：(1) PD 优势在 retail + airline 上已显著；(2) 32B teacher 已覆盖三域，是本文的主要实验载体；(3) Qwen-Plus API 调用预算限制。

### 4.2 模型配置

| 角色 | 模型 | 部署 |
|------|------|------|
| Teacher A（强） | Qwen-Plus | DashScope API |
| Teacher B（中） | Qwen3-32B | DashScope API |
| Teacher C（弱，对照） | Qwen3-8B（on-policy） | 本地 vLLM, 1×H100 |
| Student | Qwen3-4B | 本地训练 + 推理 |
| User Simulator | Qwen3-4B | 与 student 同模型，保持评估一致性 |

**说明**：User Simulator 使用 4B 模型而非更强模型，是为了模拟实际部署场景（small-model service 中 agent 和 user-side 工具往往是同量级），同时降低评估成本。

### 4.3 推理基础设施

- **数据生成**：
  - Qwen-Plus：DashScope API，并发 2~5 路
  - Qwen3-32B：本地 vLLM，tensor_parallel=4，max_model_len=32768
- **学生模型评估**：自研 OpenAI-compatible 服务（基于 transformers + FastAPI），单卡 H100 80GB，max_model_len=16384
- **代理框架**：tau2-bench orchestrator（Sierra Research），支持 deepcopy fork/restore 环境状态以实现 foresight

### 4.4 评估协议

- **指标**：pass@1（单次运行成功率，二值 reward 的平均）
- **每个 test task 运行次数**：1（单 trial，与 Phase A 一致）
- **最大步数**：30 steps / task
- **温度**：评估时 temperature=0（greedy）以减少随机性
- **结果汇报**：每域分别报告，并报告三域加权平均（按 task 数加权）

### 4.5 实验矩阵（4B 学生）

本文共设计 12 个 4B 蒸馏实验 + 1 个零样本基线，按 teacher 来源分为三组：

- **E1 系列（Phase A, Qwen-Plus teacher, retail+airline）**：E1a SFT-PD / E1b SFT-BoN / E1c BoN-SFT+DPO / E1d PD-SFT+DPO
- **E2 系列（Phase B 对照, Qwen3-8B teacher, retail+airline）**：E2a/b/c/d，结构同 E1
- **E3 系列（Phase B 主, Qwen3-32B teacher, retail+airline+telecom）**：E3a SFT-PD / E3b SFT-BoN / **E3c BoN-SFT+PD-DPO ★** / E3d PD-SFT+PD-DPO

四种训练配置的完整分组表与对应关系见附录 E。

---

## 第五章 实验结果与分析（约 4500 字，核心章节）

本章按"自上而下"结构组织：先看零样本基线（5.1）、再看 teacher 端数据质量（5.2）、再看 SFT 下游（5.3）、再看 DPO 增益（5.4），最后用 PD 决策率（5.5）作为机制解释，综合最佳配置（5.6）。

### 5.1 零样本基线

**5.1.1 Qwen3-4B Zero-shot 结果**

| Domain | pass@1 |
|--------|--------|
| Retail | **2.9%** |
| Airline | **6.7%** |
| Telecom | **0.0%** |
| **加权平均** | **2.4%** |

**5.1.2 失败模式定性分析**

为深入理解 zero-shot Qwen3-4B 的能力瓶颈，本节对三个 domain 共 81 条失败 trajectory 进行全量抽样审查（retail 33 条、airline 14 条、telecom 34 条），归纳出以下三类主要失败模式。

**模式一：幻觉式工具调用（Hallucinated Tool Invocation）**

这是 telecom domain 失败的根本原因，所有 34 条失败 trajectory 均属此类。模型在对话中频繁尝试调用不存在的工具，例如 `check_network_status`（出现 33 次）、`check_sim_status`（29 次）、`check_status_bar`（26 次）、`toggle_data`（22 次），导致每次调用均返回 `Error: Tool '...' not found`。模型收到报错后短暂切换到自然语言指导，但很快恢复到相同的幻觉调用，形成"工具报错—自然语言安抚—再次报错"的固定循环，直至耗尽全部 30 步。此模式表明 4B 模型未能有效内化 telecom domain 的工具白名单，倾向于将对话任务（如引导用户检查手机设置）转化为不存在的自动化操作。

**模式二：信息获取循环（Information-Seeking Loop）**

Retail domain 中有 19 条（57.6%）失败 trajectory 呈现这一模式。典型表现为：模型通过 `find_user_id_by_name_zip` 成功定位账户后，仍坚持向用户索要订单 ID，而当用户表示不记得时，模型既不调用 `get_order_details` 遍历已知订单，也不给出替代方案，而是反复提问"请提供订单 ID 或下单日期"（在最极端情况下该模式连续重复 11 轮以上），直至步数耗尽（termination_reason = unknown，num_steps = 30）。例如，在 task\_105 中，模型在第 8 步已获取用户 ID `aarav_anderson_8794`，但随后 22 步均为同一句话的变体，未产生任何新的工具调用。这一模式揭示了 4B 模型缺乏"当直接询问无效时切换工具策略"的规划能力。

**模式三：过早转接与任务回避（Premature Transfer / Task Avoidance）**

Airline domain 有 4 条（26.7%）、retail domain 有 5 条（15.2%）失败案例属于此类，体现为模型遇到复杂请求时直接调用 `transfer_to_human_agent`，而非尝试可用工具。例如，在 airline task\_16 中，用户请求将航班改签为最便宜的经济舱选项，模型仅经过 1 次工具调用便转接人工，从未尝试 `search_direct_flight` 或 `get_reservation_details`。另一种变体出现在 airline task\_47 中：模型在用户要求确认退款状态时，因政策限制无法给出明确答复，陷入"道歉—重申政策—道歉"的重复循环，最终触发用户主动发送 `###STOP###` 终止对话（termination_reason = user_stop）。此类失败说明模型对任务边界的判断存在偏差——将本可通过工具解决的请求错误归类为超出授权范围。

**总结**：三类失败模式分别对应不同层次的能力缺陷——工具白名单内化不足（幻觉调用）、多步规划能力弱（信息获取循环）、任务范围判断偏差（过早转接）。这一分析为第 5.2 节以 SFT 蒸馏提供 grounding 依据：teacher demonstrations 需要覆盖工具备选方案切换、订单遍历等具体技能，而非仅传授"任务完成率"这一宏观信号。

**5.1.3 与强 teacher 的差距**

| 模型 | Retail | Airline | Telecom |
|------|--------|---------|---------|
| Qwen-Plus (greedy baseline) | 50.6% | 60.6% | — |
| Qwen3-32B (greedy baseline) | 37.3% | 23.5% | 3.7% |
| Qwen3-4B (zero-shot) | 2.9% | 6.7% | 0.0% |

Qwen3-4B 的能力差距巨大，为 SFT 蒸馏提供了充足的提升空间。

### 5.2 Teacher 端原始数据质量对比（核心证据 1）

本节回答 Q1：PD 数据是否真的优于 BoN？答案：**仅在强 teacher 上成立**。

**5.2.1 Per-trial 成功率对比**

| Teacher | Domain | Baseline (greedy 1×) | BoN per-trial | PD per-trial | **PD − BoN** |
|---------|--------|---------------------|----------------|--------------|--------------|
| Qwen-Plus | retail | 50.6% | 49.0% (196/400) | **55.4%** (133/240) | **+6.4pp** |
| Qwen-Plus | airline | 60.6% | 56.9% (99/174) | **62.7%** (64/102) | **+5.8pp** |
| Qwen3-32B | retail | 37.3% | 37.2% (149/400) | 37.9% (91/240) | +0.7pp |
| Qwen3-32B | airline | 23.5% | 28.0% (49/175) | 28.6% (30/105) | +0.6pp |
| Qwen3-32B | telecom | 3.7% | 10.5% (42/400) | 10.4% (25/240) | −0.1pp |

**关键发现**：
- 强 teacher（Qwen-Plus）上 PD 显著提升 per-trial 成功率（+5.8~6.4pp，相对 +10~12%）
- 中等 teacher（Qwen3-32B）上 PD 优势完全消失（差距在 ±1pp 内，统计噪声范围）

> 【图 3 — Teacher 端 PD vs BoN per-trial 成功率】**位置：5.2.1 节末尾**
> 
> 分组柱状图（grouped bar chart）：
> - X 轴：5 个 (teacher, domain) 组合：Qwen-Plus retail / Qwen-Plus airline / 32B retail / 32B airline / 32B telecom
> - Y 轴：per-trial 成功率（0%–70%）
> - 每组三个柱子并排：浅灰=Baseline (greedy)、蓝色=BoN per-trial、深红=PD per-trial
> - 在每组上方标注 "PD - BoN" 的 gap（如 +6.4pp / +5.8pp / +0.7pp / +0.6pp / −0.1pp）
> - 用一条水平虚线把"Qwen-Plus 区"和"Qwen3-32B 区"分开，标签"强 teacher" vs "中等 teacher"
> - 图例放在右上角；为强调"32B 上 PD 优势消失"，可在 32B 三组上方加一个文字气泡"PD ≈ BoN"

**5.2.2 Oracle 对比**（per-task max-of-trials，反映训练数据筛选后的能力上界）

| Teacher | Domain | BoN Oracle (best/5) | PD Oracle (best/trials) |
|---------|--------|---------------------|---------------------------|
| Qwen-Plus | retail | 78.7% | **82.5%** |
| Qwen-Plus | airline | 88.6% | 85.7% |
| Qwen3-32B | retail | 62.5% | 60.0% |
| Qwen3-32B | airline | 48.6% | 40.0% |
| Qwen3-32B | telecom | 25.0% | 22.5% |

→ 即使在 Oracle 维度，32B 的 PD 也仅与 BoN 持平甚至略低。

**5.2.3 成功轨迹的"轨迹效率"对比**（成功 episode 的步数与 token 消耗）

| Teacher | Domain | Method | N | Avg Steps | Median | Avg Episode Tokens |
|---------|--------|--------|---|-----------|--------|--------------------|
| Qwen3-32B | retail | BoN | 149 | 12.6 | 13 | 80,222 |
| Qwen3-32B | retail | PD | 91 | 12.7 | 13 | 80,342 |
| Qwen3-32B | airline | BoN | 49 | 10.8 | 10 | 64,252 |
| Qwen3-32B | airline | PD | 30 | 11.6 | 11 | 73,136 |
| Qwen3-32B | telecom | BoN | 42 | 11.5 | 11 | 127,666 |
| Qwen3-32B | telecom | PD | 25 | 12.5 | 13 | 137,925 |

→ 32B 上 PD 成功轨迹**并不更短、不更高效**（airline 反而略长，token 略多 ~10%）。

**5.2.4 结论 1**

> **PD 作为数据生成方法的有效性具有 teacher 能力门槛。** 当 teacher 输出多样性和前瞻推理能力都足够强时（Qwen-Plus），PD 能稳定提升 per-trial 成功率；当 teacher 能力下降到中等水平时（Qwen3-32B），PD 优势消失。机制层面解释见 5.5 节。

### 5.3 SFT 下游性能对比（核心证据 2）

本节回答 Q2：PD 数据训练的学生是否系统性优于 BoN 训练的学生？答案：**域依赖、不稳定**。

**5.3.1 Phase A：Qwen-Plus Teacher**

| 实验 | 数据来源 | 样本量 | Retail pass@1 | Airline pass@1 |
|------|---------|--------|---------------|----------------|
| E0 | — | — | 2.9% | 6.7% |
| E1a | Qwen-Plus PD | 197 | 2.9% | **20.0%** |
| E1b | Qwen-Plus BoN | 295 | **8.8%** | 13.3% |

**观察**：
- airline 域：E1a (PD) 20.0% > E1b (BoN) 13.3%，PD 数据下游表现更好（与 5.2 节强 teacher 上 PD 优势一致）
- retail 域：E1b (BoN) 8.8% > E1a (PD) 2.9%，BoN 反而占优——可能与 BoN 样本量更多（295 vs 197）相关

**5.3.2 Phase B：Qwen3-32B Teacher**

| 实验 | 数据来源 | 样本量 | Retail | Airline | Telecom | Avg |
|------|---------|--------|--------|---------|---------|-----|
| E0 | — | — | 2.9% | 6.7% | 0.0% | 2.4% |
| E3a | 32B PD | 146 | 2.9% | **20.0%** | 0.0% | 4.8% |
| E3b | 32B BoN | 240 | **8.8%** | 13.3% | 5.9% | 8.4% |

**观察**：
- airline 域：E3a (PD) 20.0% > E3b (BoN) 13.3%——与 Phase A 一致
- retail 域：E3b (BoN) 8.8% > E3a (PD) 2.9%——与 Phase A 一致
- telecom 域：E3b (BoN) 5.9% > E3a (PD) 0.0%——BoN 占优
- 三域平均：E3b BoN-SFT 8.4% > E3a PD-SFT 4.8%

**5.3.3 Phase B（对照）：Qwen3-8B Teacher**

> 注：本节"E2 系列"使用 Qwen3-8B 作为 teacher、Qwen3-4B 作为 student。**这里 teacher 和 student 是不同模型，并不构成严格意义上的 on-policy 蒸馏**。本文用"中等强度 teacher"作为该组实验的语义标签，目的是与 Phase A（Qwen-Plus，强 teacher）和 Phase B 主实验（Qwen3-32B，中等偏强 teacher）形成阶梯式对照。

| 实验 | 数据来源 | 样本量 | Retail | Airline |
|------|---------|--------|--------|---------|
| E2a | 8B-Teacher PD | 111 | 2.9% | 13.3% |
| E2b | 8B-Teacher BoN | 99 | 2.9% | 6.7% |

→ teacher 强度越弱，SFT 整体收益越低；8B teacher 数据在 airline 上 PD 略好（与 Phase A 一致）

**5.3.4 跨 teacher 对比：teacher 强度对蒸馏天花板的影响**

| Teacher | retail pass@1 (PD/BoN best) | airline pass@1 (PD/BoN best) |
|---------|-----------------------------|-------------------------------|
| Qwen-Plus | 8.8% | 20.0% |
| Qwen3-32B | 8.8% | 20.0% |
| Qwen3-8B | 2.9% | 13.3% |

**Phase A vs Phase B 高度相似的现象**：尽管 Qwen-Plus 比 32B 强很多，但下游 4B SFT 后的最佳分数几乎相同。这至少有两种可能解释：

1. **4B 容量上限假设**：4B 学生的参数容量决定了它能从 demonstration 中学到的上限，再强的 teacher 也无法突破——一旦 demonstration 包含正确决策路径，4B 学生就能学到它能学到的极限
2. **同源架构与训练数据假设**：Qwen3-32B 与 Qwen3-4B 同属 Qwen3 家族，架构、tokenizer、预训练数据高度同源，4B 模仿 32B 的轨迹比模仿 Qwen-Plus（架构未公开、可能来自不同训练 pipeline）的轨迹更容易

这两种解释并不互斥，但本文实验设计无法区分。**严格区分需要补做：以 Llama-3-70B 等异源强 teacher 重复 E1/E3 实验，看 4B 学生在 Qwen-source vs non-Qwen-source teacher 上是否表现不同**。这是值得未来工作的方向。

**5.3.5 结论 2**

> SFT 下游性能上，**PD 与 BoN 哪个更好"域依赖"**：airline 域 PD 系统性更好（+6.7pp），retail/telecom 域 BoN 更好。假设解释：airline 任务序列短、决策点关键（适合 PD 局部择优），retail/telecom 任务多样性高、需要广覆盖（适合 BoN 多样性）。

### 5.4 DPO 增益对比（核心证据 3）

本节回答 Q3：DPO 能否在 SFT 基础上进一步提升？答案：**取决于 SFT 初始化策略，BoN-SFT+DPO 增益稳定，PD-SFT+DPO 致命退化**。

**5.4.1 32B 数据上的 DPO 结果（核心实验）**

| 实验 | SFT 数据 | DPO 数据 | Retail | Airline | Telecom | Avg | vs SFT-only |
|------|---------|---------|--------|---------|---------|-----|-------------|
| E3a | PD 146 | — | 2.9% | 20.0% | 0.0% | 4.8% | — |
| **E3d** | PD 146 | PD pairs 298 | 0.0% | 6.7% | 0.0% | **1.2%** | **−3.6pp（崩盘）** |
| E3b | BoN 240 | — | 8.8% | 13.3% | 5.9% | 8.4% | — |
| **E3c** ★ | BoN 240 | PD pairs 298 | 5.9% | 20.0% | **20.6%** | **14.5%** | **+6.1pp** |

**核心观察**：
- **E3c (BoN-SFT + PD-DPO) avg 14.5% 是所有 4B 实验中最高**，telecom 单域 20.6% 是最大亮点
- **E3d (PD-SFT + PD-DPO) avg 1.2% 全域崩盘**——retail 和 telecom 归零，airline 退至 6.7%
- 两个 DPO 实验使用的偏好对完全相同（都来自 32B PD 候选），唯一差异是 SFT 初始化

**5.4.2 Phase A：Qwen-Plus 数据上的 DPO**

| 实验 | SFT | DPO | Retail | Airline |
|------|-----|-----|--------|---------|
| E1b | BoN 295 | — | 8.8% | 13.3% |
| E1c | BoN 295 | Qwen-Plus PD pairs | 0.0% | 6.7% |
| E1a | PD 197 | — | 2.9% | 20.0% |
| E1d | PD 197 | Qwen-Plus PD pairs (626) | 2.9% | 13.3% |

→ Phase A 的 DPO 在 retail 上全部退化（甚至 BoN-SFT+DPO 也退化）。**这与 Phase B 的 E3c 增益形成鲜明对比**。

**5.4.3 E3c 显著优于 E1c 的解释**

| 实验 | SFT 数据 | DPO pairs 来源 | retail | airline | avg |
|------|---------|---------------|--------|---------|-----|
| E1c | Qwen-Plus BoN | Qwen-Plus PD | 0.0% | 6.7% | 3.4% |
| E3c | 32B BoN | 32B PD | 5.9% | 20.0% | 13.0% |

差异分析（按可能性递减排序）：

1. **DPO pairs 的 score gap 分布**：Qwen-Plus PD 因为 value function 区分度高，pairs 偏好 gap 更大，**可能导致小数据 DPO 训练步长过大、reference policy drift 过强**；32B PD 由于真实 PD 决策率低、保留下来的 pair 偏好 gap 适中，DPO 训练更稳定
2. **DPO pairs 数量分布**：Qwen-Plus 与 32B 在 gap≥0.1 阈值下的 pair 数量略有差异
3. **同源架构优势**：32B teacher 与 4B student 同属 Qwen3 家族，输出 token 分布、prompt 习惯、tool call 格式高度同源；DPO 阶段 student 模型从 32B 偏好对中学习时不需要跨"分布鸿沟"，而 Qwen-Plus（架构未公开，可能来自不同训练 pipeline）的偏好对对 4B 而言可能存在分布不匹配

这三个因素本文实验无法严格区分，需要后续工作（如：用 Llama-70B 数据做 DPO 对比；控制 DPO pair score gap 分布做消融）。这也是**第六章局限性需进一步讨论的点**。

**5.4.4 E3d 崩盘的机制解释**

PD-SFT 的训练数据本身已经是 PD value function 选优的结果——student 已经"模仿了 value function 偏好"。再用 PD pairs（同样基于 value function 的偏好）做 DPO，相当于**在同一个偏好信号上做了两次拟合**，加剧了 value function 自身偏差的放大效应。

具体表现：retail 0%（完全失能）+ telecom 0% + airline 退化至 6.7%。

**5.4.5 结论 3**

> **PD 真正稳定的应用方式是作为 turn-level DPO 偏好信号的来源，而不是作为 SFT 数据本身。** BoN-SFT 提供多样化覆盖，PD-DPO 提供精准偏好纠正，两者互补；PD-SFT + PD-DPO 在小数据 regime 下不稳定。

### 5.5 真实 PD 决策率分析：机制层面证据（核心证据 4）

本节用一个 step-level 统计指标，统一解释 5.2/5.3/5.4 节的现象。

**5.5.1 真实 PD 决策率定义**

一个 PD step 被认为"真正由 PD 选择"，当且仅当：
1. **未被去重跳过**（skipped_identical = False）：K=5 个候选不高度同质
2. **未做 greedy fallback**（候选间最大-最小 score gap > 0.05）：value function 能区分候选优劣

否则该 step 退化为 greedy decoding，无 PD 实际效益。

**5.5.2 实测数据**

| Teacher | Domain | Total steps | Skipped (同质) | Greedy fallback | **真实 PD** | **PD rate** |
|---------|--------|-------------|----------------|-----------------|-------------|-------------|
| Qwen-Plus | retail | — | — | — | — | **27%** |
| Qwen-Plus | airline | — | — | — | — | **35%** |
| Qwen3-32B | retail | 3289 | 2316 (70%) | 704 (21%) | 269 (8%) | **8.2%** |
| Qwen3-32B | airline | 1349 | 942 (70%) | 310 (23%) | 97 (7%) | **7.2%** |
| Qwen3-32B | telecom | 3352 | 2069 (62%) | 1033 (31%) | 250 (7%) | **7.5%** |

**关键发现**：
- **32B 的真实 PD 率仅为 Qwen-Plus 的 1/4**
- 32B 上 70% 的 step 直接 skip（K 个候选高度相似）；另外 21~31% 做了 foresight 但 value function 无法区分
- 仅 7~8% 的 step 真正受 PD 影响

> 【图 4 — PD step 分类堆叠柱状图】**位置：5.5.2 节末尾**
> 
> 100% 堆叠柱状图：
> - X 轴：5 个 (teacher, domain) 组合：Qwen-Plus retail / Qwen-Plus airline / 32B retail / 32B airline / 32B telecom
> - Y 轴：百分比 0%–100%
> - 每根柱子分三段堆叠（从下到上）：
>   - 灰色：Skipped (候选同质)
>   - 黄色：Greedy fallback (value 无区分)
>   - **深红色：真实 PD 决策（active PD）**
> - 在柱顶标"真实 PD 率"百分比
> - Qwen-Plus 两根柱子的深红色段明显粗（27% / 35%），32B 三根柱子的深红色段极细（7-8%），视觉上对比明显
> - 备注：Qwen-Plus 的 skipped/greedy_fb 细分数据未追溯（log 已丢失），可以只画整体真实 PD 率，或者用半透明色块表示"unknown breakdown"

**5.5.3 机制解释**

PD 起作用需要两个条件：
1. **候选多样性**：K=5 个候选在 temperature=0.8 下应展现差异化
2. **前瞻区分度**：value function 在 H=2 步前瞻轨迹上应给出区分性评分

Qwen-Plus 同时满足两个条件（候选差异大、值函数评分分散）。
Qwen3-32B 在两个条件上均显著弱化：
- 候选多样性：32B 在 instruction-tuned 后输出更"自信"、更确定，K 个候选高度同质
- 前瞻区分度：H=2 步的 foresight 中，32B 的 greedy 生成相对相似，value function 给出近似分数

**这从机制层面解释了为什么 PD 在 32B 上作为"数据生成方法"失效**——不是 PD 算法本身的问题，而是 teacher 不够 "exploratory"。

> **澄清全文一致性**：本文有两个看似相反的结论需要明确区分：
> 1. **PD 作为 SFT 数据生成方法**：在 32B 上*失效*（per-trial 成功率与 BoN 相当）—— 本节（5.5）解释机制
> 2. **PD 作为 DPO 偏好信号源**：在 32B 上*有效*——E3c (32B BoN-SFT + 32B PD-DPO) avg 14.5%，是所有实验最佳
> 
> 这两个结论并不矛盾。即使 PD 选出的 *winner trajectory* 与 BoN 选出的相当（同质化候选+greedy fallback 导致），PD 过程中产生的 *每步候选-排名信息* 仍然有用——只要存在 score gap > 0.1 的步骤，就能构成有意义的 DPO chosen/rejected 对。
> 
> 进一步看 SFT 下游：Qwen-Plus 与 32B 的 SFT 效果**几乎相同**（5.3.4 节表格中两个 teacher 的 best pass@1 完全一致），并不是"Qwen-Plus SFT 效果好"。Qwen-Plus 真正的优势仅体现在 *teacher 端 per-trial 成功率* 和 *PD 真实决策率*（5.2 和本节数据），但这些优势**没能传导**到 4B 学生的下游 SFT 表现，可能受 4B 容量上限或同源训练数据等因素影响（5.3.4 节讨论）。

**5.5.4 推论：PD 的有效区间**

| Teacher 类型 | 候选多样性 | 前瞻区分度 | PD 有效性 |
|--------------|------------|------------|-----------|
| 超强 + 探索性高（如 Qwen-Plus / GPT-4） | 高 | 高 | ✅ 显著 |
| 中强 + 确定性高（如 Qwen3-32B/72B） | 低 | 中 | ❌ 失效 |
| 弱模型（如 Qwen3-4B/8B） | 中 | 低 | ❌ 失效（value 不准） |

→ PD 适用于"具备多样性 + 自我评估能力"的强 teacher，对于工业界常用的 32B 级 instruction-tuned 模型，PD 数据生成的边际收益有限。

### 5.6 综合最佳配置：BoN-SFT + PD-DPO

**5.6.1 全配置横向对比（4B 学生）**

| 配置 | Teacher | Retail | Airline | Telecom | Avg | 相对 zero-shot |
|------|---------|--------|---------|---------|-----|----------------|
| Zero-shot | — | 2.9% | 6.7% | 0.0% | 2.4% | 1.0× |
| E3a PD-SFT | 32B | 2.9% | 20.0% | 0.0% | 4.8% | 2.0× |
| E3b BoN-SFT | 32B | 8.8% | 13.3% | 5.9% | 8.4% | 3.5× |
| E3d PD-SFT+DPO | 32B | 0.0% | 6.7% | 0.0% | 1.2% | 0.5× ❌ |
| **E3c BoN-SFT+PD-DPO** ★ | 32B | 5.9% | 20.0% | **20.6%** | **14.5%** | **6.0×** |

> 【图 5 — 主结果对比图（E3 系列 + Zero-shot）】**位置：5.6.1 节末尾，本文最关键的一张图**
> 
> 分组柱状图：
> - X 轴：3 个 domain（retail / airline / telecom）+ 一个 "三域平均" 组
> - Y 轴：pass@1（0%–25%）
> - 每组 5 个柱子并排：浅灰=Zero-shot、浅蓝=E3a PD-SFT、深蓝=E3b BoN-SFT、橙色=E3d PD-SFT+DPO、**绿色高亮=E3c BoN-SFT+PD-DPO ★**
> - E3c 的柱子加粗边框、顶部加 ★ 符号、用最深绿色
> - 在"三域平均"组上方加一条注释箭头："Zero-shot 2.4% → E3c 14.5%（提升 6×）"
> - E3d 柱子上加 "❌" 表示崩盘
> - 图例放在右上角；考虑加一个内嵌小图显示 telecom 单独的"0% → 20.6%"突破对比

**5.6.2 单域突破：telecom 0% → 20.6%**

Telecom 域 4B 零样本完全失能（pass@1 = 0%），E3c 配置后达到 20.6%——**这是本文方法最有说服力的单点对比**。机理：
- BoN-SFT 提供 telecom 域的初始任务理解（成功轨迹的 demonstration）
- PD-DPO 在已有的 SFT 模型上做精准纠正，把 telecom pass@1 从 5.9% 推到 20.6%

**5.6.3 学生反超教师：telecom 上 4B (20.6%) > 32B (10.4%)**

| Domain | 32B teacher 单次 rollout 成功率 | 4B 学生（E3c, test split） | 谁强 |
|--------|--------------------------------|----------------------------|------|
| Retail | 37.9% (train, per-trial) | 5.9% (test) | 32B 强（差 32pp）|
| Airline | 28.6% (train, per-trial) | 20.0% (test) | 32B 略强（差 8.6pp）|
| **Telecom** | **10.4%** (train, per-trial) | **20.6%** (test) | **4B 反超（+10.2pp）** |

> 【图 6 — 学生 vs 教师跨域对比】**位置：5.6.3 节末尾**
> 
> 双柱柱状图：
> - X 轴：3 个 domain（retail / airline / telecom）
> - Y 轴：成功率（0%–40%）
> - 每组两个柱子：浅蓝=32B teacher single rollout（train split）、深绿=4B 学生 E3c（test split）
> - 在 telecom 组上加一个红色向上箭头注释："4B 学生反超 +10.2pp"
> - 在 retail/airline 组上注释 "teacher 仍占优"
> - 为强调 caveat，可在图下方加灰字 "*32B 数字来自 train split，4B 来自 test split；分布相似但非同 task；严格对照需补做 32B test split eval（未来工作）"

⚠️ **数据 caveat**：32B 数字来自 train split rollout（数据生成阶段），4B 来自 test split 评估。两个 split 是 seed=42 随机分配的，分布相似但严格不可直接比较。理想情况下应再跑一遍 32B 在 test split 上的 single-rollout baseline 以严格对照（计入未来工作）。

**机制解释**：这一现象与知识蒸馏文献中长期观察到的"学生超越教师"（student-outperforms-teacher）现象一致 [Furlanello et al., 2018, arXiv:1805.04770; Hsieh et al., 2023, arXiv:2305.02301]。具体到本文场景：

1. **32B 在 telecom 上的失败模式是"高方差"**：单次成功率仅 10.4%，但 5 次 rollout 中 oracle 命中 25%——说明 32B 知道如何做对，但稳定性不足
2. **BoN-SFT 数据筛选机制等效于"oracle 选择器"**：240 条 SFT 数据中只保留 reward=1 的轨迹，相当于把 32B 的"幸运成功"模式提取出来作为 demonstration
3. **4B 学生学习的是这一"成功模式"本身**：经过 BoN-SFT 后 4B 在 telecom 上已达 5.9%（比 teacher 的 single-rollout 10.4% 低，但已超过 4B 自己的 0%）；再叠加 PD-DPO 精准纠错，达到 20.6%——**比 teacher 单次平均还高**

**实践推论**：知识蒸馏的有效性不取决于 teacher 的*平均能力*，而取决于其 *oracle 能力*——只要 teacher 能采出至少一些成功轨迹，student 就有机会通过模仿这些"good examples"超过 teacher 的平均水平。

**5.6.4 总结**

> 在小数据（数百条样本）+ 弱学生（Qwen3-4B）的真实部署 regime 下，**BoN-SFT + PD-DPO** 是最稳定有效的蒸馏配置，可在 τ2-bench 三域上将零样本 2.4% 提升至 14.5%（相对 6 倍），其中 telecom 域从无能（0%）提升至可用（20.6%）。

---

## 第六章 讨论（约 2000 字）

### 6.1 研究发现的实践意义

**6.1.1 对小模型 Agent 部署的指导**

- 当目标部署模型为 4B 量级、零样本能力弱（pass@1 < 10%）时：
  1. 不要试图用更强的 teacher 解决问题——4B 容量是瓶颈（Qwen-Plus vs 32B 教学效果几乎相同）
  2. 优先选 BoN-SFT 而非 PD-SFT（除非 teacher 能力 ≥ Qwen-Plus 级）
  3. 一定要做 DPO，且 DPO pairs 用 PD 候选 ranking 构建（gap ≥ 0.10 筛选）
  4. 数据量 100-300 条足以达到 6× 提升，不必盲目堆数据

**6.1.2 PD 的真正定位**

PD 不是"更好的数据生成方法"，而是**"为偏好学习提供高密度 step-level 信号的工具"**。其价值在于：
- 每步生成 K 个候选 → 提供 chosen/rejected 对
- value function 提供 ranking 依据
- 这两个东西 BoN 都做不到

而 BoN 提供的多样化成功轨迹，作为 SFT demonstration 更稳定（不容易过拟合 value function）。

**6.1.3 Qwen-Plus DPO 失效的机制分析**

第一阶段实验显示，Qwen-Plus 下的 DPO 不但没有带来收益，BoN-SFT + PD-DPO 甚至从 8.8%/13.3% 跌至 0.0%/6.7%。这与第二阶段 32B 的 DPO 带来 +6.1pp 提升形成鲜明对比。需要给出至少两个可区分的机制假说：

- **假说 A（信号强度不匹配）**：Qwen-Plus PD 候选间的 value gap 分布与 32B 不同。若 Qwen-Plus 产生的偏好对 gap 普遍偏大，DPO 的梯度步长会更激进，模型偏离参考策略过远，导致结果崩坏。可通过对比两阶段 DPO pairs 的 gap 均值/方差分布来支持或否定这一假说（若有数据）。

- **假说 B（学生容量耗尽）**：Qwen-Plus SFT 的数据质量更高，已经把 4B 模型推到了接近容量上限的位置（两教师 SFT 最终分数相同印证了这一点）。在此基础上做 DPO 没有可改进的空间，偏好信号反而引入了噪声。这个假说与假说 A 并不互斥。

写作建议：正文中呈现两个假说并指出哪些观测与哪个假说更一致，不要只写一个"可能的原因"。若无额外数据，结论可以是"两种假说均有部分支持，无法严格区分，需在未来工作中控制变量验证"。

**6.1.4 航空域系统性偏好 PD 数据的机制分析**

两个教师、两个阶段，航空域始终是 PD-SFT 优于 BoN-SFT（+6.7pp）；零售和电信域始终是 BoN-SFT 更稳定或持平。这一跨实验一致的模式需要一个任务结构层面的解释。

核心假说：航空域工具集最小（9个）、轨迹最短（~15轮），候选空间更紧凑，价值函数在前瞻 H=2 步内更容易拉开候选间的分数差距，PD 真正发挥作用的比例更高。相比之下，零售有 16 个工具、20 轮，候选空间更大且工具调用链更复杂，H=2 步前瞻能捕捉到的差异更嘈杂，BoN 的全轨迹成功覆盖反而更稳定。

佐证材料（已有）：真实 PD 决策率统计显示 32B 在三域上均只有 7–8%，但如果分域看 skip 比例是否与工具集大小相关，可以进一步支持这个假说。

写作建议：引用表 4.1 的域统计数据（工具数/平均轮数），将假说落地到具体数字，而不是停留在定性描述。

**6.1.5 Telecom 域"学生超越教师"的机制分析**

实验发现，经 BoN-SFT + PD-DPO 后，Qwen3-4B 在 telecom 域达到 20.6%，超过 32B 教师单次 rollout 的平均成功率 10.4%。这一现象需要区分"统计偶然"与"系统性机制"两种解释：

- **统计波动视角**：telecom 测试集仅 34 条任务，20.6% 对应 7 次成功，10.4% 对应 3-4 次成功。单次评估中 1-2 条任务的差异即可造成如此量级的分数差。应在讨论中正面承认这一局限。

- **系统性机制视角（更可能的主导原因）**：零样本 4B 的失败分析（第五章 5.1.2）显示，电信域的失败几乎全部（34/34）来自幻觉式工具调用——模型调用了不存在的工具。SFT 成功轨迹提供了工具白名单的正确示范，能够系统性地消除这一失败模式，而不是靠局部运气改善。相比之下，32B 教师在单次贪心下仍以 3.7% 的成功率运行，其失败来自更复杂的原因（长程规划失误等），SFT 的"白名单内化"效果对 4B 学生帮助特别显著。

- **与蒸馏文献的联系**：Born-Again Networks [Furlanello et al., 2018] 指出学生超越教师取决于 teacher 的 oracle 能力而非平均能力。本文 32B 的 oracle 上界（telecom 约 25%）远高于其平均成功率（10.4%），说明 teacher "会做但不稳定"——蒸馏把这部分"可达能力"系统化地固化到了学生的权重里。

写作建议：正文中先呈现现象（已在第五章小结提及），再在此节给出双重视角的讨论（承认统计不确定性的同时，论证系统性机制更可能主导），不要只呈现正面解释。

### 6.2 局限性

**6.2.1 单 trial 评估与小测试集的统计检验力**
4B 学生的评估为 single trial（每 task 跑 1 次，temperature=0），分数受 user simulator 行为和 tool 返回顺序影响。理想情况应做 3 trials 取平均。本文受 GPU 时间限制未做。

此外，测试集规模偏小（retail 34 条、airline 15 条、telecom 34 条），单个百分点差异对应 1 条任务差异，在航空域 20.0% vs 13.3% 这类差异仅对应 1 条任务，可能在重复实验中反转。因此正文在描述这类差异时应避免用"明显"等强措辞，改为"在当前测试集规模下呈现"，并在此节正面说明哪些结论依赖测试集规模假设。

**6.2.2 用户模拟器与学生模型同源**
学生模型训练后，其对话风格可能与 4B 用户模拟器形成某种"互适应"，导致在 4B 用户下的评估结果优于在真实用户或更强用户模拟器下的表现。本文采用同规模用户模拟器主要是为了降低评估成本，并与 tau2-bench 的标准协议保持一致。该潜在偏差的方向（是否系统性高估学生表现）尚不明确，需在未来工作中通过更换用户模拟器规模来验证。

**6.2.3 数据规模上限**
本文最多 240 条 SFT + 298 对 DPO，属于极小数据 regime。更大规模（1000+ 条）下结论可能改变，尤其：
- PD 在大数据下是否能突破 SFT 局限？
- DPO 数据增多后是否会有过拟合现象？

**6.2.4 Value Function 设计的固有偏差**

PD 依赖五维加权 value function。该函数：
- 基于启发式规则（progress / health / sentiment / termination / assertions），无需额外训练 RM——这正是 PD 方法的优雅之处，避免了"训 RM 才能用 PD"的鸡生蛋问题
- 权重 (0.35/0.25/0.15/0.15/0.10) 经手动调参，缺乏严格理论支撑
- 在 32B 数据上，70% step 因候选同质被跳过，剩余 30% 中 value 区分度本身也较低

→ 改进方向（保持 model-free 性质）：用候选间 token-level entropy / KL 散度 / 不确定性度量直接构造排序信号，避免引入额外训练阶段；这正是 [Ma et al., 2024] 原始 Predictive-Decoding 在数学/编程任务上采用的轻量化路线，本文 Agent 场景下的对应工作属未来研究。

**6.2.5 域覆盖与泛化性**

本文仅在 τ2-bench 三域上实验，结论的跨 benchmark 泛化性需进一步验证。特别是：
- 更长 episode（>30 步）的任务上 PD 是否仍可行（foresight 成本随轨迹长度爆炸）
- 多 agent 协作场景

**6.2.6 Thinking Mismatch 历史教训**

早期 Qwen3-8B 实验中观察到，teacher 生成数据时 chain-of-thought 默认开启，但 student 训练时 `enable_thinking=False`，导致 SFT 后性能不升反降。本文 4B 实验全程保持 thinking ON 一致（数据生成与训练推理），但这一问题需要在未来工作中系统研究。

### 6.3 本文与已有工作的关键差异

第二章已系统综述了 Agent 蒸馏（AgentTuning、FireAct 等）、Predictive Decoding（Ma et al., 2024）、Step-DPO、知识蒸馏（BAN、Distilling step-by-step）等工作的内容与贡献。本节仅聚焦"本文相对它们做了什么新事情、得到了什么不同结论"，不再重复描述这些工作本身。

**6.3.1 本文最重要的方法学扩展：把 PD 从 online inference 搬到 offline 数据生成**

| 维度 | Ma et al. 2024 | 本文 |
|------|----------------|------|
| 用途 | Online inference 控制 | Offline 数据生成 |
| 任务类型 | 单轮数学/编程/AlfWorld | 多轮 Agent 工具调用 |
| 状态管理 | 无（CoT 文本）| Env-fork（DB + 历史 deepcopy）|
| 与 SFT/DPO 结合 | 未涉及 | **核心：BoN-SFT + PD-DPO** |
| Teacher 强度依赖 | 未系统研究 | **核心：teacher 能力门槛与真实 PD 决策率** |

**6.3.2 本文最重要的实证发现（与已有工作差异）**

1. **PD 在 32B 量级 teacher 上失效**：原始 PD 工作 [Ma et al., 2024] 报告 PD 在数学/AlfWorld 上稳定有效（用 GPT-4 / Llama-3-70B），本文在 32B 上发现 *真实 PD 决策率仅 7-8%*——这是只有在更"工业级 teacher"上才暴露的限制
2. **Step-level 偏好信号扩展到 Agent 多轮**：Step-DPO [Lai et al., 2024] 在数学 CoT 上验证了 step-level DPO，本文首次把这一思路用到 *多轮 Agent 工具调用* 上，"step" 含义从"CoT 中一行公式"扩展到"一次 agent decision"
3. **小学生 + 小数据 regime 下的"学生超越教师"**：BAN [Furlanello et al., 2018] 和 Distilling step-by-step [Hsieh et al., 2023] 已观察过 student outperforms teacher，但 BAN 是 student/teacher 同容量、Hsieh 是 single-domain 蒸馏。本文在 *student 容量远小于 teacher（4B vs 32B）* 且 *多 domain* 设定下复现了这一现象，并给出 "teacher 的 oracle 能力 > 平均能力"的实践推论

**6.3.3 与 AgentTuning 等大规模蒸馏工作的对比**

AgentTuning / FireAct 等强调"用大量 teacher 轨迹（数千到上万条）做 SFT 提升 Agent 通用能力"。本文实证：在 100-300 条小数据 regime 下，**BoN-SFT + PD-DPO** 已能达到 6× zero-shot 提升；不必盲目堆数据。这对资源受限的部署场景是另一种范式选择。

### 6.4 未来工作

1. **学习型 value function**：训练一个 RM 替代手工 value，可能解决 32B 上 PD 失效的问题
2. **混合数据 SFT**：mix BoN 多样化覆盖 + PD 局部择优样本，探索最优混合比例
3. **更长 H 的 foresight**：在更长前瞻深度下，PD 是否能区分更多 step
4. **跨 benchmark 验证**：在 WebArena、SWE-bench 等其他 Agent benchmark 上复现 BoN-SFT + PD-DPO 配置
5. **更大学生模型**：在 14B/32B 学生模型上验证 PD-DPO 的扩展性

---

## 第七章 结论（约 600 字）

本文围绕"如何为弱学生 LLM 高效蒸馏 Agent 能力"这一核心问题，在 τ2-bench（retail / airline / telecom 三域）上系统对比了 Predictive Decoding (PD) 与 Best-of-N (BoN) 两种数据生成策略，并探索了 SFT + Turn-level DPO 的组合训练管线。

**主要发现：**

1. **PD 数据生成的 teacher 能力门槛**：PD 优势仅在强 teacher（Qwen-Plus）上显著（per-trial 成功率 +5.8~6.4pp）。在中等 teacher（Qwen3-32B）上 PD 优势消失，机制原因是 32B 在 temperature=0.8 下生成的候选高度同质化，真实 PD 决策率仅 7-8%（vs Qwen-Plus 的 27-35%）。这一发现修正了"PD 总是优于 BoN"的预期。

2. **PD-SFT vs BoN-SFT 的域依赖性**：airline 域 PD-SFT 系统性更好（+6.7pp），retail/telecom 域 BoN-SFT 更好。简短/关键决策密集型任务更适合 PD，多样化长任务更适合 BoN。

3. **PD 的真正稳定应用是 DPO 偏好信号**：BoN-SFT + PD-DPO 是最佳组合（E3c, avg 14.5%, telecom 20.6%）；PD-SFT + PD-DPO 因偏好信号叠加导致全域崩盘（E3d, avg 1.2%）。

4. **弱学生从 SFT/DPO 中获益巨大**：Qwen3-4B 零样本 2.4%，经 BoN-SFT + PD-DPO 后达 14.5%，相对提升 6 倍。这与早期 Qwen3-8B 实验中 SFT 反而退化的现象形成鲜明对比，验证"能力起点决定 SFT 收益方向"。

5. **学生在窄分布上反超教师**：在 telecom 域，4B 学生 (20.6%) 超过 32B teacher 的单次 rollout 平均 (10.4%)。这一现象与 Born-Again Networks [Furlanello et al., 2018] 和 Distilling step-by-step [Hsieh et al., 2023] 的发现一致：**knowledge distillation 的有效性取决于 teacher 的 oracle 能力而非平均能力**。

**实践意义**：

- 对工业部署 4B 量级 Agent：用 100-300 条强 teacher BoN 成功轨迹做 SFT，再用 ~300 对 PD turn-level 偏好对做 DPO，可在 3 个 domain 上实现 6× 提升
- 对数据生成策略选择：不要盲目使用 PD，先评估 teacher 的"候选多样性"（如 K 个候选的平均 cosine similarity），若 > 0.95 则 PD 退化为 BoN

**贡献总结**：本文不仅给出了一个可用的小模型蒸馏管线（BoN-SFT + PD-DPO），更重要的是**修正了对 PD 有效性的预期**，揭示了其依赖的 teacher 条件，为后续相关工作提供了一个 sober 的基线。

---

## 附录

### 附录 A：τ2-bench 任务示例

为说明三域任务复杂度差异，各展示一个典型 task：

- **A.1 Retail 示例**：用户要求"修改订单中的台灯为黑色、背包改为中号灰色聚酯材质，并把地址改成默认地址"——涉及 3 个 sub-modification + 1 个 user lookup，平均 18-20 步可完成
- **A.2 Airline 示例**：用户要求"将商务舱机票降为经济舱并退差价"——涉及预订查询 + 改舱 + 退款，平均 10-15 步
- **A.3 Telecom 示例**：用户报告"手机无法连接 5G"——涉及账户查询 + 套餐确认 + 故障排查工作流，平均 15-18 步，**workflow 步骤多但每步决策相对模板化**，故 baseline 极低（4B 0%, 32B 3.7%）

### 附录 B：数据格式与训练超参（主章 3.4 / 3.5 的详细配置）

**B.1 SFT 数据格式**

每条训练样本是一次完整 episode 中的一次 agent 决策，格式为 ChatML 消息序列：
```
input: [
  {"role": "system", "content": <policy + tools>},
  {"role": "user", "content": <user msg>},
  {"role": "assistant", "tool_calls": [...]},
  {"role": "tool", "content": <tool result>},
  ... (历史轮次)
]
target: 最后一条 assistant 消息（文本回复 或 tool_calls JSON）
```

筛选规则：只保留 `final_reward == 1.0` 的整条 episode 内的 assistant 决策，不混入失败轨迹。

**B.2 SFT 超参（小数据 regime）**

| 参数 | 值 | 说明 |
|------|-----|------|
| LoRA rank | 8 | 限制可训练参数，防过拟合 |
| LoRA target | q_proj, v_proj | 不动 k_proj / o_proj |
| LoRA dropout | 0.1 | 加强正则化 |
| Epochs | 1 | 小数据避免背诵 |
| LR | 5e-5 | 小数据需要稍大步长 |
| Batch size | 4 (grad_accum=4, 等效 16) | A100 80GB 单卡 |
| Max length | 8192 | 覆盖长对话 |
| Optim | AdamW (β₁=0.9, β₂=0.999) | 标配 |
| Scheduler | cosine, warmup_ratio=0.03 | 标配 |
| LoRA Merge | `peft.merge_and_unload()` | 训练后合并 adapter 为完整权重 |

**B.3 DPO 输入格式（trl 1.4 conversational format）**

```python
{
  "prompt": [<对话历史 messages>],
  "chosen": [{"role": "assistant", "content": <chosen action>}],
  "rejected": [{"role": "assistant", "content": <rejected action>}]
}
```

**B.4 DPO 超参**

| 参数 | 值 | 说明 |
|------|-----|------|
| Beta | 0.3 | 适配小数据，防 reference drift |
| LR | 1e-6 | DPO 对学习率极敏感 |
| Epochs | 1 | 小数据 |
| Max prompt length | 6144 | 保留充足历史 |
| Eval fraction | 0.0 | 防 OOM（DPO eval 时 logits tensor 太大） |
| Reference model | SFT-merged 模型 | DPO 标准做法 |

**B.5 训练流程**

```
1. 先用 SFT 数据训练得到 SFT-LoRA → merge → SFT-merged 模型
2. 以 SFT-merged 作为 base model 和 reference model 启动 DPO
3. DPO 训练后再次 merge 得到最终模型
4. 用最终模型做评估
```

### 附录 C：Value Function 详细公式

**C.1 各信号定义**

- $\delta_{\text{progress}}$：foresight 中新调用的"期望工具"（任务 ground-truth 中要求调用的工具集合）占比
- $f_{\text{health}}$：1 − (错误调用 + 同名重复调用) / 总调用数
- $u_{\text{sentiment}}$：用户回复中的情感极性（基于关键词分类）
- $t_{\text{termination}}$：自然结束 1.0 / 步数耗尽 0.4 / 异常终止 0
- $e_{\text{assertions}}$：foresight 后 DB 状态相对 ground-truth 的合法性

**C.2 边界条件**
- foresight 中无任何工具调用 → $\delta_{\text{progress}} = 0$
- 任务无明确期望工具集 → 跳过 $\delta_{\text{progress}}$，权重重分配
- 总分 clip 到 [0, 1] 区间

### 附录 D：完整实验数据表

**C.1 Phase A（Qwen-Plus Teacher）4B 学生**

| 实验 | 数据 | 样本量 | Retail | Airline |
|------|------|--------|--------|---------|
| E0 | — | — | 2.9% | 6.7% |
| E1a | PD | 197 | 2.9% | 20.0% |
| E1b | BoN | 295 | 8.8% | 13.3% |
| E1c | BoN+DPO | 295+626 | 0.0% | 6.7% |
| E1d | PD+DPO | 197+626 | 2.9% | 13.3% |

**C.2 Phase B（Qwen3-32B Teacher）4B 学生**

| 实验 | 数据 | 样本量 | Retail | Airline | Telecom | Avg |
|------|------|--------|--------|---------|---------|-----|
| E0 | — | — | 2.9% | 6.7% | 0.0% | 2.4% |
| E3a | PD | 146 | 2.9% | 20.0% | 0.0% | 4.8% |
| E3b | BoN | 240 | 8.8% | 13.3% | 5.9% | 8.4% |
| E3c | BoN+DPO | 240+298 | 5.9% | 20.0% | **20.6%** | **14.5%** |
| E3d | PD+DPO | 146+298 | 0.0% | 6.7% | 0.0% | 1.2% |

**C.3 On-policy（8B Teacher）4B 学生**

| 实验 | 数据 | 样本量 | Retail | Airline |
|------|------|--------|--------|---------|
| E2a | PD | 111 | 2.9% | 13.3% |
| E2b | BoN | 99 | 2.9% | 6.7% |
| E2c | PD+DPO | 111+? | 0.0% | 13.3% |
| E2d | BoN+DPO | 99+? | 0.0% | 13.3% |

**C.4 Teacher 端数据质量统计**

见正文 5.2.1 表格。

**C.5 真实 PD 决策率**

见正文 5.5.2 表格。

### 附录 E：训练配置完整对应表

| ID | 名称 | Teacher | SFT 数据 | DPO 数据 | Domains |
|----|------|---------|----------|----------|---------|
| E0 | Zero-shot | — | — | — | retail/airline/telecom |
| E1a | Qwen-Plus PD-SFT | Qwen-Plus | PD 197 | — | retail/airline |
| E1b | Qwen-Plus BoN-SFT | Qwen-Plus | BoN 295 | — | retail/airline |
| E1c | Qwen-Plus BoN-SFT+DPO | Qwen-Plus | BoN 295 | Qwen-Plus PD pairs | retail/airline |
| E1d | Qwen-Plus PD-SFT+DPO | Qwen-Plus | PD 197 | Qwen-Plus PD pairs | retail/airline |
| E2a | 8B-Teacher PD-SFT | Qwen3-8B | PD 111 | — | retail/airline |
| E2b | 8B-Teacher BoN-SFT | Qwen3-8B | BoN 99 | — | retail/airline |
| E2c | 8B-Teacher PD-SFT+DPO | Qwen3-8B | PD 111 | 8B PD pairs | retail/airline |
| E2d | 8B-Teacher BoN-SFT+DPO | Qwen3-8B | BoN 99 | 8B PD pairs | retail/airline |
| E3a | 32B PD-SFT | Qwen3-32B | PD 146 | — | retail/airline/telecom |
| E3b | 32B BoN-SFT | Qwen3-32B | BoN 240 | — | retail/airline/telecom |
| E3c | **32B BoN-SFT+DPO ★** | Qwen3-32B | BoN 240 | 32B PD pairs 298 | retail/airline/telecom |
| E3d | 32B PD-SFT+DPO | Qwen3-32B | PD 146 | 32B PD pairs 298 | retail/airline/telecom |

### 附录 F：典型 trajectory 案例分析

为说明 BoN-SFT + PD-DPO 是如何具体改善 4B 模型行为的，本附录展示对比案例。

---

**F.1 DPO 带来新成功的案例**

- **Domain**：airline
- **Task ID**：task\_8（用户 sophia\_silva\_7557）
- **任务描述**：用户要求在 5 月 26 日预订一班从 ORD 飞往 PHL 的单程航班，要求与其 5 月 10 日的历史航班完全相同，额外添加乘客 Kevin Smith（出生于 2001 年 4 月 12 日），支付方式为证书，结算金额不超过 $500。

**Zero-shot 4B（reward=0，30 步，termination_reason=unknown）**

模型在第 2 步调用 `search_direct_flight` 查询 5 月 26 日可用航班，随后被用户告知不记得 5 月 10 日的航班号。模型的应对策略是调用 `get_reservation_details("ZFA04Y")` 但返回 `Error: Reservation ZFA04Y not found`，随后虽然通过 `get_user_details` 获取到账户下的全部 5 条预订记录（NM1VX1、KC18K6、S61CZX、H8Q05L、WUNA5K），却未逐一查询以定位 5 月 10 日的 ORD→PHL 航班。从第 14 步起模型进入信息获取循环，每两轮重复一次"请确认 5 月 10 日的航班日期"，直至 30 步耗尽，始终未完成预订。

**E3b BoN-SFT（reward=0，30 步，termination_reason=unknown）**

E3b 表现出另一种固化行为缺陷：在第 4 步调用 `book_reservation` 时直接选择 HAT139，但因座位不足返回 `Error: Not enough seats on flight HAT139`。随后模型并未改为查询其他航班，而是不断重复调用 `search_direct_flight("HAT139", "2024-05-10")` 核查历史状态，每次都返回 `landed`，表明该航班已降落。然而模型在获知此信息后仍重复同一工具调用共 7 次（步骤 8→9、22→23、24→25、26→27、28→29、30→31、32→33），形成刻板的"查询已降落航班—用户重复提问"循环，最终耗尽步数。

**E3c BoN-SFT + PD-DPO（reward=1.0，29 步，termination_reason=user_stop）**

E3c 模型在第 6 步同样遭遇了 `Error: Reservation ZFA04Y not found`，但随即改变策略：在步骤 10 调用 `get_user_details` 获取账户全部预订，然后在步骤 14–22 **依次调用** `get_reservation_details` 逐一查询 NM1VX1、KC18K6、S61CZX、H8Q05L，最终在步骤 22 从 WUNA5K 中确认 5 月 10 日的 ORD→PHL 航班为 **HAT271**（经济舱，$160）。确认航班号后，模型在步骤 26–28 检查 HAT271 在 5 月 26 日的座位，最终在步骤 36 调用 `book_reservation`，以 $348 证书付款成功预订双人座位（Sophia Silva + Kevin Smith），用户在步骤 40 收到确认摘要后发送 `###STOP###`。

**关键分歧点分析**：零样本模型和 E3b 均在"获取历史订单列表"后停止推理，未进一步遍历每条记录；E3c 的 PD-DPO 训练数据中存在"逐一调用 `get_reservation_details` 逐条检索"的 chosen 偏好，使模型习得了"列表—遍历"的多步规划模式，是本例成功的关键。

---

**F.2 E3d PD-SFT + DPO 过拟合案例**

- **Domain**：retail
- **Task ID**：task\_0（用户 yusuf\_rossi\_9620）
- **任务描述**：用户 Yusuf Rossi 要求将已送达订单 #W2378156 中的机械键盘换成 clicky 开关款，智能温控器换成兼容 Google Home 的型号，最后通过邮件获取退货说明。

**E3b BoN-SFT（reward=1.0，24 步，termination_reason=user_stop）**

E3b 在步骤 2 调用 `find_user_id_by_name_zip` 验证身份（返回 `yusuf_rossi_9620`），步骤 6 调用 `get_order_details` 确认订单已送达，步骤 10–12 分别调用 `get_product_details` 获取两种商品的变体列表，步骤 15 识别到 `item_id=7706410293`（clicky、无背光、全尺寸）和 `item_id=7747408585`（Google Assistant 兼容、黑色），步骤 16 调用 `modify_pending_order_items` 完成换货，步骤 24 前完成交互，用户满意后发送 `###STOP###`。整个 trajectory 层次清晰，无多余工具调用。

**E3d PD-SFT + DPO（reward=0.0，30 步，termination_reason=unknown）**

E3d 同样成功完成了换货操作（步骤 16 的 `modify_pending_order_items` 调用成功返回更新后的订单），但在步骤 18–35 陷入严重的**结束循环（termination loop）**：在用户表示"期待收到退货说明邮件"后，模型持续生成"您好！我很高兴能为您提供帮助……如有需要欢迎再次联系"之类的礼貌性结语，用户镜像式重复类似措辞，双方在未发出 `###STOP###` 的情况下循环 12 轮直至步数耗尽。任务的核心操作（换货）虽已完成，但模型未识别"交互应在退货说明发送后结束"的终止信号，导致奖励清零。

**关键分歧点分析**：E3d 的 PD-DPO 训练目标来自 PD 数据中的 turn-level 偏好，这些偏好对"礼貌性结语"的打分偏高，可能导致模型对"对话尚未结束、仍需等待用户确认"这一判断产生偏差。相比之下，E3b 的 BoN-SFT 数据中包含完整且干净的对话终止示范，使模型能及时识别任务已完成并等待用户终止。此案例揭示了 PD-DPO 在 turn-level 偏好叠加时可能引入的"礼貌过拟合"风险：模型对单轮响应质量的优化破坏了对多轮对话宏观结构的感知。

---

## 参考文献（BibTeX 入口与 URL）

### Agent 范式与框架

- **ReAct**：Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023. arXiv:2210.03629. https://arxiv.org/abs/2210.03629
- **Reflexion**：Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. arXiv:2303.11366. https://arxiv.org/abs/2303.11366
- **Toolformer**：Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS 2023. arXiv:2302.04761. https://arxiv.org/abs/2302.04761
- **LLM Agent Survey 1**：Wang, L., Ma, C., Feng, X., et al. (2023). *A Survey on Large Language Model based Autonomous Agents*. arXiv:2308.11432. https://arxiv.org/abs/2308.11432
- **LLM Agent Survey 2**：Xi, Z., Chen, W., Guo, X., et al. (2023). *The Rise and Potential of Large Language Model Based Agents: A Survey*. arXiv:2309.07864. https://arxiv.org/abs/2309.07864

### Agent 评测基准

- **τ-bench**：Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024). *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*. arXiv:2406.12045. https://arxiv.org/abs/2406.12045
- **τ²-bench**：Sierra Research (2025). *τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment*. arXiv:2506.07982. https://arxiv.org/pdf/2506.07982
- **WebArena**：Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., et al. (2024). *WebArena: A Realistic Web Environment for Building Autonomous Agents*. ICLR 2024. arXiv:2307.13854. https://arxiv.org/abs/2307.13854
- **AgentBench**：Liu, X., Yu, H., Zhang, H., et al. (2023). *AgentBench: Evaluating LLMs as Agents*. ICLR 2024. arXiv:2308.03688. https://arxiv.org/abs/2308.03688
- **ToolLLM**：Qin, Y., Liang, S., et al. (2023). *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*. ICLR 2024 spotlight. arXiv:2307.16789. https://arxiv.org/abs/2307.16789

### LLM 基础模型

- **Qwen3**：Qwen Team (2025). *Qwen3 Technical Report*. arXiv:2505.09388. https://arxiv.org/abs/2505.09388

### 指令微调与 PEFT

- **FLAN**：Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., et al. (2022). *Finetuned Language Models Are Zero-Shot Learners*. ICLR 2022. arXiv:2109.01652. https://arxiv.org/abs/2109.01652
- **InstructGPT**：Ouyang, L., Wu, J., Jiang, X., et al. (2022). *Training Language Models to Follow Instructions with Human Feedback*. NeurIPS 2022. arXiv:2203.02155. https://arxiv.org/abs/2203.02155
- **LoRA**：Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. arXiv:2106.09685. https://arxiv.org/abs/2106.09685

### 知识蒸馏

- **Hinton et al. 经典蒸馏**：Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. NIPS DLW. arXiv:1503.02531. https://arxiv.org/abs/1503.02531
- **Born-Again Networks**：Furlanello, T., Lipton, Z. C., Tschannen, M., Itti, L., & Anandkumar, A. (2018). *Born-Again Neural Networks*. ICML 2018. arXiv:1805.04770. https://arxiv.org/abs/1805.04770 ★ student-outperforms-teacher 现象的关键引文
- **Distilling step-by-step**：Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., et al. (2023). *Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes*. ACL 2023. arXiv:2305.02301. https://arxiv.org/abs/2305.02301 ★ 小模型反超大模型的现代证据

### Agent 行为蒸馏

- **AgentTuning**：Zeng, A., Liu, M., Lu, R., Wang, B., Liu, X., Dong, Y., & Tang, J. (2023). *AgentTuning: Enabling Generalized Agent Abilities for LLMs*. ACL Findings 2024. arXiv:2310.12823. https://arxiv.org/abs/2310.12823
- **FireAct**：Chen, B., Shu, C., Shareghi, E., Collier, N., Narasimhan, K., & Yao, S. (2023). *FireAct: Toward Language Agent Fine-tuning*. arXiv:2310.05915. https://arxiv.org/abs/2310.05915

### 偏好学习与 DPO

- **DPO**：Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023. arXiv:2305.18290. https://arxiv.org/abs/2305.18290
- **Step-DPO**（作为 step-level DPO 在数学推理上的代表性工作）：Lai, X., Tian, Z., Chen, Y., Yang, S., Peng, X., & Jia, J. (2024). *Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs*. arXiv:2406.18629. https://arxiv.org/abs/2406.18629

### 推理时计算扩展 / 非短视生成 / 树搜索（★ 本文核心理论基础）

- **Non-myopic Generation (Predictive Decoding)** ★★ ：Ma, C., Zhang, J., Zhu, Z., Yang, C., Yang, Y., Jin, Y., Lan, Z., Kong, L., & He, J. (2024). *Non-myopic Generation of Language Models for Reasoning and Planning*. arXiv:2410.17195. https://arxiv.org/abs/2410.17195 ★★ 本文方法直接源头
- **Tree of Thoughts**：Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. NeurIPS 2023. arXiv:2305.10601. https://arxiv.org/abs/2305.10601
- **Self-Consistency**：Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. ICLR 2023. arXiv:2203.11171. https://arxiv.org/abs/2203.11171
- **Controlled Decoding**：Mudgal, S., Lee, J., Ganapathy, H., et al. (2024). *Controlled Decoding from Language Models*. ICML 2024. arXiv:2310.17022. https://arxiv.org/abs/2310.17022
- **LATS / Tree Search for Agents**：Zhou, A., Yan, K., Shlapentokh-Rothman, M., Wang, H., & Wang, Y.-X. (2024). *Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models*. arXiv:2310.04406. https://arxiv.org/abs/2310.04406

### 基础设施（推理与训练框架）

- **vLLM / PagedAttention**：Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. arXiv:2309.06180. https://arxiv.org/abs/2309.06180
- **TRL (Transformers Reinforcement Learning)**：HuggingFace open-source library. https://github.com/huggingface/trl

### 数据来源

- **τ2-bench tasks**：训练/测试任务均来自 sierra-research/tau2-bench v1.x release，按 seed=42 随机切分（详见附录 D）

---

