Value Function 区分度不够
你的 progress 里提到平均 score gap 只有 ~0.024，这确实偏低。问题出在当前 value function 的构成上。
当前问题诊断：你的 value function 有三个信号（数据库状态匹配、action overlap、对话正常结束）。对于那些没有 env_assertions 的任务，数据库状态匹配这个最强信号用不了，只能靠 action overlap，而 action overlap 在 foresight 只看 2 步的情况下，很多候选可能调用了相同的工具，区分度就很低。
具体优化方案：
方案 A：加入 LLM-as-Judge 信号（推荐，效果最好但多花一点 API）
在 foresight 结束后，把 foresight 的完整对话喂给 Qwen，让它从多个维度打分：
pythondef llm_judge_score(foresight_conversation, task_description, domain_rules):
    prompt = f"""你是一个客服质量评估员。以下是一段客服对话的前几轮。
    
任务描述：{task_description}
域规则摘要：{domain_rules}
对话内容：{foresight_conversation}

请从以下 4 个维度各打 1-5 分：
1. 信息收集完整性：agent 是否在前几步就收集了必要信息？
2. 规则遵守：agent 的行为是否符合域规则？
3. 工具使用效率：agent 是否调用了正确的工具，没有多余调用？
4. 对话推进度：对话是否在朝着解决用户问题的方向推进？

只输出 JSON：{{"info": X, "rules": X, "tools": X, "progress": X}}"""
    
    response = call_qwen(prompt)
    scores = parse_json(response)
    return sum(scores.values()) / 20.0  # 归一化到 0-1
然后把这个 LLM judge score 作为 value function 的第四个信号，权重设为 0.3，同时把其他三个信号的权重调低一些。
成本增加不大——每个候选每步多一次 API 调用，但用的是短 prompt + 短 response，大概让总成本增加 30-50%。
方案 B：增加 foresight 深度 H（简单但效果有限）
把 H 从 2 增加到 3。多模拟一轮意味着更多信息——比如 H=2 可能还没到用户确认步骤，但 H=3 就能看到用户确认后 agent 是否真的调了工具。但这会让成本线性增加。
方案 C：增加候选数 K（配合 temperature 调整）
K 从 5 增到 8，同时 temperature 从 0.8 调到 1.0。这样候选的多样性更大，score 的区分度自然会提高。成本也是线性增加。
我的建议是 A + C 组合：加 LLM judge + 把 K 提到 8、temperature 提到 1.0。这样既增加了打分的区分度，又增加了候选的多样性。
优化 2：任务筛选——哪些任务 PD 收益大
你提到了一个很重要的观察：顺序性强的任务（每步只有一个正确 tool call）PD 收益小。这很合理——如果一个任务的每一步都是确定性的（必须先查订单、再查产品、再执行换货），那不同候选之间的区别只在文本措辞上，工具调用序列是一样的，PD 没什么发挥空间。
具体做法：
先跑一遍所有任务的 baseline（standard decoding），统计每个任务的特征：
pythondef classify_task_pd_potential(task, baseline_trajectory):
    """判断一个任务是否适合 PD 优化"""
    features = {
        # 1. reward_basis 是否包含 COMMUNICATE
        #    COMMUNICATE 意味着评估关注 agent 的文本回复质量，PD 在这里更有空间
        "has_communicate": "COMMUNICATE" in task.reward_basis,
        
        # 2. 任务需要的 tool call 数量
        #    tool call 越多 → 决策点越多 → PD 发挥空间越大
        "num_tool_calls": count_tool_calls(baseline_trajectory),
        
        # 3. baseline 是否失败
        #    baseline 成功的任务 PD 改进空间有限（除非关注 pass^k 可靠性）
        "baseline_failed": baseline_trajectory["final_reward"] == 0,
        
        # 4. 对话轮数
        #    轮数越多 → 早期错误影响越大 → PD 收益越大
        "num_turns": count_turns(baseline_trajectory),
    }
    
    # 高潜力任务：baseline 失败 + 多轮对话 + 有 COMMUNICATE
    pd_potential = (
        features["baseline_failed"] 
        and features["num_turns"] >= 4
        and (features["has_communicate"] or features["num_tool_calls"] >= 3)
    )
    
    return pd_potential, features
然后在全量生成时，对高潜力任务用更大的 K 和 H（K=8, H=3），对低潜力任务用小参数（K=3, H=1）甚至跳过 PD 直接用 standard decoding。这样能把 API 预算集中在收益最大的任务上。
优化 3：Temperature 调参
你提到 temperature=0.8 时某些步骤仍然产出重复候选。这说明在一些"确定性高"的步骤（比如 agent 必须调用 find_user_id_by_name_zip），即使 temperature=0.8，模型也倾向于输出几乎相同的内容。
具体做法：自适应 temperature
pythondef adaptive_temperature(candidates_so_far, base_temp=0.8, max_temp=1.2):
    """如果已有候选太相似，自动提高 temperature"""
    if len(candidates_so_far) < 2:
        return base_temp
    
    # 计算已有候选之间的文本相似度
    from difflib import SequenceMatcher
    similarities = []
    for i in range(len(candidates_so_far)):
        for j in range(i+1, len(candidates_so_far)):
            sim = SequenceMatcher(
                None, 
                candidates_so_far[i]["content"], 
                candidates_so_far[j]["content"]
            ).ratio()
            similarities.append(sim)
    
    avg_sim = sum(similarities) / len(similarities)
    
    # 相似度 > 0.9 说明候选太重复，提高 temperature
    if avg_sim > 0.9:
        return min(base_temp + 0.3, max_temp)
    elif avg_sim > 0.8:
        return min(base_temp + 0.15, max_temp)
    return base_temp
在采样循环中使用：
pythoncandidates = []
for i in range(K):
    temp = adaptive_temperature(candidates)
    response = agent_llm.generate(messages=history, temperature=temp)
    candidates.append(response)
另外一个更简单的做法是：如果 K 个候选中有超过一半文本相似度 > 0.95，就认为这一步"确定性太高，PD 无法提供额外信息"，直接跳过 PD 用 greedy 选第一个。这能节省 API 成本。