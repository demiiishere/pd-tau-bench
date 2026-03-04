"""
Turn-level Predictive Decoding for tau-bench.

At each agent turn, instead of greedily picking one response, we:
  1. Sample K candidate responses (temperature=0.8)
  2. For each candidate, fork the environment state and simulate H more turns
  3. Score each fork using a value function
  4. Choose the candidate with the highest score
  5. Continue the real simulation with that choice
"""

import time
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Optional

from loguru import logger
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.tasks import Task
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.utils.llm_utils import generate
from tau2.utils.utils import get_now

from src.predictive_decoding.tau_bench_adapter import (
    restore_orchestrator_state,
    save_orchestrator_state,
    set_agent_temperature,
)
from src.predictive_decoding.value_function import compute_value


def _sum_usage(messages) -> dict:
    """Sum prompt_tokens and completion_tokens from a list of messages."""
    p, c = 0, 0
    for msg in messages:
        u = getattr(msg, "usage", None)
        if u and isinstance(u, dict):
            p += u.get("prompt_tokens", 0) or 0
            c += u.get("completion_tokens", 0) or 0
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _add_usage(a: dict, b: dict) -> dict:
    """Add two usage dicts together."""
    return {
        "prompt_tokens": a["prompt_tokens"] + b["prompt_tokens"],
        "completion_tokens": a["completion_tokens"] + b["completion_tokens"],
        "total_tokens": a["total_tokens"] + b["total_tokens"],
    }


_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def run_pd_episode(
    orchestrator: Orchestrator,
    task: Task,
    K: int = 5,
    H: int = 2,
    candidate_temperature: float = 0.8,
    foresight_temperature: float = 0.0,
    max_steps: int = 30,
) -> dict:
    """
    Run a full episode using turn-level Predictive Decoding.

    Args:
        orchestrator: A freshly created (not yet initialized) Orchestrator.
        task: The task being solved.
        K: Number of candidate responses to sample per agent turn.
        H: Number of foresight turns to simulate per candidate.
        candidate_temperature: Temperature for sampling diverse candidates.
        foresight_temperature: Temperature for greedy foresight rollouts.
        max_steps: Max real simulation steps.

    Returns:
        A dict with: task_id, steps, final_reward, conversation, termination_reason,
        wall_time_s, api_calls_approx, pd_steps_count, pd_steps_skipped_count
    """
    episode_start = time.time()

    # Initialize the orchestrator (sends the first "Hi! How can I help you?" message)
    orchestrator.initialize()

    steps = []  # PD decision records for each agent turn
    pd_steps_count = 0
    pd_steps_skipped_count = 0   # Steps skipped: all candidates near-identical text
    pd_steps_greedy_fb_count = 0  # Steps using greedy fallback: gap < threshold

    step_count = 0
    while not orchestrator.done and step_count < max_steps:
        # Check if it's the agent's turn to generate a response
        if (
            orchestrator.to_role == Role.AGENT
            and orchestrator.from_role in (Role.USER, Role.ENV)
        ):
            # ── Predictive Decoding decision point ───────────────────────────
            current_message = orchestrator.message
            step_record = _pd_step(
                orchestrator=orchestrator,
                task=task,
                incoming_message=current_message,
                K=K,
                H=H,
                candidate_temperature=candidate_temperature,
                foresight_temperature=foresight_temperature,
            )
            steps.append(step_record)
            pd_steps_count += 1
            if step_record.get("skipped_identical"):
                pd_steps_skipped_count += 1
            if step_record.get("greedy_fallback"):
                pd_steps_greedy_fb_count += 1
            # After _pd_step, the orchestrator has advanced past the agent turn
        else:
            # Non-agent turn: just step normally (user response, tool call execution)
            orchestrator.step()

        # Check step limits
        if orchestrator.to_role != Role.ENV:
            step_count += 1
            if step_count >= max_steps and not orchestrator.done:
                orchestrator.done = True
                orchestrator.termination_reason = TerminationReason.MAX_STEPS

    # Compute final reward using EnvironmentEvaluator
    final_reward = _compute_final_reward(orchestrator, task)

    # Build conversation in simple format for training
    conversation = _extract_conversation(orchestrator)

    wall_time_s = time.time() - episode_start

    # Token usage breakdown:
    #   episode_tokens  = tokens in the actual final trajectory (what becomes training data)
    #   overhead_tokens = tokens spent on PD exploration (candidate gen + foresight),
    #                     i.e. the cost of lookahead that doesn't appear in the trajectory
    #   total_tokens    = episode + overhead
    episode_usage = _sum_usage(orchestrator.get_trajectory())
    overhead_usage = dict(_ZERO_USAGE)
    for s in steps:
        u = s.get("token_usage_overhead", _ZERO_USAGE)
        overhead_usage = _add_usage(overhead_usage, u)
    total_usage = _add_usage(episode_usage, overhead_usage)

    return {
        "task_id": task.id,
        "steps": steps,
        "final_reward": final_reward,
        "conversation": conversation,
        "termination_reason": (
            orchestrator.termination_reason.value
            if orchestrator.termination_reason
            else "unknown"
        ),
        "wall_time_s": round(wall_time_s, 2),
        "tokens": {
            "episode": episode_usage,
            "overhead": overhead_usage,
            "total": total_usage,
        },
        "pd_steps_count": pd_steps_count,
        "pd_steps_skipped_count": pd_steps_skipped_count,
        "pd_steps_greedy_fb_count": pd_steps_greedy_fb_count,
        "source": "pd",
    }


def _pd_step(
    orchestrator: Orchestrator,
    task: Task,
    incoming_message,
    K: int,
    H: int,
    candidate_temperature: float,
    foresight_temperature: float,
) -> dict:
    """
    Perform one PD decision step at an agent turn.

    1. Save current state.
    2. Generate K candidate agent responses (with adaptive temperature).
    3. If all candidates are near-identical, skip PD and use the first one.
    4. For each candidate, fork → inject → simulate H turns → score.
    5. Pick best candidate.
    6. Restore original state, inject best candidate into the real orchestrator.

    Returns a record of this step for DPO data construction.
    """
    # Save state BEFORE the agent responds
    saved_state = save_orchestrator_state(orchestrator)
    conversation_snapshot = deepcopy(orchestrator.trajectory)

    # ── Step 1: Generate K candidate responses (with adaptive temperature) ────
    candidates, cand_usage = _generate_candidates(
        orchestrator=orchestrator,
        incoming_message=incoming_message,
        K=K,
        temperature=candidate_temperature,
    )

    # ── Optimization: Skip PD if all candidates are near-identical ────────────
    if _candidates_are_identical(candidates, threshold=0.95):
        restore_orchestrator_state(orchestrator, saved_state)
        _inject_agent_response(orchestrator, incoming_message, candidates[0])
        logger.debug("PD step SKIPPED: all candidates near-identical, using greedy.")
        return {
            "conversation_history": [_msg_to_dict(m) for m in conversation_snapshot],
            "candidates": [_msg_to_dict(candidates[0])],
            "scores": [None],
            "chosen_idx": 0,
            "chosen_action": _msg_to_dict(candidates[0]),
            "skipped_identical": True,
            "greedy_fallback": False,
            # overhead = candidate generation only (no foresight for skipped steps)
            "token_usage_overhead": cand_usage,
        }

    # ── Step 2 & 3: Foresight rollout + scoring for each candidate ────────────
    candidate_scores = []
    foresight_usage = dict(_ZERO_USAGE)
    for i, candidate in enumerate(candidates):
        restore_orchestrator_state(orchestrator, saved_state)
        score, fs_usage = _evaluate_candidate(
            orchestrator=orchestrator,
            task=task,
            incoming_message=incoming_message,
            candidate=candidate,
            H=H,
            foresight_temperature=foresight_temperature,
        )
        candidate_scores.append(score)
        foresight_usage = _add_usage(foresight_usage, fs_usage)
        logger.debug(f"  Candidate {i}: score={score:.3f} | {_preview(candidate)}")

    # ── Step 4: Pick best candidate ───────────────────────────────────────────
    best_idx = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
    score_gap = max(candidate_scores) - min(candidate_scores)

    # Total overhead so far: all K candidates + all foresight
    # (the chosen candidate's tokens will also appear in episode_tokens via trajectory,
    #  so we subtract them from overhead to avoid double-counting)
    chosen_cand_usage = _sum_usage([candidates[best_idx]])
    overhead_this_step = _add_usage(cand_usage, foresight_usage)
    overhead_this_step = {
        "prompt_tokens": overhead_this_step["prompt_tokens"] - chosen_cand_usage["prompt_tokens"],
        "completion_tokens": overhead_this_step["completion_tokens"] - chosen_cand_usage["completion_tokens"],
        "total_tokens": overhead_this_step["total_tokens"] - chosen_cand_usage["total_tokens"],
    }

    # ── Greedy fallback when gap is negligible ────────────────────────────────
    GREEDY_FALLBACK_THRESHOLD = 0.05
    if score_gap < GREEDY_FALLBACK_THRESHOLD:
        restore_orchestrator_state(orchestrator, saved_state)
        greedy_list, greedy_usage = _generate_candidates(
            orchestrator=orchestrator,
            incoming_message=incoming_message,
            K=1,
            temperature=0.0,
        )
        greedy = greedy_list[0]
        _inject_agent_response(orchestrator, incoming_message, greedy)
        logger.debug(
            f"PD step GREEDY FALLBACK: gap={score_gap:.3f} < {GREEDY_FALLBACK_THRESHOLD}"
        )
        # Overhead = all K exploration candidates + foresight + greedy gen
        # (greedy will appear in trajectory so subtract it from overhead)
        fb_overhead = _add_usage(
            _add_usage(cand_usage, foresight_usage), greedy_usage
        )
        greedy_cand_usage = _sum_usage([greedy])
        fb_overhead = {
            "prompt_tokens": fb_overhead["prompt_tokens"] - greedy_cand_usage["prompt_tokens"],
            "completion_tokens": fb_overhead["completion_tokens"] - greedy_cand_usage["completion_tokens"],
            "total_tokens": fb_overhead["total_tokens"] - greedy_cand_usage["total_tokens"],
        }
        return {
            "conversation_history": [_msg_to_dict(m) for m in conversation_snapshot],
            "candidates": [_msg_to_dict(c) for c in candidates],
            "scores": candidate_scores,
            "chosen_idx": -1,
            "chosen_action": _msg_to_dict(greedy),
            "skipped_identical": False,
            "greedy_fallback": True,
            "score_gap": score_gap,
            "token_usage_overhead": fb_overhead,
        }

    chosen = candidates[best_idx]
    logger.info(
        f"PD step: chose candidate {best_idx} (score={candidate_scores[best_idx]:.3f}, "
        f"gap={score_gap:.3f})"
    )

    # ── Step 5: Restore state and inject the chosen candidate ─────────────────
    restore_orchestrator_state(orchestrator, saved_state)
    _inject_agent_response(orchestrator, incoming_message, chosen)

    return {
        "conversation_history": [_msg_to_dict(m) for m in conversation_snapshot],
        "candidates": [_msg_to_dict(c) for c in candidates],
        "scores": candidate_scores,
        "chosen_idx": best_idx,
        "chosen_action": _msg_to_dict(chosen),
        "skipped_identical": False,
        "greedy_fallback": False,
        "score_gap": score_gap,
        "token_usage_overhead": overhead_this_step,
    }


def _generate_candidates(
    orchestrator: Orchestrator,
    incoming_message,
    K: int,
    temperature: float,
) -> tuple[list[AssistantMessage], dict]:
    """
    Generate K diverse candidate responses from the agent LLM.
    Uses adaptive temperature: if early candidates are too similar, raises temp.

    Returns:
        (candidates, usage) where usage = sum of token usage across all K calls.
    """
    agent: LLMAgent = orchestrator.agent
    agent_state = orchestrator.agent_state

    if isinstance(incoming_message, MultiToolMessage):
        history = list(agent_state.messages) + list(incoming_message.tool_messages)
    else:
        history = list(agent_state.messages) + [incoming_message]

    messages = agent_state.system_messages + history

    candidates = []
    current_temp = temperature
    for i in range(K):
        if i >= 2:
            current_temp = _adaptive_temperature(candidates, base_temp=temperature)

        try:
            candidate = generate(
                model=agent.llm,
                messages=messages,
                tools=agent.tools,
                temperature=current_temp,
            )
            candidates.append(candidate)
        except Exception as e:
            logger.warning(f"Failed to generate candidate {i} (temp={current_temp:.2f}): {e}")
            try:
                candidate = generate(
                    model=agent.llm,
                    messages=messages,
                    tools=agent.tools,
                    temperature=temperature,
                )
                candidates.append(candidate)
            except Exception as e2:
                logger.error(f"Fallback generation also failed: {e2}")

    if not candidates:
        raise RuntimeError("No candidates generated. Check API connectivity.")

    usage = _sum_usage(candidates)
    return candidates, usage


def _adaptive_temperature(
    candidates_so_far: list,
    base_temp: float = 0.8,
    max_temp: float = 1.2,
) -> float:
    """
    Raise temperature if existing candidates are too similar to each other.
    This encourages diversity when early samples are near-identical.
    """
    if len(candidates_so_far) < 2:
        return base_temp

    texts = []
    for c in candidates_so_far:
        if c.content:
            texts.append(c.content)
        elif c.tool_calls:
            texts.append(f"{c.tool_calls[0].name}({c.tool_calls[0].arguments})")
        else:
            texts.append("")

    similarities = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = SequenceMatcher(None, texts[i], texts[j]).ratio()
            similarities.append(sim)

    avg_sim = sum(similarities) / len(similarities)

    if avg_sim > 0.9:
        return min(base_temp + 0.3, max_temp)
    elif avg_sim > 0.8:
        return min(base_temp + 0.15, max_temp)
    return base_temp


def _candidates_are_identical(candidates: list, threshold: float = 0.95) -> bool:
    """
    Return True if more than half of candidate pairs are near-identical.
    Used to skip PD foresight when the step is highly deterministic.
    """
    if len(candidates) < 2:
        return False

    texts = []
    for c in candidates:
        if c.content:
            texts.append(c.content)
        elif c.tool_calls:
            texts.append(f"{c.tool_calls[0].name}({c.tool_calls[0].arguments})")
        else:
            texts.append("")

    total_pairs = 0
    similar_pairs = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = SequenceMatcher(None, texts[i], texts[j]).ratio()
            total_pairs += 1
            if sim > threshold:
                similar_pairs += 1

    # Skip if more than half of pairs are near-identical
    return similar_pairs > total_pairs / 2


def _evaluate_candidate(
    orchestrator: Orchestrator,
    task: Task,
    incoming_message,
    candidate: AssistantMessage,
    H: int,
    foresight_temperature: float,
) -> tuple[float, dict]:
    """
    Evaluate a candidate by injecting it and running H foresight turns.
    Returns (value_score, foresight_token_usage).

    foresight_token_usage includes the injected candidate itself plus all
    messages generated during the H foresight steps.
    """
    original_temp = orchestrator.agent.llm_args.get("temperature", 0.0)
    set_agent_temperature(orchestrator, foresight_temperature)

    traj_len_before = len(orchestrator.trajectory)
    try:
        _inject_agent_response(orchestrator, incoming_message, candidate)

        for _ in range(H):
            if orchestrator.done:
                break
            try:
                orchestrator.step()
            except Exception as e:
                logger.warning(f"Foresight step failed: {e}")
                break

        score = compute_value(orchestrator, task, foresight_start_idx=traj_len_before)
        # Sum usage from all new messages added during this foresight evaluation
        new_messages = orchestrator.trajectory[traj_len_before:]
        usage = _sum_usage(new_messages)
    finally:
        set_agent_temperature(orchestrator, original_temp)

    return score, usage


def _inject_agent_response(
    orchestrator: Orchestrator,
    incoming_message,
    candidate: AssistantMessage,
) -> None:
    """
    Inject a pre-generated agent response into the orchestrator,
    advancing its state as if the agent had just generated that response.
    """
    agent_state = orchestrator.agent_state

    # Update agent_state.messages (same logic as LLMAgent.generate_next_message)
    if isinstance(incoming_message, MultiToolMessage):
        agent_state.messages.extend(incoming_message.tool_messages)
    else:
        agent_state.messages.append(incoming_message)

    candidate.timestamp = get_now()
    agent_state.messages.append(candidate)
    orchestrator.trajectory.append(candidate)
    orchestrator.message = candidate
    orchestrator.from_role = Role.AGENT

    if orchestrator.agent.is_stop(candidate):
        orchestrator.done = True
        orchestrator.termination_reason = TerminationReason.AGENT_STOP
        orchestrator.to_role = Role.USER
    elif candidate.is_tool_call():
        orchestrator.to_role = Role.ENV
    else:
        orchestrator.to_role = Role.USER

    orchestrator.step_count += 1
    orchestrator.environment.sync_tools()


def _compute_final_reward(orchestrator: Orchestrator, task: Task) -> float:
    """
    Compute the final binary reward using EnvironmentEvaluator.
    """
    from tau2.data_model.simulation import SimulationRun, TerminationReason as TR
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    if orchestrator.termination_reason not in {TR.AGENT_STOP, TR.USER_STOP}:
        return 0.0

    messages = orchestrator.get_trajectory()
    sim_run = SimulationRun(
        id="pd_eval",
        task_id=task.id,
        start_time=get_now(),
        end_time=get_now(),
        duration=0.0,
        termination_reason=orchestrator.termination_reason.value,
        messages=messages,
    )
    try:
        reward_info = evaluate_simulation(
            simulation=sim_run,
            task=task,
            evaluation_type=EvaluationType.ENV,
            solo_mode=False,
            domain=orchestrator.domain,
        )
        return float(reward_info.reward)
    except Exception as e:
        logger.error(f"Final reward evaluation failed: {e}")
        return 0.0


def _extract_conversation(orchestrator: Orchestrator) -> list[dict]:
    """Extract the conversation history in a simple dict format."""
    return [_msg_to_dict(m) for m in orchestrator.get_trajectory()]


def _msg_to_dict(msg) -> dict:
    """Convert a tau2 message to a serializable dict."""
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage, UserMessage, SystemMessage

    if isinstance(msg, AssistantMessage):
        d = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ]
        return d
    elif isinstance(msg, UserMessage):
        d = {"role": "user", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ]
        return d
    elif isinstance(msg, ToolMessage):
        return {"role": "tool", "content": msg.content, "tool_call_id": msg.id}
    elif isinstance(msg, MultiToolMessage):
        return {
            "role": "tool",
            "content": [_msg_to_dict(m) for m in msg.tool_messages],
        }
    elif isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    else:
        return {"role": "unknown", "content": str(msg)}


def _preview(msg: AssistantMessage, max_len: int = 80) -> str:
    """Short preview of a message for logging."""
    if msg.content:
        return msg.content[:max_len]
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        return f"{tc.name}({tc.arguments})"[:max_len]
    return "(empty)"
