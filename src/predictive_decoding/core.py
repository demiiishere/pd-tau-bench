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
    pd_steps_skipped_count = 0  # Steps where all candidates were identical → skipped PD

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

    # Estimate API calls: each PD step makes K candidate calls + K*H foresight calls
    # Skipped steps only made K calls (no foresight). Non-PD steps make ~2 calls (agent+user).
    non_pd_steps = step_count - pd_steps_count
    api_calls_approx = (
        sum(s.get("api_calls_approx", 0) for s in steps)
        + non_pd_steps * 2
    )

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
        "api_calls_approx": api_calls_approx,
        "pd_steps_count": pd_steps_count,
        "pd_steps_skipped_count": pd_steps_skipped_count,
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
    candidates = _generate_candidates(
        orchestrator=orchestrator,
        incoming_message=incoming_message,
        K=K,
        temperature=candidate_temperature,
    )

    # ── Optimization: Skip PD if all candidates are near-identical ────────────
    # (happens when the step is highly deterministic, e.g. a required tool call)
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
            "api_calls_approx": len(candidates),  # Only candidate generation, no foresight
        }

    # ── Step 2 & 3: Foresight rollout + scoring for each candidate ────────────
    candidate_scores = []
    total_foresight_steps = 0
    for i, candidate in enumerate(candidates):
        restore_orchestrator_state(orchestrator, saved_state)
        score, fs = _evaluate_candidate(
            orchestrator=orchestrator,
            task=task,
            incoming_message=incoming_message,
            candidate=candidate,
            H=H,
            foresight_temperature=foresight_temperature,
        )
        candidate_scores.append(score)
        total_foresight_steps += fs
        logger.debug(f"  Candidate {i}: score={score:.3f} | {_preview(candidate)}")

    # ── Step 4: Pick best candidate ───────────────────────────────────────────
    best_idx = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
    chosen = candidates[best_idx]
    logger.info(
        f"PD step: chose candidate {best_idx} (score={candidate_scores[best_idx]:.3f})"
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
        "api_calls_approx": len(candidates) + total_foresight_steps,
    }


def _generate_candidates(
    orchestrator: Orchestrator,
    incoming_message,
    K: int,
    temperature: float,
) -> list[AssistantMessage]:
    """
    Generate K diverse candidate responses from the agent LLM.
    Uses adaptive temperature: if early candidates are too similar, raises temp
    to encourage diversity.
    """
    agent: LLMAgent = orchestrator.agent
    agent_state = orchestrator.agent_state

    # Build the message list the agent would see
    if isinstance(incoming_message, MultiToolMessage):
        history = list(agent_state.messages) + list(incoming_message.tool_messages)
    else:
        history = list(agent_state.messages) + [incoming_message]

    messages = agent_state.system_messages + history

    candidates = []
    current_temp = temperature
    for i in range(K):
        # Adaptive temperature: if previous candidates are too similar, raise temp
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
            # Fall back to base temperature
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

    return candidates


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
) -> tuple[float, int]:
    """
    Evaluate a candidate by injecting it and running H foresight turns.
    Returns (value_score, foresight_steps_taken).
    """
    # Set greedy temperature for foresight
    original_temp = orchestrator.agent.llm_args.get("temperature", 0.0)
    set_agent_temperature(orchestrator, foresight_temperature)

    foresight_steps = 0
    try:
        # Inject the candidate as the agent's response
        _inject_agent_response(orchestrator, incoming_message, candidate)

        # Run H more turns (or until done), counting non-ENV steps
        for _ in range(H):
            if orchestrator.done:
                break
            try:
                orchestrator.step()
                if orchestrator.to_role != Role.ENV:
                    foresight_steps += 1
            except Exception as e:
                logger.warning(f"Foresight step failed: {e}")
                break

        # Score the resulting state
        score = compute_value(orchestrator, task)
    finally:
        # Always restore temperature
        set_agent_temperature(orchestrator, original_temp)

    return score, foresight_steps


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
