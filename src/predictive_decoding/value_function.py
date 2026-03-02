"""
Value function for scoring foresight rollout states.

After running H turns of foresight, we need to estimate how "good" the
current environment state is. We use a combination of signals:
  1. Env assertions: partial goal satisfaction (most informative)
  2. Termination status: did the episode end cleanly?
  3. Error count: how many tool call errors occurred?
"""

from tau2.data_model.simulation import TerminationReason
from tau2.data_model.tasks import Task
from tau2.orchestrator.orchestrator import Orchestrator


def compute_value(orch: Orchestrator, task: Task) -> float:
    """
    Compute a value score for the current state of the orchestrator
    after a foresight rollout.

    Returns a float in [0, 1]. Higher is better.
    """
    score = 0.0

    # ── Signal 1: Env assertions (weight 0.5) ────────────────────────────────
    # Run each env_assertion on the current environment state.
    # These check partial goal completion (e.g., "is the order cancelled?").
    assertion_score = _compute_assertion_score(orch, task)
    score += 0.5 * assertion_score

    # ── Signal 2: Clean termination (weight 0.3) ─────────────────────────────
    termination_score = _compute_termination_score(orch)
    score += 0.3 * termination_score

    # ── Signal 3: Error penalty (weight 0.2) ─────────────────────────────────
    error_score = _compute_error_score(orch)
    score += 0.2 * error_score

    return score


def _compute_assertion_score(orch: Orchestrator, task: Task) -> float:
    """
    Check how many env_assertions pass against the current environment state.
    Returns fraction of passing assertions, or 0.5 if no assertions defined.
    """
    if task.evaluation_criteria is None:
        return 0.5  # No criteria, neutral score

    env_assertions = task.evaluation_criteria.env_assertions
    if not env_assertions:
        # Fall back to action overlap if available
        return _compute_action_overlap_score(orch, task)

    passed = 0
    total = len(env_assertions)
    for assertion in env_assertions:
        try:
            result = orch.environment.run_env_assertion(
                assertion, raise_assertion_error=False
            )
            if result:
                passed += 1
        except Exception:
            pass  # If assertion fails to run, skip it

    return passed / total if total > 0 else 0.5


def _compute_action_overlap_score(orch: Orchestrator, task: Task) -> float:
    """
    Compute overlap between expected actions and actual tool calls in trajectory.
    """
    if task.evaluation_criteria is None or not task.evaluation_criteria.actions:
        return 0.5

    expected_actions = task.evaluation_criteria.actions
    expected_names = {a.name for a in expected_actions if a.requestor == "assistant"}
    if not expected_names:
        return 0.5

    from tau2.data_model.message import AssistantMessage
    actual_names = set()
    for msg in orch.trajectory:
        if isinstance(msg, AssistantMessage) and msg.is_tool_call():
            for tc in msg.tool_calls:
                actual_names.add(tc.name)

    overlap = len(expected_names & actual_names)
    return overlap / len(expected_names)


def _compute_termination_score(orch: Orchestrator) -> float:
    """
    Score based on how the simulation terminated (or if it's still running).
      - AGENT_STOP (normal end): 1.0
      - USER_STOP: 0.7 (user ended, might be partial success)
      - Still running (not done): 0.4
      - Error terminations: 0.0
    """
    if not orch.done:
        return 0.4

    reason = orch.termination_reason
    if reason == TerminationReason.AGENT_STOP:
        return 1.0
    elif reason == TerminationReason.USER_STOP:
        return 0.7
    elif reason == TerminationReason.MAX_STEPS:
        return 0.2
    else:
        # AGENT_ERROR, USER_ERROR, TOO_MANY_ERRORS
        return 0.0


def _compute_error_score(orch: Orchestrator) -> float:
    """
    Penalize tool call errors. Returns 1.0 if no errors, decreasing with more errors.
    """
    if orch.num_errors == 0:
        return 1.0
    elif orch.num_errors <= 2:
        return 0.5
    elif orch.num_errors <= 5:
        return 0.2
    else:
        return 0.0
