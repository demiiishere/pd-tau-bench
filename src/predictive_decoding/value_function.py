"""
Value function for scoring foresight rollout states.

After running H turns of foresight, we estimate how "good" the current state is.

Signals (all computed from trajectory data, zero extra API calls):
  1. Delta action progress  (0.35): how many NEW expected tools were called in foresight
  2. Foresight health       (0.25): tool errors and redundancy during foresight
  3. User sentiment         (0.15): positive/negative signals in user's foresight response
  4. Termination status     (0.15): did the episode end cleanly?
  5. Env assertions         (0.10): partial goal satisfaction (rarely fires in H=2)

Key design principle: signals 1-3 are computed on the *foresight delta* only
(trajectory[foresight_start_idx:]), not the full trajectory. This makes them
sensitive to differences between candidates even when the pre-foresight context
is identical for all candidates.
"""

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import TerminationReason
from tau2.data_model.tasks import Task
from tau2.orchestrator.orchestrator import Orchestrator

# Keywords for user sentiment scoring
_POSITIVE_WORDS = {
    "yes", "ok", "okay", "sure", "please", "proceed", "correct", "thank",
    "thanks", "great", "right", "good", "sounds", "alright", "perfect",
    "exactly", "confirmed", "confirm", "understood", "clear", "agree",
}
_NEGATIVE_WORDS = {
    "no", "wrong", "mistake", "confused", "confusion", "wait", "actually",
    "not", "incorrect", "misunderstood", "unclear", "don't", "doesnt",
    "error", "problem", "issue", "never", "stop", "cancel",
}


def compute_value(orch: Orchestrator, task: Task, foresight_start_idx: int = 0) -> float:
    """
    Compute a value score for the current state after a foresight rollout.

    Args:
        orch: orchestrator after H foresight steps
        task: the current task (for expected action sequence)
        foresight_start_idx: index in orch.trajectory where foresight began.
            Signals 1-3 are computed on trajectory[foresight_start_idx:] only.

    Returns a float in [0, 1]. Higher is better.
    """
    foresight_msgs = orch.trajectory[foresight_start_idx:]

    # ── Signal 1: Delta action progress (weight 0.35) ────────────────────────
    # Measures how many NEW expected tools were called during foresight.
    # Computed on delta only → sensitive even when pre-foresight context is identical.
    delta_score = _compute_delta_progress(orch, task, foresight_start_idx)

    # ── Signal 2: Foresight health (weight 0.25) ─────────────────────────────
    # Penalises tool call errors and redundant calls within the foresight window.
    health_score = _compute_foresight_health(foresight_msgs)

    # ── Signal 3: User sentiment in foresight (weight 0.15) ──────────────────
    # If the simulated user replied during foresight, score their response tone.
    sentiment_score = _compute_user_sentiment(foresight_msgs)

    # ── Signal 4: Termination (weight 0.15, down from 0.3) ───────────────────
    termination_score = _compute_termination_score(orch)

    # ── Signal 5: Env assertions (weight 0.10, down from 0.5) ────────────────
    # Kept for cases where assertions DO fire within H steps (e.g. H≥3).
    assertion_score = _compute_assertion_score(orch, task)

    return (
        0.35 * delta_score
        + 0.25 * health_score
        + 0.15 * sentiment_score
        + 0.15 * termination_score
        + 0.10 * assertion_score
    )


# ── Signal implementations ────────────────────────────────────────────────────

def _compute_delta_progress(
    orch: Orchestrator, task: Task, foresight_start_idx: int
) -> float:
    """
    Measures how many expected tools were newly called during foresight.

    Algorithm:
      1. Build ordered list of expected tool names from evaluation criteria.
      2. Find which expected tools were already called BEFORE foresight.
      3. Of the remaining expected tools, count how many appear in foresight.
      4. Score = 0.5 (base) + 0.5 * (matched_in_foresight / remaining_count).
         — 0.5 if nothing new was done (neutral, not penalised)
         — up to 1.0 if all remaining expected tools were called
         — 0.5 base avoids unfairly penalising text-response steps
    """
    if task.evaluation_criteria is None:
        return 0.5
    actions = task.evaluation_criteria.actions
    if not actions:
        return 0.5

    expected_names = [a.name for a in actions if a.requestor == "assistant"]
    if not expected_names:
        return 0.5

    # Tools called before foresight
    pre_called: list[str] = []
    for msg in orch.trajectory[:foresight_start_idx]:
        if isinstance(msg, AssistantMessage) and msg.is_tool_call():
            for tc in msg.tool_calls:
                pre_called.append(tc.name)

    # Compute remaining expected tools (remove already-completed ones)
    remaining = list(expected_names)
    for tool in pre_called:
        if tool in remaining:
            remaining.remove(tool)

    if not remaining:
        # All expected work already done before foresight → best possible
        return 1.0

    # Tools called DURING foresight
    new_called: list[str] = []
    for msg in orch.trajectory[foresight_start_idx:]:
        if isinstance(msg, AssistantMessage) and msg.is_tool_call():
            for tc in msg.tool_calls:
                new_called.append(tc.name)

    if not new_called:
        # Pure text response — no new tool calls, return neutral
        return 0.5

    # Count how many remaining expected tools appear in foresight (ordered match)
    matched = 0
    remaining_copy = list(remaining)
    for tool in new_called:
        if tool in remaining_copy:
            matched += 1
            remaining_copy.remove(tool)

    frac = matched / len(remaining)
    return 0.5 + 0.5 * frac


def _compute_foresight_health(foresight_msgs: list) -> float:
    """
    Score foresight tool call quality.

    Checks:
      - ToolMessage.error == True  → hard errors (wrong args, API failure)
      - Duplicate tool calls within foresight (same name called twice → inefficiency)

    Returns 1.0 for clean foresight, lower for errors/redundancy.
    """
    tool_results = [m for m in foresight_msgs if isinstance(m, ToolMessage)]

    if not tool_results:
        # No tool calls made during foresight (text-response step) → neutral
        return 0.7

    errors = sum(1 for m in tool_results if m.error)
    total = len(tool_results)

    # Check redundant calls (same tool called more than once in foresight)
    tool_call_names: list[str] = []
    for msg in foresight_msgs:
        if isinstance(msg, AssistantMessage) and msg.is_tool_call():
            for tc in msg.tool_calls:
                tool_call_names.append(tc.name)
    redundant = len(tool_call_names) - len(set(tool_call_names))

    penalty = errors + 0.5 * redundant
    if penalty == 0:
        return 1.0
    elif penalty <= 1:
        return 0.6
    elif penalty <= 2:
        return 0.3
    else:
        return 0.0


def _compute_user_sentiment(foresight_msgs: list) -> float:
    """
    Score the tone of the user's response(s) during foresight.

    Uses keyword matching on the last user message in foresight.
    Returns:
      0.75  — clearly positive (confirming, agreeing)
      0.50  — neutral or no user message
      0.25  — clearly negative (confused, objecting)
    """
    user_msgs = [m for m in foresight_msgs if isinstance(m, UserMessage)]
    if not user_msgs:
        return 0.5

    last_content = user_msgs[-1].content or ""
    words = set(last_content.lower().split())

    # Strip punctuation from words for cleaner matching
    words = {w.strip(".,!?;:\"'()[]{}") for w in words}

    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)

    if pos > neg:
        return 0.75
    elif neg > pos:
        return 0.25
    else:
        return 0.5


def _compute_termination_score(orch: Orchestrator) -> float:
    """
    Score based on how the simulation terminated (or if it's still running).
      AGENT_STOP (normal end): 1.0
      USER_STOP:               0.7
      Still running:           0.4  (most common in foresight)
      MAX_STEPS:               0.2
      Error terminations:      0.0
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
        return 0.0


def _compute_assertion_score(orch: Orchestrator, task: Task) -> float:
    """
    Check how many env_assertions pass against the current environment state.
    Returns fraction of passing assertions, or 0.5 if no assertions defined.
    """
    if task.evaluation_criteria is None:
        return 0.5

    env_assertions = task.evaluation_criteria.env_assertions
    if not env_assertions:
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
            pass

    return passed / total if total > 0 else 0.5


def _compute_action_overlap_score(orch: Orchestrator, task: Task) -> float:
    """
    Fallback when no env_assertions: set-intersection of expected vs actual tools.
    """
    if task.evaluation_criteria is None or not task.evaluation_criteria.actions:
        return 0.5

    expected_names = {
        a.name
        for a in task.evaluation_criteria.actions
        if a.requestor == "assistant"
    }
    if not expected_names:
        return 0.5

    actual_names = set()
    for msg in orch.trajectory:
        if isinstance(msg, AssistantMessage) and msg.is_tool_call():
            for tc in msg.tool_calls:
                actual_names.add(tc.name)

    overlap = len(expected_names & actual_names)
    return overlap / len(expected_names)
