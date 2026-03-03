"""
Adapter layer: connects our PD implementation to tau2-bench's internal APIs.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.user.user_simulator import UserSimulator


def get_environment_constructor(domain: str):
    """Return the environment constructor function for a given domain."""
    if domain == "retail":
        from tau2.domains.retail.environment import get_environment
        return get_environment
    elif domain == "airline":
        from tau2.domains.airline.environment import get_environment
        return get_environment
    else:
        raise ValueError(f"Unsupported domain: {domain}. Choose from: retail, airline")


def get_tasks(domain: str, task_split: str = "base") -> list[Task]:
    """Load tasks for a given domain."""
    if domain == "retail":
        from tau2.domains.retail.environment import get_tasks as _get_tasks
    elif domain == "airline":
        from tau2.domains.airline.environment import get_tasks as _get_tasks
    else:
        raise ValueError(f"Unsupported domain: {domain}")
    return _get_tasks(task_split_name=task_split)


def load_task_split(
    domain: str,
    split: str,
    splits_path: str = "configs/task_splits.json",
) -> list[str]:
    """
    Load task IDs for a given domain and split ('train' or 'test').

    Args:
        domain: 'retail' or 'airline'
        split: 'train', 'test', or 'all'
        splits_path: path to the task_splits.json file

    Returns:
        List of task ID strings.
    """
    if split == "all":
        return None  # Caller interprets None as "use all tasks"
    path = Path(splits_path)
    if not path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        path = project_root / splits_path
    if not path.exists():
        raise FileNotFoundError(
            f"Task splits file not found: {splits_path}. "
            "Run: python scripts/create_task_splits.py"
        )
    with open(path, encoding="utf-8") as f:
        splits = json.load(f)
    if domain not in splits:
        raise ValueError(f"Domain '{domain}' not in {splits_path}")
    if split not in splits[domain]:
        raise ValueError(f"Split '{split}' not in {splits_path}['{domain}']")
    return splits[domain][split]


def create_agent(
    domain: str,
    environment: Environment,
    model_name: str,
    model_args: Optional[dict] = None,
) -> LLMAgent:
    """Create an LLMAgent for the given domain and environment."""
    if model_args is None:
        model_args = {"temperature": 0.0}
    tools = environment.get_tools()
    policy = environment.get_policy()
    return LLMAgent(
        tools=tools,
        domain_policy=policy,
        llm=model_name,
        llm_args=model_args,
    )


def create_user_simulator(
    domain: str,
    environment: Environment,
    task: Task,
    model_name: str,
    model_args: Optional[dict] = None,
) -> UserSimulator:
    """Create a UserSimulator for the given domain, environment, and task."""
    if model_args is None:
        model_args = {"temperature": 0.0}
    user_tools = None
    try:
        user_tools = environment.get_user_tools()
    except ValueError:
        pass  # No user tools for this domain
    return UserSimulator(
        tools=user_tools,
        instructions=task.user_scenario.instructions if task.user_scenario else None,
        llm=model_name,
        llm_args=model_args,
    )


def create_orchestrator(
    domain: str,
    task: Task,
    agent_model: str,
    user_model: str,
    agent_model_args: Optional[dict] = None,
    user_model_args: Optional[dict] = None,
    max_steps: int = 30,
) -> Orchestrator:
    """
    Create a fully initialized Orchestrator for a given task.
    Returns the orchestrator BEFORE calling initialize() so callers can control that.
    """
    env_constructor = get_environment_constructor(domain)
    environment = env_constructor()

    agent = create_agent(domain, environment, agent_model, agent_model_args)
    user = create_user_simulator(domain, environment, task, user_model, user_model_args)

    orch = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=max_steps,
        solo_mode=False,
    )
    return orch


# ─── State fork / restore ───────────────────────────────────────────────────

def save_orchestrator_state(orch: Orchestrator) -> dict:
    """
    Snapshot all mutable state from the orchestrator.
    Safe to call at any point during the simulation.
    """
    state = {
        "agent_state": deepcopy(orch.agent_state),
        "user_state": deepcopy(orch.user_state),
        "trajectory": deepcopy(orch.trajectory),
        "from_role": orch.from_role,
        "to_role": orch.to_role,
        "message": deepcopy(orch.message),
        "step_count": orch.step_count,
        "done": orch.done,
        "termination_reason": orch.termination_reason,
        "num_errors": orch.num_errors,
        # DB state
        "agent_db": deepcopy(orch.environment.tools.db) if orch.environment.tools else None,
        "user_db": deepcopy(orch.environment.user_tools.db) if orch.environment.user_tools else None,
    }
    return state


def restore_orchestrator_state(orch: Orchestrator, state: dict) -> None:
    """
    Restore the orchestrator to a previously saved state.
    Modifies the orchestrator in place.
    """
    orch.agent_state = deepcopy(state["agent_state"])
    orch.user_state = deepcopy(state["user_state"])
    orch.trajectory = deepcopy(state["trajectory"])
    orch.from_role = state["from_role"]
    orch.to_role = state["to_role"]
    orch.message = deepcopy(state["message"])
    orch.step_count = state["step_count"]
    orch.done = state["done"]
    orch.termination_reason = state["termination_reason"]
    orch.num_errors = state["num_errors"]
    # Restore DB state
    if state["agent_db"] is not None and orch.environment.tools is not None:
        orch.environment.tools.db = deepcopy(state["agent_db"])
    if state["user_db"] is not None and orch.environment.user_tools is not None:
        orch.environment.user_tools.db = deepcopy(state["user_db"])
    orch.environment.sync_tools()


def set_agent_temperature(orch: Orchestrator, temperature: float) -> None:
    """Temporarily change the agent's generation temperature."""
    orch.agent.llm_args["temperature"] = temperature


def configure_litellm_for_dashscope(api_key: Optional[str] = None) -> None:
    """
    Configure litellm to use the DashScope API.
    Call this once before creating any orchestrators.
    """
    import litellm

    key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise EnvironmentError(
            "DASHSCOPE_API_KEY not set. Export it or pass it explicitly."
        )
    # litellm picks up OPENAI_API_KEY and OPENAI_API_BASE for openai/* models
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    litellm.drop_params = True
