"""
Generate baseline (standard greedy decoding) trajectories.

Run:
    python -m src.data_generation.generate_baseline \
        --domain retail \
        --task-ids 0 1 2 \
        --model openai/qwen-plus
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from loguru import logger
from tau2.data_model.simulation import TerminationReason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.data_model.simulation import SimulationRun
from tau2.utils.utils import get_now

from src.predictive_decoding.tau_bench_adapter import (
    configure_litellm_for_dashscope,
    create_orchestrator,
    get_tasks,
)
from src.predictive_decoding.core import _extract_conversation, _compute_final_reward


def run_baseline_episode(
    domain: str,
    task,
    agent_model: str,
    user_model: str,
    temperature: float = 0.0,
    max_steps: int = 30,
) -> dict:
    """Run a single episode with standard greedy decoding."""
    orch = create_orchestrator(
        domain=domain,
        task=task,
        agent_model=agent_model,
        user_model=user_model,
        agent_model_args={"temperature": temperature},
        user_model_args={"temperature": 0.0},
        max_steps=max_steps,
    )
    orch.initialize()

    step_count = 0
    while not orch.done and step_count < max_steps:
        orch.step()
        from tau2.orchestrator.orchestrator import Role
        if orch.to_role != Role.ENV:
            step_count += 1

    final_reward = _compute_final_reward(orch, task)
    conversation = _extract_conversation(orch)

    return {
        "task_id": task.id,
        "conversation": conversation,
        "final_reward": final_reward,
        "termination_reason": (
            orch.termination_reason.value if orch.termination_reason else "unknown"
        ),
        "num_steps": step_count,
        "source": "baseline",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="retail")
    parser.add_argument(
        "--task-ids", nargs="*", default=None,
        help="Specific task IDs to generate. Omit to run all tasks."
    )
    parser.add_argument("--model", default="openai/qwen-plus")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", default="data/raw_trajectories")
    parser.add_argument("--task-split", default="base")
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()

    configure_litellm_for_dashscope()

    tasks = get_tasks(args.domain, task_split=args.task_split)
    if args.task_ids:
        tasks = [t for t in tasks if t.id in args.task_ids]
    logger.info(f"Running baseline on {len(tasks)} tasks in '{args.domain}'")

    output_dir = Path(args.output_dir) / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    for i, task in enumerate(tasks):
        logger.info(f"[{i+1}/{len(tasks)}] Task {task.id}")
        try:
            result = run_baseline_episode(
                domain=args.domain,
                task=task,
                agent_model=args.model,
                user_model=args.model,
                temperature=args.temperature,
                max_steps=args.max_steps,
            )
            out_path = output_dir / f"task_{task.id}_baseline.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            reward = result["final_reward"]
            successes += int(reward == 1.0)
            logger.info(
                f"  reward={reward}, steps={result['num_steps']}, "
                f"termination={result['termination_reason']}"
            )
        except Exception as e:
            logger.error(f"  Task {task.id} failed: {e}")

    logger.info(
        f"\nBaseline done: {successes}/{len(tasks)} succeeded "
        f"({100*successes/len(tasks):.1f}%)"
    )


if __name__ == "__main__":
    main()
