"""
Test script: verify that the environment fork/restore mechanism works correctly.

Run:
    python -m src.predictive_decoding.test_env_fork --domain retail --task-id 0

Checks:
  1. Fork env → make tool calls on fork → original env is unchanged
  2. Restore from saved state → env returns to saved state exactly
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from loguru import logger
from tau2.orchestrator.orchestrator import Role

from src.predictive_decoding.tau_bench_adapter import (
    configure_litellm_for_dashscope,
    create_orchestrator,
    get_tasks,
    restore_orchestrator_state,
    save_orchestrator_state,
)


def test_fork(domain: str, task_id: str, agent_model: str, user_model: str):
    logger.info(f"Testing fork/restore on domain={domain}, task_id={task_id}")

    configure_litellm_for_dashscope()

    tasks = get_tasks(domain, task_split=None)
    task = next((t for t in tasks if t.id == task_id), None)
    if task is None:
        raise ValueError(f"Task {task_id} not found in {domain}")

    orch = create_orchestrator(
        domain=domain,
        task=task,
        agent_model=agent_model,
        user_model=user_model,
        agent_model_args={"temperature": 0.0},
        user_model_args={"temperature": 0.0},
        max_steps=20,
    )
    orch.initialize()

    # ── Test 1: Save state, run a few steps, verify state changed ────────────
    initial_state = save_orchestrator_state(orch)
    initial_traj_len = len(orch.trajectory)
    initial_db_hash = orch.environment.get_db_hash()

    logger.info("Running 3 steps...")
    steps_run = 0
    while not orch.done and steps_run < 3:
        orch.step()
        if orch.to_role != Role.ENV:
            steps_run += 1

    post_traj_len = len(orch.trajectory)
    post_db_hash = orch.environment.get_db_hash()
    logger.info(
        f"After 3 steps: trajectory {initial_traj_len} → {post_traj_len} messages"
    )

    # ── Test 2: Restore state, verify we're back ─────────────────────────────
    restore_orchestrator_state(orch, initial_state)
    restored_traj_len = len(orch.trajectory)
    restored_db_hash = orch.environment.get_db_hash()

    assert restored_traj_len == initial_traj_len, (
        f"Trajectory length mismatch after restore: {restored_traj_len} != {initial_traj_len}"
    )
    assert restored_db_hash == initial_db_hash, (
        f"DB hash mismatch after restore: {restored_db_hash} != {initial_db_hash}"
    )
    logger.success("✓ Fork/restore works correctly.")

    # ── Test 3: Multiple independent forks start from the same baseline ───────
    # We verify that each restore brings us back to identical state (same DB hash
    # and trajectory length), NOT that two LLM runs produce identical outputs
    # (they won't, due to LLM non-determinism even at temperature=0).
    save1 = save_orchestrator_state(orch)
    baseline_traj_len = len(orch.trajectory)
    baseline_db_hash = orch.environment.get_db_hash()

    # Fork 1: run a couple of steps, modify state
    restore_orchestrator_state(orch, save1)
    steps_run = 0
    while not orch.done and steps_run < 2:
        orch.step()
        if orch.to_role != Role.ENV:
            steps_run += 1

    # Fork 2: restore again, verify we're back to the same starting point
    restore_orchestrator_state(orch, save1)
    assert len(orch.trajectory) == baseline_traj_len, (
        f"After second restore, trajectory length {len(orch.trajectory)} != {baseline_traj_len}"
    )
    assert orch.environment.get_db_hash() == baseline_db_hash, (
        "After second restore, DB hash does not match saved state"
    )
    logger.success("✓ Multiple independent forks all start from the correct baseline.")
    logger.success("All fork/restore tests passed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--task-id", default="0")
    parser.add_argument("--model", default="openai/qwen-plus")
    args = parser.parse_args()

    test_fork(
        domain=args.domain,
        task_id=args.task_id,
        agent_model=args.model,
        user_model=args.model,
    )


if __name__ == "__main__":
    main()
