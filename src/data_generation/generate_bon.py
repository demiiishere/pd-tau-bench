"""
Generate Best-of-N (BoN) trajectories.

BoN samples N full episodes with temperature=0.8, then picks the one with the
highest reward. This is the key comparison baseline (E2 in the research plan):
it uses the same inference budget as PD but without lookahead.

Run:
    # Train split (default)
    python -m src.data_generation.generate_bon \
        --domain retail airline --N 5 \
        --model openai/qwen-plus --max-concurrency 5

    # Debug: single task
    python -m src.data_generation.generate_bon \
        --domain retail --task-ids 0 --N 3 --model openai/qwen-plus
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from loguru import logger

from src.predictive_decoding.tau_bench_adapter import (
    configure_litellm_for_dashscope,
    create_orchestrator,
    get_tasks,
    load_task_split,
)
from src.predictive_decoding.core import (
    _compute_final_reward,
    _extract_conversation,
)
from tau2.data_model.simulation import TerminationReason


def run_bon_episode(
    domain: str,
    task,
    agent_model: str,
    user_model: str,
    temperature: float = 0.8,
    max_steps: int = 30,
) -> dict:
    """
    Run a single episode with stochastic decoding (temperature > 0).
    Used as the building block for Best-of-N.
    """
    episode_start = time.time()

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
    wall_time_s = time.time() - episode_start

    from src.predictive_decoding.core import _sum_usage
    traj_usage = _sum_usage(orch.get_trajectory())

    return {
        "task_id": task.id,
        "conversation": conversation,
        "final_reward": final_reward,
        "termination_reason": (
            orch.termination_reason.value if orch.termination_reason else "unknown"
        ),
        "num_steps": step_count,
        "wall_time_s": round(wall_time_s, 2),
        "tokens": traj_usage,
        "source": "bon_sample",
    }


def run_bon_task(
    domain: str,
    task,
    N: int,
    agent_model: str,
    user_model: str,
    temperature: float = 0.8,
    max_steps: int = 30,
    output_dir: Path = None,
) -> dict:
    """
    Run N episodes for a single task and pick the best one (oracle best reward).

    Saves all N individual trajectories and a summary JSON with oracle reward.
    Returns a summary dict.
    """
    task_start = time.time()
    samples = []
    total_api_calls = 0

    for n in range(N):
        try:
            result = run_bon_episode(
                domain=domain,
                task=task,
                agent_model=agent_model,
                user_model=user_model,
                temperature=temperature,
                max_steps=max_steps,
            )
            result["sample_idx"] = n
            samples.append(result)
            total_api_calls += result.get("api_calls_approx", 0)
            logger.debug(
                f"  BoN sample {n}/{N}: reward={result['final_reward']}, "
                f"steps={result['num_steps']}"
            )

            # Save individual sample
            if output_dir is not None:
                sample_path = output_dir / f"task_{task.id}_bon_n{n}.json"
                with open(sample_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"BoN sample {n} failed for task {task.id}: {e}")

    if not samples:
        raise RuntimeError(f"All {N} BoN samples failed for task {task.id}")

    best_sample = max(samples, key=lambda s: s["final_reward"])
    oracle_reward = best_sample["final_reward"]
    avg_reward = sum(s["final_reward"] for s in samples) / len(samples)
    success_count = sum(1 for s in samples if s["final_reward"] == 1.0)

    wall_time_s = time.time() - task_start

    # Sum token usage across all N samples
    from src.predictive_decoding.core import _sum_usage, _add_usage, _ZERO_USAGE
    total_tokens = dict(_ZERO_USAGE)
    for s in samples:
        total_tokens = _add_usage(total_tokens, s.get("tokens", _ZERO_USAGE))

    summary = {
        "task_id": task.id,
        "N": N,
        "oracle_reward": oracle_reward,
        "avg_reward": round(avg_reward, 4),
        "success_count": success_count,
        "best_sample_idx": best_sample["sample_idx"],
        "best_conversation": best_sample["conversation"],
        "best_termination_reason": best_sample["termination_reason"],
        "wall_time_s": round(wall_time_s, 2),
        "tokens": total_tokens,
        "source": "bon",
    }

    if output_dir is not None:
        summary_path = output_dir / f"task_{task.id}_bon_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", nargs="+", default=["retail"])
    parser.add_argument("--task-ids", nargs="*", default=None,
                        help="Specific task IDs. If given, overrides --split.")
    parser.add_argument(
        "--split", default="train", choices=["train", "test", "all"],
        help="Which task split to use (default: train)."
    )
    parser.add_argument("--N", type=int, default=5,
                        help="Number of samples per task for BoN.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (should be > 0 for diversity).")
    parser.add_argument("--model", default="openai/qwen-plus")
    parser.add_argument("--output-dir", default="data/raw_trajectories")
    parser.add_argument("--task-split", default="base",
                        help="tau2-bench internal split name (usually 'base')")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=5)
    args = parser.parse_args()

    configure_litellm_for_dashscope()

    jobs = []
    for domain in args.domain:
        all_tasks = get_tasks(domain, task_split=args.task_split)

        if args.task_ids:
            tasks = [t for t in all_tasks if t.id in args.task_ids]
        else:
            split_ids = load_task_split(domain, args.split)
            if split_ids is not None:
                tasks = [t for t in all_tasks if t.id in split_ids]
            else:
                tasks = all_tasks

        logger.info(f"Domain={domain}: {len(tasks)} tasks (split={args.task_ids or args.split})")

        output_dir = Path(args.output_dir) / domain
        output_dir.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            summary_path = output_dir / f"task_{task.id}_bon_summary.json"
            if summary_path.exists():
                logger.info(f"Skip existing: {summary_path.name}")
                continue
            jobs.append((domain, task, output_dir))

    logger.info(f"Total BoN tasks: {len(jobs)} (N={args.N} samples each)")
    if not jobs:
        logger.info("Nothing to do.")
        return

    successes = 0
    failures = 0
    oracle_rewards = []
    avg_rewards = []
    total_api_calls = 0
    start = time.time()

    def run_job(job):
        domain, task, output_dir = job
        return run_bon_task(
            domain=domain,
            task=task,
            N=args.N,
            agent_model=args.model,
            user_model=args.model,
            temperature=args.temperature,
            max_steps=args.max_steps,
            output_dir=output_dir,
        )

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            domain, task, _ = futures[future]
            try:
                result = future.result()
                oracle_r = result["oracle_reward"]
                avg_r = result["avg_reward"]
                n_ok = result["success_count"]
                api_c = result.get("api_calls_approx", 0)
                total_api_calls += api_c
                oracle_rewards.append(oracle_r)
                avg_rewards.append(avg_r)
                logger.info(
                    f"[{domain}] Task {task.id}: "
                    f"oracle={oracle_r}, avg={avg_r:.3f}, "
                    f"successes={n_ok}/{args.N}, api_calls≈{api_c}"
                )
                successes += 1
            except Exception as e:
                logger.error(f"[{domain}] Task {task.id} FAILED: {e}")
                failures += 1

    elapsed = time.time() - start

    if oracle_rewards:
        pass1 = sum(oracle_rewards) / len(oracle_rewards)
        avg_pass = sum(avg_rewards) / len(avg_rewards)
        logger.info(
            f"\nBoN Done in {elapsed:.1f}s: {successes} tasks succeeded, {failures} failed"
            f"\n  Oracle pass@{args.N}: {pass1:.3f} ({sum(oracle_rewards)}/{len(oracle_rewards)})"
            f"\n  Avg pass@1:    {avg_pass:.3f}"
            f"\n  Total api_calls≈{total_api_calls}"
        )
    else:
        logger.info(f"\nBoN Done: {successes} succeeded, {failures} failed")


if __name__ == "__main__":
    main()
