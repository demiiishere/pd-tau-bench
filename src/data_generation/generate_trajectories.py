"""
Generate PD trajectories and optionally baseline trajectories.

Run:
    # Debug: single task, small K and H
    python -m src.data_generation.generate_trajectories \
        --domain retail --task-ids 0 --K 3 --H 1 --model openai/qwen-plus

    # Full run
    python -m src.data_generation.generate_trajectories \
        --domain retail airline --K 5 --H 2 --num-trials 3 \
        --model openai/qwen-plus --max-concurrency 5
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

from src.data_generation.generate_baseline import run_baseline_episode
from src.predictive_decoding.core import run_pd_episode
from src.predictive_decoding.tau_bench_adapter import (
    configure_litellm_for_dashscope,
    create_orchestrator,
    get_tasks,
)


def run_one_trial(
    domain: str,
    task,
    trial: int,
    agent_model: str,
    user_model: str,
    K: int,
    H: int,
    candidate_temperature: float,
    foresight_temperature: float,
    max_steps: int,
    output_dir: Path,
    also_baseline: bool = True,
) -> dict:
    """Run one PD trial (and optionally a baseline) for a single task."""
    # PD trajectory
    orch = create_orchestrator(
        domain=domain,
        task=task,
        agent_model=agent_model,
        user_model=user_model,
        agent_model_args={"temperature": 0.0},
        user_model_args={"temperature": 0.0},
        max_steps=max_steps,
    )
    pd_result = run_pd_episode(
        orchestrator=orch,
        task=task,
        K=K,
        H=H,
        candidate_temperature=candidate_temperature,
        foresight_temperature=foresight_temperature,
        max_steps=max_steps,
    )
    pd_path = output_dir / f"task_{task.id}_trial_{trial}_pd.json"
    with open(pd_path, "w", encoding="utf-8") as f:
        json.dump(pd_result, f, indent=2, ensure_ascii=False)

    result = {"task_id": task.id, "trial": trial, "pd_reward": pd_result["final_reward"]}

    # Baseline trajectory (trial 0 only to save API cost)
    if also_baseline and trial == 0:
        baseline_path = output_dir / f"task_{task.id}_baseline.json"
        if not baseline_path.exists():
            bl_result = run_baseline_episode(
                domain=domain,
                task=task,
                agent_model=agent_model,
                user_model=user_model,
                temperature=0.0,
                max_steps=max_steps,
            )
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(bl_result, f, indent=2, ensure_ascii=False)
            result["baseline_reward"] = bl_result["final_reward"]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", nargs="+", default=["retail"])
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--candidate-temperature", type=float, default=0.8)
    parser.add_argument("--foresight-temperature", type=float, default=0.0)
    parser.add_argument("--model", default="openai/qwen-plus")
    parser.add_argument("--output-dir", default="data/raw_trajectories")
    parser.add_argument("--task-split", default="base")
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    configure_litellm_for_dashscope()

    # Collect all (domain, task, trial) jobs
    jobs = []
    for domain in args.domain:
        tasks = get_tasks(domain, task_split=args.task_split)
        if args.task_ids:
            tasks = [t for t in tasks if t.id in args.task_ids]
        output_dir = Path(args.output_dir) / domain
        output_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            for trial in range(args.num_trials):
                pd_path = output_dir / f"task_{task.id}_trial_{trial}_pd.json"
                if pd_path.exists():
                    logger.info(f"Skip existing: {pd_path.name}")
                    continue
                jobs.append((domain, task, trial, output_dir))

    logger.info(f"Total jobs: {len(jobs)}")
    if not jobs:
        logger.info("Nothing to do.")
        return

    successes = 0
    failures = 0
    start = time.time()

    def run_job(job):
        domain, task, trial, output_dir = job
        return run_one_trial(
            domain=domain,
            task=task,
            trial=trial,
            agent_model=args.model,
            user_model=args.model,
            K=args.K,
            H=args.H,
            candidate_temperature=args.candidate_temperature,
            foresight_temperature=args.foresight_temperature,
            max_steps=args.max_steps,
            output_dir=output_dir,
            also_baseline=not args.no_baseline,
        )

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            domain, task, trial, _ = futures[future]
            try:
                result = future.result()
                pd_r = result.get("pd_reward", "?")
                bl_r = result.get("baseline_reward", "-")
                logger.info(
                    f"[{domain}] Task {task.id} trial {trial}: "
                    f"PD={pd_r}, baseline={bl_r}"
                )
                successes += 1
            except Exception as e:
                logger.error(f"[{domain}] Task {task.id} trial {trial} FAILED: {e}")
                failures += 1

    elapsed = time.time() - start
    logger.info(
        f"\nDone in {elapsed:.1f}s: {successes} succeeded, {failures} failed"
    )


if __name__ == "__main__":
    main()
