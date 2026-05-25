"""
Generate Best-of-N (BoN) trajectories using local Qwen3-8B via vLLM (on-policy).

Both agent and user simulator use the local vLLM endpoint — no internet / DashScope needed.
Agent runs with thinking ON; user simulator runs with thinking OFF.

Run on server (vllm-env must be serving base model first):
    bash scripts/start_vllm.sh base   # serve base Qwen3-8B, no LoRA, thinking ON

Then in pd-qwen3-8b env:
    # Single task smoke test
    python -m src.data_generation.generate_bon_onpolicy \
        --domain retail --task-ids 0 --N 3

    # Full train split
    python -m src.data_generation.generate_bon_onpolicy \
        --domain retail airline --N 5 --max-concurrency 5
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
    create_orchestrator,
    get_tasks,
    load_task_split,
)
from src.predictive_decoding.core import (
    _compute_final_reward,
    _extract_conversation,
    _sum_usage,
    _add_usage,
    _ZERO_USAGE,
)
from tau2.data_model.simulation import TerminationReason


def _configure_vllm(vllm_url: str) -> None:
    """Route all openai/* calls to the local vLLM endpoint."""
    import litellm
    os.environ["OPENAI_API_KEY"] = "fake"
    os.environ["OPENAI_API_BASE"] = vllm_url
    litellm.drop_params = True
    # Bypass HTTP proxy for localhost — server proxy would return 502 otherwise
    for var in ("no_proxy", "NO_PROXY"):
        existing = os.environ.get(var, "")
        if "localhost" not in existing:
            os.environ[var] = f"localhost,127.0.0.1,{existing}".strip(",")


def run_bon_episode(
    domain: str,
    task,
    agent_model: str,
    user_model: str,
    agent_model_args: dict,
    user_model_args: dict,
    temperature: float = 0.8,
    max_steps: int = 30,
) -> dict:
    """Run a single episode with stochastic decoding. Building block for BoN."""
    episode_start = time.time()

    orch = create_orchestrator(
        domain=domain,
        task=task,
        agent_model=agent_model,
        user_model=user_model,
        agent_model_args={**agent_model_args, "temperature": temperature},
        user_model_args=user_model_args,
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
    agent_model_args: dict,
    user_model_args: dict,
    temperature: float = 0.8,
    max_steps: int = 30,
    output_dir: Path = None,
) -> dict:
    """Run N episodes for a single task and pick the best (oracle reward)."""
    task_start = time.time()
    samples = []

    for n in range(N):
        try:
            result = run_bon_episode(
                domain=domain,
                task=task,
                agent_model=agent_model,
                user_model=user_model,
                agent_model_args=agent_model_args,
                user_model_args=user_model_args,
                temperature=temperature,
                max_steps=max_steps,
            )
            result["sample_idx"] = n
            samples.append(result)
            logger.debug(
                f"  BoN sample {n}/{N}: reward={result['final_reward']}, "
                f"steps={result['num_steps']}"
            )

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
    parser.add_argument(
        "--vllm-url", default="http://localhost:8001/v1",
        help="vLLM OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--agent-model", default="openai/finetuned",
        help="Agent model name as seen by litellm (matches --served-model-name in vLLM).",
    )
    parser.add_argument(
        "--user-model", default=None,
        help="User simulator model (default: same as --agent-model).",
    )
    parser.add_argument("--output-dir", default="data/raw_trajectories_onpolicy")
    parser.add_argument("--task-split", default="base")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=5)
    args = parser.parse_args()

    if args.user_model is None:
        args.user_model = args.agent_model

    _configure_vllm(args.vllm_url)

    agent_model_args = {
        "api_base": args.vllm_url,
        "api_key": "fake",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    user_model_args = {
        "temperature": 0.0,
        "api_base": args.vllm_url,
        "api_key": "fake",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

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
    success_counts = []   # per-task: how many of the N samples succeeded
    timing_rows = []      # for timing summary JSON
    start = time.time()

    def run_job(job):
        domain, task, output_dir = job
        return run_bon_task(
            domain=domain,
            task=task,
            N=args.N,
            agent_model=args.agent_model,
            user_model=args.user_model,
            agent_model_args=agent_model_args,
            user_model_args=user_model_args,
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
                oracle_rewards.append(result["oracle_reward"])
                avg_rewards.append(result["avg_reward"])
                success_counts.append(result["success_count"])
                task_time = result.get("wall_time_s", 0)
                task_tokens = result.get("tokens", {}).get("total_tokens", 0)
                logger.info(
                    f"[{domain}] Task {task.id}: "
                    f"oracle={result['oracle_reward']}, avg={result['avg_reward']:.3f}, "
                    f"successes={result['success_count']}/{args.N}, "
                    f"time={task_time:.1f}s, tokens={task_tokens}"
                )
                timing_rows.append({
                    "domain": domain,
                    "task_id": str(task.id),
                    "N": args.N,
                    "oracle_reward": result["oracle_reward"],
                    "avg_reward": result["avg_reward"],
                    "success_count": result["success_count"],
                    "total_time_s": round(task_time, 2),
                    "total_tokens": task_tokens,
                })
                successes += 1
            except Exception as e:
                logger.error(f"[{domain}] Task {task.id} FAILED: {e}")
                failures += 1

    elapsed = time.time() - start

    if oracle_rewards:
        n_tasks = len(oracle_rewards)
        n_oracle = sum(oracle_rewards)
        avg_pass1 = sum(avg_rewards) / n_tasks
        avg_time = sum(r["total_time_s"] for r in timing_rows) / max(len(timing_rows), 1)

        # Success-count distribution
        from collections import Counter
        dist = Counter(success_counts)

        logger.info(f"\n=== BoN (N={args.N}) — {args.domain} {args.split} split ({n_tasks} tasks) ===")
        logger.info(f"  Oracle (任意1次成功): {int(n_oracle)}/{n_tasks} = {100*n_oracle/n_tasks:.1f}%")
        logger.info(f"  Avg pass@1:          {avg_pass1:.3f}")
        logger.info(f"  Avg time / task:     {avg_time:.1f}s")
        logger.info(f"  Pass@1 分布:")
        for k in range(args.N + 1):
            cnt = dist.get(k, 0)
            logger.info(f"    {k}/{args.N} 次成功: {cnt} tasks ({100*cnt/n_tasks:.0f}%)")

        # Save timing summary
        summary_path = Path(args.output_dir) / "bon_timing_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(timing_rows, f, indent=2, ensure_ascii=False)
        logger.info(f"\nTiming summary saved to: {summary_path}")
        logger.info(f"Done in {elapsed:.1f}s: {successes} tasks OK, {failures} failed")
    else:
        logger.info(f"\nBoN done: {successes} succeeded, {failures} failed")


if __name__ == "__main__":
    main()
