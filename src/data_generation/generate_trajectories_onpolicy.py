"""
Generate PD trajectories using local Qwen3-8B via vLLM (on-policy / self-distillation).

Both agent and user simulator use the local vLLM endpoint — no internet / DashScope needed.
Agent runs with thinking ON; user simulator runs with thinking OFF.

Run on server (vllm-env must be serving base model first):
    bash scripts/start_vllm.sh base   # serve base Qwen3-8B, no LoRA, thinking ON

Then in pd-qwen3-8b env:
    # Single task smoke test
    python -m src.data_generation.generate_trajectories_onpolicy \
        --domain retail --task-ids 0 --K 3 --H 2

    # Full train split
    python -m src.data_generation.generate_trajectories_onpolicy \
        --domain retail airline --K 5 --H 2 --num-trials 3 --max-concurrency 3
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
    create_orchestrator,
    get_tasks,
    load_task_split,
)


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


def run_one_trial(
    domain: str,
    task,
    trial: int,
    agent_model: str,
    user_model: str,
    agent_model_args: dict,
    user_model_args: dict,
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
        agent_model_args={**agent_model_args, "temperature": 0.0},
        user_model_args=user_model_args,
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

    pd_tokens = pd_result.get("tokens", {})
    result = {
        "task_id": task.id,
        "trial": trial,
        "pd_reward": pd_result["final_reward"],
        "pd_time_s": pd_result.get("wall_time_s", 0),
        "pd_tokens_total": pd_tokens.get("total", {}).get("total_tokens", 0),
        "pd_tokens_episode": pd_tokens.get("episode", {}).get("total_tokens", 0),
        "pd_tokens_overhead": pd_tokens.get("overhead", {}).get("total_tokens", 0),
        "pd_steps_skipped": pd_result.get("pd_steps_skipped_count", 0),
        "pd_steps_greedy_fb": pd_result.get("pd_steps_greedy_fb_count", 0),
    }

    # Baseline trajectory (trial 0 only to save cost)
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
                agent_model_args=agent_model_args,
                user_model_args=user_model_args,
            )
            with open(baseline_path, "w", encoding="utf-8") as f:
                json.dump(bl_result, f, indent=2, ensure_ascii=False)
            result["baseline_reward"] = bl_result["final_reward"]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", nargs="+", default=["retail"])
    parser.add_argument("--task-ids", nargs="*", default=None,
                        help="Specific task IDs. If given, overrides --split.")
    parser.add_argument(
        "--split", default="train", choices=["train", "test", "all"],
        help="Which task split to use (default: train)."
    )
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--candidate-temperature", type=float, default=0.8)
    parser.add_argument("--foresight-temperature", type=float, default=0.0)
    parser.add_argument(
        "--vllm-url", default="http://localhost:8001/v1",
        help="vLLM OpenAI-compatible endpoint (default: http://localhost:8001/v1)",
    )
    parser.add_argument(
        "--agent-model", default="openai/finetuned",
        help="Agent model name as seen by litellm (matches --served-model-name in vLLM).",
    )
    parser.add_argument(
        "--user-model", default=None,
        help="User simulator model (default: same as --agent-model)",
    )
    parser.add_argument("--output-dir", default="data/raw_trajectories_onpolicy")
    parser.add_argument("--task-split", default="base")
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    # Keep concurrency low: PD makes K*H parallel calls per step
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    if args.user_model is None:
        args.user_model = args.agent_model

    _configure_vllm(args.vllm_url)

    # Agent: thinking ON (to get reasoning in trajectories)
    # User:  thinking OFF (user simulator doesn't need deep reasoning)
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

    # Per-task aggregation: key=(domain, task_id)
    from collections import defaultdict
    task_stats: dict = defaultdict(lambda: {
        "domain": "", "task_id": None,
        "total_time_s": 0.0, "total_tokens": 0,
        "rewards": [], "baseline_reward": None,
    })

    def run_job(job):
        domain, task, trial, output_dir = job
        return run_one_trial(
            domain=domain,
            task=task,
            trial=trial,
            agent_model=args.agent_model,
            user_model=args.user_model,
            agent_model_args=agent_model_args,
            user_model_args=user_model_args,
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
            key = (domain, str(task.id))
            try:
                result = future.result()
                pd_r = result.get("pd_reward", "?")
                bl_r = result.get("baseline_reward", "-")
                tok_total = result.get("pd_tokens_total", 0)
                pd_time = result.get("pd_time_s", 0)
                skipped = result.get("pd_steps_skipped", 0)
                logger.info(
                    f"[{domain}] Task {task.id} trial {trial}: "
                    f"PD={pd_r}, baseline={bl_r}, time={pd_time:.1f}s, "
                    f"tokens={tok_total}, skipped={skipped}"
                )
                # Aggregate per-task stats
                s = task_stats[key]
                s["domain"] = domain
                s["task_id"] = str(task.id)
                s["total_time_s"] += pd_time
                s["total_tokens"] += tok_total
                s["rewards"].append(pd_r)
                if bl_r != "-" and s["baseline_reward"] is None:
                    s["baseline_reward"] = bl_r
                successes += 1
            except Exception as e:
                logger.error(f"[{domain}] Task {task.id} trial {trial} FAILED: {e}")
                failures += 1

    elapsed = time.time() - start

    # ── Per-task timing summary ────────────────────────────────────────────────
    logger.info("\n=== Per-task timing summary (PD) ===")
    logger.info(f"{'domain':<10} {'task_id':<10} {'trials':<7} {'total_time_s':<14} {'rewards':<15} {'baseline'}")
    timing_rows = []
    for key in sorted(task_stats):
        s = task_stats[key]
        n_trials = len(s["rewards"])
        rewards_str = str(s["rewards"])
        bl = s["baseline_reward"] if s["baseline_reward"] is not None else "-"
        logger.info(
            f"{s['domain']:<10} {s['task_id']:<10} {n_trials:<7} "
            f"{s['total_time_s']:<14.1f} {rewards_str:<15} {bl}"
        )
        timing_rows.append({
            "domain": s["domain"],
            "task_id": s["task_id"],
            "num_trials": n_trials,
            "total_time_s": round(s["total_time_s"], 2),
            "total_tokens": s["total_tokens"],
            "rewards": s["rewards"],
            "pass_at_1": sum(1 for r in s["rewards"] if r == 1.0) / max(len(s["rewards"]), 1),
            "baseline_reward": s["baseline_reward"],
        })

    # Save timing summary JSON
    summary_path = Path(args.output_dir) / "pd_timing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(timing_rows, f, indent=2, ensure_ascii=False)

    total_tasks = len(task_stats)
    avg_time = sum(s["total_time_s"] for s in task_stats.values()) / max(total_tasks, 1)

    # ── Success-rate summary ───────────────────────────────────────────────────
    if timing_rows:
        from collections import Counter
        n_tasks = total_tasks
        n_oracle = sum(1 for r in timing_rows if any(x == 1.0 for x in r["rewards"]))
        avg_pass1 = sum(r["pass_at_1"] for r in timing_rows) / n_tasks
        num_trials = args.num_trials
        dist = Counter(sum(1 for x in r["rewards"] if x == 1.0) for r in timing_rows)

        logger.info(f"\n=== PD (K={args.K}, H={args.H}, {num_trials} trials) — {args.domain} {args.split} split ({n_tasks} tasks) ===")
        logger.info(f"  Oracle (任意1次成功): {n_oracle}/{n_tasks} = {100*n_oracle/n_tasks:.1f}%")
        logger.info(f"  Avg pass@1:          {avg_pass1:.3f}")
        logger.info(f"  Avg time / task:     {avg_time:.1f}s  (all {num_trials} trials combined)")
        logger.info(f"  Pass@1 分布 (per task across {num_trials} trials):")
        for k in range(num_trials + 1):
            cnt = dist.get(k, 0)
            logger.info(f"    {k}/{num_trials} 次成功: {cnt} tasks ({100*cnt/max(n_tasks,1):.0f}%)")

    logger.info(
        f"\nDone in {elapsed:.1f}s: {successes} trials OK, {failures} failed, "
        f"{total_tasks} tasks, avg {avg_time:.1f}s/task"
    )
    logger.info(f"Timing summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
