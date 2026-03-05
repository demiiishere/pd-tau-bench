"""
Evaluate a fine-tuned model on tau-bench test split.

Setup: SSH tunnel to the GPU server (in a separate terminal, leave it running):
    ssh -N -L 8001:localhost:8001 <ssh-host>
    # Verify: curl http://localhost:8001/v1/models

Run:
    python3.11 -m src.evaluation.eval_on_tau_bench \
        --domain retail \
        --split test \
        --agent-model openai/finetuned \
        --vllm-url http://localhost:8001/v1 \
        --user-model openai/qwen-plus \
        --num-trials 3 \
        --output-dir outputs/results/sft_pd
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import litellm
from loguru import logger


def evaluate_model(
    domain: str,
    agent_model: str,
    vllm_url: str,
    user_model: str,
    num_trials: int,
    split: str,
    output_dir: Path,
    max_concurrency: int,
):
    from src.predictive_decoding.tau_bench_adapter import (
        get_tasks,
        load_task_split,
    )
    from src.data_generation.generate_baseline import run_baseline_episode

    # Routing strategy (inverse pattern — vLLM is the global default):
    #   Agent model  → vLLM via OPENAI_API_BASE (global default)
    #   User model   → DashScope via explicit per-call api_base/api_key in user_model_args
    #
    # This avoids relying on per-call api_base overriding a conflicting global env var.
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not dashscope_key:
        raise EnvironmentError("DASHSCOPE_API_KEY not set")

    os.environ["OPENAI_API_BASE"] = vllm_url
    os.environ["OPENAI_API_KEY"] = "fake"   # vLLM does not validate the key
    litellm.drop_params = True

    # agent_model_args: temperature only; routing uses the global OPENAI_API_BASE above
    agent_model_args = {}

    # user_model_args: explicit DashScope credentials to override the global vLLM base
    user_model_args = {
        "temperature": 0.0,
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": dashscope_key,
    }

    # Load only the test split tasks
    all_tasks = get_tasks(domain, task_split="base")
    split_ids = load_task_split(domain, split)
    if split_ids is not None:
        tasks = [t for t in all_tasks if t.id in split_ids]
    else:
        tasks = all_tasks
    logger.info(f"Evaluating {len(tasks)} tasks (domain={domain}, split={split})")

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(task, trial) for task in tasks for trial in range(num_trials)]

    results = []

    def run_job(job):
        task, trial = job
        out_path = output_dir / f"task_{task.id}_trial_{trial}.json"
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                return json.load(f)
        result = run_baseline_episode(
            domain=domain,
            task=task,
            agent_model=agent_model,
            user_model=user_model,
            temperature=0.0,
            agent_model_args=agent_model_args,
            user_model_args=user_model_args,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            task, trial = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Task {task.id} trial {trial}: reward={result['final_reward']}")
            except Exception as e:
                logger.error(f"Task {task.id} trial {trial} FAILED: {e}")

    # ── Aggregate results ─────────────────────────────────────────────────────
    rewards_by_task: dict[str, list[float]] = {}
    for r in results:
        tid = str(r["task_id"])
        rewards_by_task.setdefault(tid, []).append(r["final_reward"])

    num_tasks = len(rewards_by_task)

    # pass@1: mean reward across all individual trials (standard τ-bench metric)
    all_rewards = [r for rewards in rewards_by_task.values() for r in rewards]
    pass1 = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    # pass@k: proportion of tasks where ALL k trials succeed (reliability metric)
    passk = sum(
        1 for rewards in rewards_by_task.values() if all(r == 1.0 for r in rewards)
    ) / num_tasks if num_tasks else 0.0

    # oracle: proportion of tasks where at least one trial succeeds
    oracle = sum(
        1 for rewards in rewards_by_task.values() if any(r == 1.0 for r in rewards)
    ) / num_tasks if num_tasks else 0.0

    summary = {
        "domain": domain,
        "split": split,
        "agent_model": agent_model,
        "num_tasks": num_tasks,
        "num_trials": num_trials,
        "pass@1": round(pass1, 4),
        f"pass@{num_trials}": round(passk, 4),
        "oracle": round(oracle, 4),
    }
    print(json.dumps(summary, indent=2))

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", nargs="+", default=["retail"])
    parser.add_argument(
        "--split", default="test", choices=["train", "test", "all"],
        help="Which task split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--agent-model", default="finetuned",
        help="Model name as served by vLLM (matches --served-model-name).",
    )
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1")
    parser.add_argument("--user-model", default="openai/qwen-plus")
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/results/eval")
    parser.add_argument("--max-concurrency", type=int, default=5)
    args = parser.parse_args()

    for domain in args.domain:
        evaluate_model(
            domain=domain,
            agent_model=args.agent_model,
            vllm_url=args.vllm_url,
            user_model=args.user_model,
            num_trials=args.num_trials,
            split=args.split,
            output_dir=Path(args.output_dir) / domain,
            max_concurrency=args.max_concurrency,
        )


if __name__ == "__main__":
    main()
