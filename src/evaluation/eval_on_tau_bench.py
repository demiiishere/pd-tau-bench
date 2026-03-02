"""
Evaluate a fine-tuned model on tau-bench (Phase B — run on GPU machine).

Requires the model to be served via vLLM:
    vllm serve outputs/models/sft_pd/final --port 8001 --trust-remote-code

Run:
    python -m src.evaluation.eval_on_tau_bench \
        --domain retail \
        --model-url http://localhost:8001/v1 \
        --user-model openai/qwen-plus \
        --num-trials 5
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from loguru import logger


def evaluate_model(
    domain: str,
    model_url: str,
    model_name: str,
    user_model: str,
    num_trials: int,
    task_split: str,
    output_dir: Path,
    max_concurrency: int,
):
    from src.predictive_decoding.tau_bench_adapter import (
        configure_litellm_for_dashscope,
        create_orchestrator,
        get_tasks,
    )
    from src.data_generation.generate_baseline import run_baseline_episode

    configure_litellm_for_dashscope()

    # Override API base for the fine-tuned model
    import litellm
    import os as _os
    _os.environ["OPENAI_API_BASE"] = model_url

    tasks = get_tasks(domain, task_split=task_split)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(task, trial) for task in tasks for trial in range(num_trials)]
    results = []

    def run_job(job):
        task, trial = job
        result = run_baseline_episode(
            domain=domain,
            task=task,
            agent_model=model_name,
            user_model=user_model,
            temperature=0.0,
        )
        out_path = output_dir / f"task_{task.id}_trial_{trial}.json"
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
                logger.error(f"Task {task.id} trial {trial} failed: {e}")

    # Aggregate results
    rewards_by_task = {}
    for r in results:
        tid = r["task_id"]
        if tid not in rewards_by_task:
            rewards_by_task[tid] = []
        rewards_by_task[tid].append(r["final_reward"])

    pass1 = sum(
        1 for rewards in rewards_by_task.values() if rewards[0] == 1.0
    ) / len(rewards_by_task)
    passk = sum(
        1 for rewards in rewards_by_task.values() if any(r == 1.0 for r in rewards)
    ) / len(rewards_by_task)

    summary = {
        "domain": domain,
        "model": model_name,
        "num_tasks": len(rewards_by_task),
        "num_trials": num_trials,
        "pass@1": round(pass1, 4),
        f"pass@{num_trials}": round(passk, 4),
    }
    print(json.dumps(summary, indent=2))

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--model-url", default="http://localhost:8001/v1")
    parser.add_argument("--model-name", default="openai/finetuned-model")
    parser.add_argument("--user-model", default="openai/qwen-plus")
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--task-split", default="base")
    parser.add_argument("--output-dir", default="outputs/results/eval")
    parser.add_argument("--max-concurrency", type=int, default=5)
    args = parser.parse_args()

    evaluate_model(
        domain=args.domain,
        model_url=args.model_url,
        model_name=args.model_name,
        user_model=args.user_model,
        num_trials=args.num_trials,
        task_split=args.task_split,
        output_dir=Path(args.output_dir),
        max_concurrency=args.max_concurrency,
    )


if __name__ == "__main__":
    main()
