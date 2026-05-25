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


def _apply_sliding_window_patch():
    """Monkey-patch tau2.utils.llm_utils.completion to handle context overflow.

    tau2 does `from litellm import completion` at import time, so we must replace
    the name in that module's namespace (not litellm.completion globally).
    When the model raises a context-length error, we drop the two oldest
    non-system messages and retry until the context fits or nothing left to drop.
    """
    import tau2.utils.llm_utils as llm_utils
    from litellm import completion as _orig

    _CTX_KEYWORDS = (
        "context_length_exceeded",
        "maximum context length",
        "input too long",
        "maximum input",
        "token limit",
        "40960",
    )

    def _completion_sliding_window(model, messages, **kwargs):
        current = list(messages)
        while True:
            try:
                return _orig(model=model, messages=current, **kwargs)
            except Exception as exc:
                err = str(exc).lower()
                if not any(k in err for k in _CTX_KEYWORDS):
                    raise
                sys_msgs = [m for m in current if m.get("role") == "system"]
                conv_msgs = [m for m in current if m.get("role") != "system"]
                if len(conv_msgs) <= 2:
                    raise  # nothing left to drop
                dropped = conv_msgs[:2]
                logger.warning(
                    f"Context overflow — dropping oldest 2 messages "
                    f"({dropped[0].get('role')}, {conv_msgs[1].get('role')}). "
                    f"Remaining conv turns: {(len(conv_msgs) - 2) // 2}"
                )
                current = sys_msgs + conv_msgs[2:]

    llm_utils.completion = _completion_sliding_window


def evaluate_model(
    domain: str,
    agent_model: str,
    vllm_url: str,
    user_model: str,
    num_trials: int,
    split: str,
    output_dir: Path,
    max_concurrency: int,
    local: bool = False,
    enable_thinking: bool = False,
):
    from src.predictive_decoding.tau_bench_adapter import (
        get_tasks,
        load_task_split,
    )
    from src.data_generation.generate_baseline import run_baseline_episode

    _apply_sliding_window_patch()

    litellm.drop_params = True

    # Bypass proxy for localhost (server has http_proxy set)
    for var in ("no_proxy", "NO_PROXY"):
        existing = os.environ.get(var, "")
        if "localhost" not in existing:
            os.environ[var] = f"localhost,127.0.0.1,{existing}".strip(",")

    # agent_model_args: local vLLM
    agent_model_args = {
        "api_base": vllm_url,
        "api_key": "fake",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    }

    if local:
        # Both agent and user use local vLLM — no internet required
        user_model_args = {
            "temperature": 0.0,
            "api_base": vllm_url,
            "api_key": "fake",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    else:
        dashscope_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not dashscope_key:
            raise EnvironmentError("DASHSCOPE_API_KEY not set (or use --local)")
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
                failed = {"task_id": task.id, "final_reward": 0.0, "error": str(e)}
                out_path = output_dir / f"task_{task.id}_trial_{trial}.json"
                with open(out_path, "w") as f:
                    json.dump(failed, f, indent=2)
                results.append(failed)

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
    parser.add_argument("--local", action="store_true",
                        help="Use local vLLM for user model too (no internet required).")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking for agent model (use for on-policy models).")
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
            local=args.local,
            enable_thinking=args.enable_thinking,
        )


if __name__ == "__main__":
    main()
