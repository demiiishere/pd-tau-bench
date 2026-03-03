"""
Build SFT and DPO training datasets from raw PD trajectories.

Run:
    python -m src.data_generation.build_dataset \
        --raw-dir data/raw_trajectories/ \
        --sft-output data/sft_dataset/train.jsonl \
        --dpo-output data/dpo_dataset/train.jsonl

    # Build test split for evaluation
    python -m src.data_generation.build_dataset \
        --split test \
        --sft-output data/sft_dataset/test.jsonl \
        --dpo-output data/dpo_dataset/test.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.predictive_decoding.tau_bench_adapter import load_task_split


def _get_split_ids(split: str, domains: list[str]) -> set[str] | None:
    """
    Return the set of task IDs for the given split across all domains,
    or None if split='all'.
    """
    if split == "all":
        return None
    ids = set()
    for domain in domains:
        try:
            domain_ids = load_task_split(domain, split)
            if domain_ids:
                ids.update(domain_ids)
        except Exception:
            pass  # If splits file missing, don't filter
    return ids if ids else None


def build_sft_dataset(
    raw_dir: Path,
    output_path: Path,
    source_filter: str = "pd",
    allowed_task_ids: set | None = None,
):
    """
    Extract successful trajectories and convert to SFT format (ChatML).
    Only includes trajectories where final_reward == 1.

    Args:
        allowed_task_ids: If set, only include trajectories from these task IDs.
                          Use this to restrict to train split.
    """
    sft_data = []
    if source_filter == "baseline":
        pattern = "**/*_baseline.json"
    elif source_filter == "bon":
        pattern = "**/*_bon_summary.json"
    else:
        pattern = f"**/*_{source_filter}.json"

    for filepath in sorted(raw_dir.glob(pattern)):
        with open(filepath, encoding="utf-8") as f:
            traj = json.load(f)

        task_id = str(traj.get("task_id", ""))
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue

        if traj.get("final_reward") != 1.0:
            continue

        # BoN uses best_conversation, others use conversation
        conv_key = "best_conversation" if source_filter == "bon" else "conversation"
        conv = traj.get(conv_key, traj.get("conversation", []))

        # Convert conversation to ChatML messages
        messages = _conv_to_chatml(conv)
        if not messages:
            continue

        sft_data.append({
            "messages": messages,
            "task_id": task_id,
            "source": source_filter,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"SFT dataset ({source_filter}): {len(sft_data)} successful trajectories → {output_path}")
    return len(sft_data)


def _conv_to_chatml(conversation: list[dict]) -> list[dict]:
    """Convert raw conversation list to ChatML format."""
    messages = []
    for msg in conversation:
        role = msg["role"]
        if role == "system":
            messages.append({"role": "system", "content": msg["content"]})
        elif role == "user":
            messages.append({"role": "user", "content": msg.get("content", "")})
        elif role == "assistant":
            if msg.get("tool_calls"):
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in msg["tool_calls"]
                    ],
                })
            else:
                messages.append({"role": "assistant", "content": msg.get("content", "")})
        elif role == "tool":
            if isinstance(msg.get("content"), list):
                # MultiToolMessage → expand to individual tool messages
                for sub in msg["content"]:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": sub.get("tool_call_id", ""),
                        "content": sub.get("content", ""),
                    })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                })
    return messages


def _candidate_to_chatml(candidate: dict) -> dict:
    """
    Convert a candidate message dict (from PD step record) to OpenAI-compatible
    ChatML format suitable for DPO training.

    This preserves the full structure (tool_calls if present), unlike the old
    plain-text _candidate_to_text() which lost tool_call structure.
    """
    if candidate.get("tool_calls"):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in candidate["tool_calls"]
            ],
        }
    else:
        return {
            "role": "assistant",
            "content": candidate.get("content") or "",
        }


def build_dpo_dataset(
    raw_dir: Path,
    output_path: Path,
    min_score_gap: float = 0.1,
    allowed_task_ids: set | None = None,
):
    """
    Extract preference pairs from PD trajectories.
    Each PD decision step with a meaningful score gap → one DPO pair.

    The chosen/rejected fields are full structured message dicts (OpenAI format),
    not plain text, so they work directly with TRL's DPO trainer.

    Args:
        allowed_task_ids: If set, only include trajectories from these task IDs.
    """
    dpo_data = []

    for filepath in sorted(raw_dir.glob("**/*_pd.json")):
        with open(filepath, encoding="utf-8") as f:
            traj = json.load(f)

        task_id = str(traj.get("task_id", ""))
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue

        for step in traj.get("steps", []):
            scores = step.get("scores", [])
            candidates = step.get("candidates", [])

            # Skip steps that were skipped due to identical candidates
            if step.get("skipped_identical"):
                continue

            if not scores or not candidates:
                continue

            # Filter out None scores (from skipped steps)
            valid = [(s, c) for s, c in zip(scores, candidates) if s is not None]
            if len(valid) < 2:
                continue

            valid_scores = [s for s, _ in valid]
            valid_candidates = [c for _, c in valid]

            best_idx = max(range(len(valid_scores)), key=lambda i: valid_scores[i])
            worst_idx = min(range(len(valid_scores)), key=lambda i: valid_scores[i])
            gap = valid_scores[best_idx] - valid_scores[worst_idx]

            if gap < min_score_gap:
                continue

            chosen_raw = valid_candidates[best_idx]
            rejected_raw = valid_candidates[worst_idx]

            # Convert to full structured ChatML format (preserves tool_calls)
            chosen = _candidate_to_chatml(chosen_raw)
            rejected = _candidate_to_chatml(rejected_raw)

            # Skip if chosen and rejected are identical after conversion
            if chosen == rejected:
                continue

            # Prompt = conversation history up to (and including) the user/env message
            prompt = _conv_to_chatml(step.get("conversation_history", []))

            dpo_data.append({
                "prompt": prompt,           # list of ChatML message dicts
                "chosen": chosen,           # single ChatML message dict (assistant)
                "rejected": rejected,       # single ChatML message dict (assistant)
                "chosen_score": valid_scores[best_idx],
                "rejected_score": valid_scores[worst_idx],
                "score_gap": gap,
                "task_id": task_id,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"DPO dataset: {len(dpo_data)} preference pairs → {output_path}")
    return len(dpo_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw_trajectories")
    parser.add_argument("--sft-output", default="data/sft_dataset/train.jsonl")
    parser.add_argument("--sft-baseline-output", default="data/sft_dataset/train_baseline.jsonl")
    parser.add_argument("--sft-bon-output", default="data/sft_dataset/train_bon.jsonl")
    parser.add_argument("--dpo-output", default="data/dpo_dataset/train.jsonl")
    parser.add_argument("--min-score-gap", type=float, default=0.1)
    parser.add_argument(
        "--split", default="train", choices=["train", "test", "all"],
        help="Only include trajectories from this task split (default: train). "
             "IMPORTANT: always use 'train' for building training data."
    )
    parser.add_argument(
        "--domains", nargs="+", default=["retail", "airline"],
        help="Domains to load split IDs from."
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    # Load allowed task IDs for the requested split
    allowed_ids = _get_split_ids(args.split, args.domains)
    if allowed_ids is not None:
        print(f"Using task split '{args.split}': {len(allowed_ids)} task IDs allowed.")
    else:
        print("Using all tasks (no split filter).")

    # SFT from PD successful trajectories
    n_sft = build_sft_dataset(
        raw_dir, Path(args.sft_output), source_filter="pd",
        allowed_task_ids=allowed_ids,
    )

    # SFT from baseline successful trajectories (for E2 experiment)
    n_sft_bl = build_sft_dataset(
        raw_dir, Path(args.sft_baseline_output), source_filter="baseline",
        allowed_task_ids=allowed_ids,
    )

    # SFT from BoN successful trajectories (for E3 experiment)
    n_sft_bon = build_sft_dataset(
        raw_dir, Path(args.sft_bon_output), source_filter="bon",
        allowed_task_ids=allowed_ids,
    )

    # DPO preference pairs from PD decision steps
    n_dpo = build_dpo_dataset(
        raw_dir, Path(args.dpo_output), args.min_score_gap,
        allowed_task_ids=allowed_ids,
    )

    print(f"\nSummary (split='{args.split}'):")
    print(f"  SFT (PD):       {n_sft} samples")
    print(f"  SFT (baseline): {n_sft_bl} samples")
    print(f"  SFT (BoN):      {n_sft_bon} samples")
    print(f"  DPO pairs:      {n_dpo} pairs")


if __name__ == "__main__":
    main()
