"""
Build SFT and DPO training datasets from raw PD trajectories.

Run:
    python -m src.data_generation.build_dataset \
        --raw-dir data/raw_trajectories/ \
        --sft-output data/sft_dataset/train.jsonl \
        --dpo-output data/dpo_dataset/train.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def build_sft_dataset(raw_dir: Path, output_path: Path, source_filter: str = "pd"):
    """
    Extract successful trajectories and convert to SFT format (ChatML).
    Only includes trajectories where final_reward == 1.
    """
    sft_data = []
    pattern = f"**/*_{source_filter}.json" if source_filter != "baseline" else "**/*_baseline.json"

    for filepath in sorted(raw_dir.glob(pattern)):
        with open(filepath, encoding="utf-8") as f:
            traj = json.load(f)

        if traj.get("final_reward") != 1.0:
            continue

        # Convert conversation to ChatML messages
        messages = []
        for msg in traj["conversation"]:
            role = msg["role"]
            if role == "system":
                messages.append({"role": "system", "content": msg["content"]})
            elif role == "user":
                messages.append({"role": "user", "content": msg.get("content", "")})
            elif role == "assistant":
                if msg.get("tool_calls"):
                    # Tool call message
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
                    # MultiToolMessage
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

        sft_data.append({
            "messages": messages,
            "task_id": traj["task_id"],
            "source": source_filter,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"SFT dataset ({source_filter}): {len(sft_data)} successful trajectories → {output_path}")
    return len(sft_data)


def build_dpo_dataset(raw_dir: Path, output_path: Path, min_score_gap: float = 0.1):
    """
    Extract preference pairs from PD trajectories.
    Each PD decision step with a meaningful score gap → one DPO pair.
    """
    dpo_data = []

    for filepath in sorted(raw_dir.glob("**/*_pd.json")):
        with open(filepath, encoding="utf-8") as f:
            traj = json.load(f)

        for step in traj.get("steps", []):
            scores = step.get("scores", [])
            candidates = step.get("candidates", [])
            if not scores or not candidates:
                continue

            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            worst_idx = min(range(len(scores)), key=lambda i: scores[i])
            gap = scores[best_idx] - scores[worst_idx]

            if gap < min_score_gap:
                continue

            # Prompt = conversation history up to (and including) the user message
            prompt = step.get("conversation_history", [])
            chosen = step["candidates"][best_idx]
            rejected = step["candidates"][worst_idx]

            # Convert chosen/rejected to simple text for DPO
            chosen_text = _candidate_to_text(chosen)
            rejected_text = _candidate_to_text(rejected)

            if chosen_text == rejected_text:
                continue  # Skip identical candidates

            dpo_data.append({
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "chosen_score": scores[best_idx],
                "rejected_score": scores[worst_idx],
                "score_gap": gap,
                "task_id": traj["task_id"],
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"DPO dataset: {len(dpo_data)} preference pairs → {output_path}")
    return len(dpo_data)


def _candidate_to_text(candidate: dict) -> str:
    """Convert a candidate message dict to a text representation."""
    if candidate.get("tool_calls"):
        tc = candidate["tool_calls"][0]
        return f"[TOOL] {tc['name']}({json.dumps(tc['arguments'])})"
    return candidate.get("content") or ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw_trajectories")
    parser.add_argument("--sft-output", default="data/sft_dataset/train.jsonl")
    parser.add_argument("--sft-baseline-output", default="data/sft_dataset/train_baseline.jsonl")
    parser.add_argument("--dpo-output", default="data/dpo_dataset/train.jsonl")
    parser.add_argument("--min-score-gap", type=float, default=0.1)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)

    # SFT from PD successful trajectories
    n_sft = build_sft_dataset(raw_dir, Path(args.sft_output), source_filter="pd")

    # SFT from baseline successful trajectories (for E2 experiment)
    n_sft_bl = build_sft_dataset(
        raw_dir, Path(args.sft_baseline_output), source_filter="baseline"
    )

    # DPO preference pairs from PD decision steps
    n_dpo = build_dpo_dataset(raw_dir, Path(args.dpo_output), args.min_score_gap)

    print(f"\nSummary:")
    print(f"  SFT (PD):       {n_sft} samples")
    print(f"  SFT (baseline): {n_sft_bl} samples")
    print(f"  DPO pairs:      {n_dpo} pairs")


if __name__ == "__main__":
    main()
