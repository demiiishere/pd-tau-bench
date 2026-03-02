"""
Inspect a generated PD trajectory file to verify quality.

Run:
    python -m src.data_generation.inspect_trajectories \
        --trajectory-file data/raw_trajectories/retail/task_0_trial_0_pd.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def inspect(filepath: str):
    with open(filepath, encoding="utf-8") as f:
        traj = json.load(f)

    print(f"\n{'='*60}")
    print(f"Task ID:          {traj['task_id']}")
    print(f"Final reward:     {traj['final_reward']}")
    print(f"Termination:      {traj['termination_reason']}")
    print(f"Conversation:     {len(traj['conversation'])} messages")
    print(f"PD steps:         {len(traj['steps'])}")

    print(f"\n{'─'*60}")
    print("PD Decision Steps:")
    for i, step in enumerate(traj["steps"]):
        scores = step["scores"]
        chosen_idx = step["chosen_idx"]
        best_score = scores[chosen_idx]
        worst_score = min(scores)
        score_gap = best_score - worst_score
        n_candidates = len(scores)

        print(f"\n  Step {i+1}:")
        print(f"    Candidates: {n_candidates}")
        print(f"    Scores:     {[f'{s:.3f}' for s in scores]}")
        print(f"    Best idx:   {chosen_idx} (score={best_score:.3f})")
        print(f"    Score gap:  {score_gap:.3f}")

        chosen = step["chosen_action"]
        if chosen.get("tool_calls"):
            tc = chosen["tool_calls"][0]
            print(f"    Chosen:     [tool] {tc['name']}({tc['arguments']})")
        else:
            content = (chosen.get("content") or "")[:80]
            print(f"    Chosen:     [text] {content}...")

        if score_gap < 0.05:
            print(f"    ⚠️  Low score gap — PD may not be differentiating candidates")

    print(f"\n{'─'*60}")
    print("Conversation Preview:")
    for msg in traj["conversation"][:8]:
        role = msg["role"]
        if "tool_calls" in msg and msg["tool_calls"]:
            tc = msg["tool_calls"][0]
            content = f"[TOOL CALL] {tc['name']}({tc['arguments']})"
        else:
            content = (msg.get("content") or "")[:100]
        print(f"  [{role:10s}] {content}")
    if len(traj["conversation"]) > 8:
        print(f"  ... ({len(traj['conversation']) - 8} more messages)")

    # Summary assessment
    print(f"\n{'='*60}")
    all_scores = [s for step in traj["steps"] for s in step["scores"]]
    if all_scores:
        import statistics
        score_range = max(all_scores) - min(all_scores)
        avg_gap = statistics.mean(
            [max(s["scores"]) - min(s["scores"]) for s in traj["steps"] if s["scores"]]
        )
        print(f"Score range across all candidates: {score_range:.3f}")
        print(f"Avg score gap per step:           {avg_gap:.3f}")
        if avg_gap < 0.05:
            print("⚠️  Value function has low discrimination — consider tuning it")
        else:
            print("✓  Value function shows meaningful discrimination")

    dpo_pairs = sum(
        1 for step in traj["steps"]
        if step["scores"] and (max(step["scores"]) - min(step["scores"])) >= 0.1
    )
    print(f"Valid DPO pairs from this trajectory: {dpo_pairs}/{len(traj['steps'])}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-file", required=True)
    args = parser.parse_args()
    inspect(args.trajectory_file)


if __name__ == "__main__":
    main()
