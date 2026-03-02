"""
Analysis script: compare PD trajectories vs baseline trajectories.

Run:
    python -m src.evaluation.analysis \
        --pd-dir data/raw_trajectories/ \
        --baseline-dir data/raw_trajectories/
"""

import argparse
import json
from pathlib import Path


def load_trajectories(directory: Path, pattern: str) -> dict:
    """Load all trajectories matching pattern, keyed by task_id."""
    results = {}
    for filepath in sorted(Path(directory).glob(pattern)):
        with open(filepath, encoding="utf-8") as f:
            traj = json.load(f)
        task_id = traj["task_id"]
        if task_id not in results:
            results[task_id] = []
        results[task_id].append(traj)
    return results


def analyze(pd_dir: str, baseline_dir: str):
    pd_data = {}
    bl_data = {}

    # Load data from all domains
    for domain in ["retail", "airline"]:
        pd_trajs = load_trajectories(Path(pd_dir) / domain, "*_pd.json")
        bl_trajs = load_trajectories(Path(baseline_dir) / domain, "*_baseline.json")
        pd_data.update(pd_trajs)
        bl_data.update(bl_trajs)

    print(f"Loaded {len(pd_data)} tasks with PD trajectories")
    print(f"Loaded {len(bl_data)} tasks with baseline trajectories")

    # ── Success rate comparison ───────────────────────────────────────────────
    pd_wins, bl_wins, both_win, both_fail = [], [], [], []
    common_tasks = set(pd_data.keys()) & set(bl_data.keys())

    for task_id in common_tasks:
        pd_reward = max(t["final_reward"] for t in pd_data[task_id])
        bl_reward = max(t["final_reward"] for t in bl_data[task_id])

        if pd_reward > bl_reward:
            pd_wins.append(task_id)
        elif bl_reward > pd_reward:
            bl_wins.append(task_id)
        elif pd_reward == 1.0:
            both_win.append(task_id)
        else:
            both_fail.append(task_id)

    print(f"\n{'='*60}")
    print(f"Results on {len(common_tasks)} common tasks:")
    print(f"  PD wins (PD=1, BL=0):  {len(pd_wins)} ({100*len(pd_wins)/len(common_tasks):.1f}%)")
    print(f"  BL wins (BL=1, PD=0):  {len(bl_wins)} ({100*len(bl_wins)/len(common_tasks):.1f}%)")
    print(f"  Both succeed:           {len(both_win)} ({100*len(both_win)/len(common_tasks):.1f}%)")
    print(f"  Both fail:              {len(both_fail)} ({100*len(both_fail)/len(common_tasks):.1f}%)")

    pd_success_rate = (len(pd_wins) + len(both_win)) / len(common_tasks)
    bl_success_rate = (len(bl_wins) + len(both_win)) / len(common_tasks)
    print(f"\n  PD success rate:  {pd_success_rate:.3f}")
    print(f"  BL success rate:  {bl_success_rate:.3f}")
    print(f"  PD improvement:   {pd_success_rate - bl_success_rate:+.3f}")

    # ── Decision divergence analysis ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PD vs Baseline — Key Decision Divergences:")
    print(f"(Tasks where PD succeeds but baseline fails)")

    for task_id in pd_wins[:5]:  # Show top 5
        pd_traj = pd_data[task_id][0]
        bl_traj = bl_data[task_id][0]

        pd_conv = pd_traj["conversation"]
        bl_conv = bl_traj["conversation"]

        print(f"\n  Task {task_id}:")
        # Find first divergence
        for i, (pd_msg, bl_msg) in enumerate(zip(pd_conv, bl_conv)):
            if pd_msg["role"] != bl_msg["role"]:
                print(f"    Diverged at message {i} (different roles)")
                break
            if pd_msg["role"] == "assistant":
                pd_content = _msg_preview(pd_msg)
                bl_content = _msg_preview(bl_msg)
                if pd_content != bl_content:
                    print(f"    First divergence at turn {i} (assistant response):")
                    print(f"      BL: {bl_content[:100]}")
                    print(f"      PD: {pd_content[:100]}")
                    break

    # ── Score gap analysis ────────────────────────────────────────────────────
    all_gaps = []
    for task_id, trajs in pd_data.items():
        for traj in trajs:
            for step in traj.get("steps", []):
                scores = step.get("scores", [])
                if scores:
                    gap = max(scores) - min(scores)
                    all_gaps.append(gap)

    if all_gaps:
        import statistics
        print(f"\n{'='*60}")
        print(f"PD Score Gap Analysis ({len(all_gaps)} decision steps):")
        print(f"  Mean gap:    {statistics.mean(all_gaps):.3f}")
        print(f"  Median gap:  {statistics.median(all_gaps):.3f}")
        print(f"  Max gap:     {max(all_gaps):.3f}")
        print(f"  Steps with gap >= 0.1: {sum(1 for g in all_gaps if g >= 0.1)} ({100*sum(1 for g in all_gaps if g >= 0.1)/len(all_gaps):.1f}%)")


def _msg_preview(msg: dict) -> str:
    if msg.get("tool_calls"):
        tc = msg["tool_calls"][0]
        return f"[TOOL] {tc['name']}({tc['arguments']})"
    return msg.get("content", "")[:100]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pd-dir", default="data/raw_trajectories")
    parser.add_argument("--baseline-dir", default="data/raw_trajectories")
    args = parser.parse_args()
    analyze(args.pd_dir, args.baseline_dir)


if __name__ == "__main__":
    main()
