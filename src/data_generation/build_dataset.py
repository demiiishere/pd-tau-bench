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
        # Use all individual BoN samples (not summary), so all successful
        # trajectories are included — gives ~196 samples vs 61 from summary.
        pattern = "**/*_bon_n*.json"
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

        conv = traj.get("conversation", [])

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


def build_bon_dpo_dataset(
    raw_dir: Path,
    output_path: Path,
    allowed_task_ids: set | None = None,
):
    """
    Build episode-level DPO pairs from BoN trajectories (for E2+ baseline).

    For each task that has both successful and failed BoN episodes, pair the
    FIRST agent action from a randomly-chosen success with that from a failure.
    The prompt is the shared prefix [system, first_user_msg] — identical for
    both since they start from the same task.

    This gives episode-level preference pairs: the DPO signal is "given this
    task, the first agent action in a successful episode is preferred over the
    first action in a failed episode."

    Pairs where chosen==rejected are dropped (often happens when the first
    action is deterministic, e.g. always calling get_order_details first).
    """
    import random as _random
    from collections import defaultdict

    successes: dict[str, list] = defaultdict(list)
    failures: dict[str, list] = defaultdict(list)

    for filepath in sorted(raw_dir.glob("**/*_bon_n*.json")):
        with open(filepath, encoding="utf-8") as f:
            traj = json.load(f)

        task_id = str(traj.get("task_id", ""))
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue

        msgs = _conv_to_chatml(traj.get("conversation", []))
        if not msgs:
            continue

        if traj.get("final_reward") == 1.0:
            successes[task_id].append(msgs)
        else:
            failures[task_id].append(msgs)

    _random.seed(42)
    dpo_data = []

    for task_id in sorted(set(successes) & set(failures)):
        s_msgs = _random.choice(successes[task_id])
        f_msgs = _random.choice(failures[task_id])

        # Find first assistant message and everything before it
        def _split_at_first_agent(msgs):
            for i, m in enumerate(msgs):
                if m["role"] == "assistant":
                    return msgs[:i], msgs[i]
            return None, None

        prompt_s, chosen = _split_at_first_agent(s_msgs)
        _, rejected = _split_at_first_agent(f_msgs)

        if chosen is None or rejected is None:
            continue
        if chosen == rejected:
            continue

        dpo_data.append({
            "prompt": prompt_s,       # [system, first_user_msg]
            "chosen": chosen,         # first agent action from successful episode
            "rejected": rejected,     # first agent action from failed episode
            "chosen_score": 1.0,
            "rejected_score": 0.0,
            "score_gap": 1.0,
            "task_id": task_id,
            "source": "bon_episode",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"BoN-DPO dataset (episode-level): {len(dpo_data)} pairs → {output_path}")
    return len(dpo_data)


def _append_jsonl(src: Path, dst: Path):
    """Append contents of src JSONL to dst JSONL."""
    with open(src, encoding="utf-8") as fsrc, \
         open(dst, "a", encoding="utf-8") as fdst:
        for line in fsrc:
            line = line.strip()
            if line:
                fdst.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains", nargs="+", default=["retail", "airline"],
        help="Domains to build datasets for. Each domain is processed separately "
             "to avoid task-ID collisions across domains."
    )
    parser.add_argument("--raw-dir", default="data/raw_trajectories",
                        help="Parent directory containing per-domain subdirectories.")
    parser.add_argument("--sft-output", default="data/sft_dataset/train.jsonl")
    parser.add_argument("--sft-baseline-output", default="data/sft_dataset/train_baseline.jsonl")
    parser.add_argument("--sft-bon-output", default="data/sft_dataset/train_bon.jsonl")
    parser.add_argument("--dpo-output", default="data/dpo_dataset/train.jsonl")
    parser.add_argument("--bon-dpo-output", default="data/dpo_dataset/train_bon_episode.jsonl",
                        help="Episode-level DPO pairs from BoN (for E2+ baseline).")
    parser.add_argument("--min-score-gap", type=float, default=0.1)
    parser.add_argument(
        "--split", default="train", choices=["train", "test", "all"],
        help="Task split to include (default: train)."
    )
    args = parser.parse_args()

    # Output files start fresh (we'll append per-domain below)
    for out in [args.sft_output, args.sft_baseline_output,
                args.sft_bon_output, args.dpo_output, args.bon_dpo_output]:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text("")  # truncate

    totals = {"pd": 0, "baseline": 0, "bon": 0, "dpo": 0, "bon_dpo": 0}

    for domain in args.domains:
        domain_dir = Path(args.raw_dir) / domain
        if not domain_dir.exists():
            print(f"[{domain}] Directory not found, skipping: {domain_dir}")
            continue

        # Each domain gets its own allowed task IDs → no cross-domain ID collision
        allowed_ids = _get_split_ids(args.split, [domain])
        n_ids = len(allowed_ids) if allowed_ids else "all"
        print(f"\n[{domain}] split='{args.split}', {n_ids} task IDs allowed")

        import tempfile, os
        tmp_sft     = Path(tempfile.mktemp(suffix=".jsonl"))
        tmp_bl      = Path(tempfile.mktemp(suffix=".jsonl"))
        tmp_bon     = Path(tempfile.mktemp(suffix=".jsonl"))
        tmp_dpo     = Path(tempfile.mktemp(suffix=".jsonl"))
        tmp_bon_dpo = Path(tempfile.mktemp(suffix=".jsonl"))

        n_sft     = build_sft_dataset(domain_dir, tmp_sft, "pd",       allowed_ids)
        n_bl      = build_sft_dataset(domain_dir, tmp_bl,  "baseline", allowed_ids)
        n_bon     = build_sft_dataset(domain_dir, tmp_bon, "bon",      allowed_ids)
        n_dpo     = build_dpo_dataset(domain_dir, tmp_dpo, args.min_score_gap, allowed_ids)
        n_bon_dpo = build_bon_dpo_dataset(domain_dir, tmp_bon_dpo, allowed_ids)

        _append_jsonl(tmp_sft,     Path(args.sft_output))
        _append_jsonl(tmp_bl,      Path(args.sft_baseline_output))
        _append_jsonl(tmp_bon,     Path(args.sft_bon_output))
        _append_jsonl(tmp_dpo,     Path(args.dpo_output))
        _append_jsonl(tmp_bon_dpo, Path(args.bon_dpo_output))

        for p in [tmp_sft, tmp_bl, tmp_bon, tmp_dpo, tmp_bon_dpo]:
            p.unlink(missing_ok=True)

        totals["pd"]      += n_sft
        totals["baseline"] += n_bl
        totals["bon"]     += n_bon
        totals["dpo"]     += n_dpo
        totals["bon_dpo"] += n_bon_dpo

    print(f"\n=== Combined summary (domains={args.domains}, split='{args.split}') ===")
    print(f"  SFT (PD):           {totals['pd']} samples  → {args.sft_output}")
    print(f"  SFT (baseline):     {totals['baseline']} samples  → {args.sft_baseline_output}")
    print(f"  SFT (BoN):          {totals['bon']} samples  → {args.sft_bon_output}")
    print(f"  DPO pairs (PD):     {totals['dpo']} pairs    → {args.dpo_output}")
    print(f"  DPO pairs (BoN ep): {totals['bon_dpo']} pairs    → {args.bon_dpo_output}")


if __name__ == "__main__":
    main()
