# Predictive Decoding for LLM Agent Distillation

## Overview

This repository contains the full implementation for studying **Predictive Decoding (PD)** as a data-generation strategy for distilling large teacher LLM agents into smaller student models, evaluated on the [τ²-bench](https://github.com/sierra-research/tau2-bench) conversational agent benchmark.

**Core question**: Does giving a teacher model "foresight" (simulating K candidates × H future turns before committing to an action) produce better training data for student distillation than simply taking the best of N independent rollouts (Best-of-N)?

**Key finding**: PD's advantage is teacher-capability-gated. With a strong teacher (Qwen-Plus), PD improves per-trial success rate by +5.8–6.4 pp over BoN. With a weaker teacher (Qwen3-32B), PD ≈ BoN. PD's real value lies in generating **turn-level preference pairs for DPO**, not in producing better SFT trajectories. The best student (Qwen3-4B, E3c: BoN-SFT + PD-DPO) achieves 14.5% avg pass@1 vs 2.4% zero-shot — a 6× improvement.

---

## Repository Structure

```
pd-tau-bench/
├── src/
│   ├── predictive_decoding/
│   │   ├── core.py              # PD episode runner (K candidates × H foresight)
│   │   ├── value_function.py    # 5-signal value function for scoring foresight states
│   │   ├── tau_bench_adapter.py # State save/restore for environment forking
│   │   └── test_env_fork.py     # Unit tests for fork/restore correctness
│   ├── data_generation/
│   │   ├── generate_trajectories.py   # PD data generation (teacher = API model)
│   │   ├── generate_bon.py            # Best-of-N data generation
│   │   ├── generate_baseline.py       # Greedy baseline (teacher pass@1)
│   │   ├── generate_trajectories_onpolicy.py  # On-policy PD (local vLLM teacher)
│   │   ├── generate_bon_onpolicy.py           # On-policy BoN
│   │   ├── build_dataset.py           # Build SFT/DPO datasets from raw trajectories
│   │   └── inspect_trajectories.py    # Analysis utilities
│   ├── training/
│   │   ├── sft_train.py         # LoRA SFT training (Qwen3-4B/8B)
│   │   └── dpo_train.py         # LoRA DPO training (on top of SFT checkpoint)
│   └── evaluation/
│       ├── eval_on_tau_bench.py # vLLM-backed evaluation on τ²-bench test split
│       └── analysis.py          # Result aggregation and statistics
├── experiments/
│   ├── lib_sft_eval.sh          # Shared bash functions (server start, eval, merge LoRA)
│   ├── lib_multiseed.sh         # Multi-seed evaluation helpers
│   ├── lib_trainseed.sh         # Multi-seed training helpers
│   ├── run_e1*.sh               # Phase A: Qwen-Plus teacher experiments
│   ├── run_e2*.sh               # On-policy (Qwen3-8B teacher) experiments
│   ├── run_e3*.sh               # Phase B: Qwen3-32B teacher experiments
│   ├── run_zero_shot_4b*.sh     # Zero-shot baselines
│   ├── aggregate_multiseed.py   # Aggregate results across seeds
│   └── compute_ci.py            # Bootstrap confidence intervals
├── configs/
│   ├── generation_config.yaml   # Teacher model, PD hyperparams (K=5, H=2)
│   └── task_splits.json         # Train/test task ID splits per domain
├── scripts/
│   ├── start_vllm.sh            # Start vLLM server for local inference
│   └── serve_model.py           # Model serving utilities
├── data/                        # (gitignored) raw trajectories and built datasets
│   ├── raw_trajectories/        # Qwen-Plus PD/BoN trajectories
│   ├── raw_trajectories_qwen3_32b/  # Qwen3-32B trajectories
│   ├── sft_dataset/             # Built SFT JSONL files
│   ├── dpo_dataset/             # Built DPO preference pair JSONL files
│   └── results/                 # Evaluation results per experiment
├── tau2-bench/                  # τ²-bench submodule
├── THESIS_OUTLINE.md            # Full thesis outline with results and analysis
├── CLAUDE_CONTEXT.md            # Technical reference for development
└── PROGRESS.md                  # Experiment log and status
```

---

## Method

### Predictive Decoding (PD)

At each agent turn, instead of greedily generating one response:

1. **Sample** K=5 candidate responses (temperature=0.8)
2. **Fork** the environment state for each candidate (deepcopy of agent + user + DB state)
3. **Simulate** H=2 foresight turns per fork (greedy, temperature=0)
4. **Score** each fork with a 5-signal value function
5. **Commit** to the candidate with the highest score

```
turn t:  [state] → sample K candidates → fork × K → simulate H turns each
                                                     ↓
                              value_function scores all K × forks
                                                     ↓
                              pick argmax → commit to real trajectory
```

### Value Function (5 signals)

All signals are computed on the **foresight delta** only (not the full trajectory), making them sensitive to differences between candidates.

| Signal | Weight | Description |
|--------|--------|-------------|
| `delta_progress` | 0.35 | New expected tools called during foresight |
| `foresight_health` | 0.25 | Tool error rate and redundancy in foresight |
| `user_sentiment` | 0.15 | Positive/negative signals in user foresight responses |
| `termination` | 0.15 | Clean task completion during foresight |
| `env_assertions` | 0.10 | Partial goal satisfaction (DB state checks) |

### Dataset Construction

From raw PD trajectories, two datasets are built:

- **SFT dataset**: successful trajectories (reward=1) converted to multi-turn chat format
- **DPO dataset**: for each decision step where PD was active (non-greedy fallback), pairs of `(chosen_response, rejected_response)` where `score_gap ≥ 0.05`

```bash
python -m src.data_generation.build_dataset
# outputs: data/sft_dataset/train.jsonl, data/dpo_dataset/train.jsonl
```

### Training Pipeline (2-stage)

```
Qwen3-4B (base)
    │
    ▼  Stage 1: LoRA SFT on successful trajectories
SFT checkpoint  (r=8, q_proj/v_proj, lr=5e-5, 1 epoch)
    │  merge LoRA
    ▼  Stage 2: LoRA DPO on turn-level preference pairs
DPO checkpoint  (r=8, q_proj/v_proj, β=0.3, lr=1e-6, 1 epoch)
    │  merge LoRA
    ▼
Final model → vLLM serving → τ²-bench evaluation
```

---

## Experiments

### Experiment Matrix

| ID | Description | Teacher | SFT Data | DPO Data | Domains |
|----|-------------|---------|----------|----------|---------|
| **E0** | Zero-shot Qwen3-4B | — | — | — | retail/airline/telecom |
| **E1a** | Qwen-Plus PD-SFT | Qwen-Plus | PD 197 | — | retail/airline |
| **E1b** | Qwen-Plus BoN-SFT | Qwen-Plus | BoN 295 | — | retail/airline |
| **E1c** | Qwen-Plus BoN-SFT + DPO | Qwen-Plus | BoN 295 | 313 pairs | retail/airline |
| **E1d** | Qwen-Plus PD-SFT + DPO | Qwen-Plus | PD 197 | 313 pairs | retail/airline |
| **E3a** | 32B PD-SFT | Qwen3-32B | PD 146 | — | retail/airline/telecom |
| **E3b** | 32B BoN-SFT | Qwen3-32B | BoN 240 | — | retail/airline/telecom |
| **E3c** ★ | 32B BoN-SFT + PD-DPO | Qwen3-32B | BoN 240 | 298 pairs | retail/airline/telecom |
| **E3d** | 32B PD-SFT + DPO | Qwen3-32B | PD 146 | 298 pairs | retail/airline/telecom |

★ Best configuration.

### Results (Qwen3-4B student, pass@1)

**Phase A — Qwen-Plus teacher (retail + airline)**

| Experiment | Retail | Airline |
|------------|--------|---------|
| E0 Zero-shot | 2.9% | 6.7% |
| E1a PD-SFT | 2.9% | 20.0% |
| E1b BoN-SFT | 8.8% | 13.3% |
| E1c BoN-SFT + DPO | 0.0% | 6.7% |
| E1d PD-SFT + DPO | 2.9% | 13.3% |

**Phase B — Qwen3-32B teacher (retail + airline + telecom)**

| Experiment | Retail | Airline | Telecom | Avg |
|------------|--------|---------|---------|-----|
| E0 Zero-shot | 2.9% | 6.7% | 0.0% | 2.4% |
| E3a PD-SFT | 2.9% | 20.0% | 0.0% | 4.8% |
| E3b BoN-SFT | 8.8% | 13.3% | 5.9% | 8.4% |
| **E3c BoN-SFT + PD-DPO** | **5.9%** | **20.0%** | **20.6%** | **14.5%** |
| E3d PD-SFT + PD-DPO | 0.0% | 6.7% | 0.0% | 1.2% |

---

## Setup

### Prerequisites

- Python 3.11
- CUDA 12.6+ (A100 40GB recommended for training)
- τ²-bench submodule

```bash
git clone --recurse-submodules <this-repo>
cd pd-tau-bench
pip install -e tau2-bench/
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers peft trl vllm litellm loguru pyyaml
```

### Environment Variables

```bash
export DASHSCOPE_API_KEY="your-key"   # for Qwen-Plus teacher
export PYTHONPATH="/path/to/pd-tau-bench"
```

---

## Reproducing Experiments

### Step 1 — Generate teacher trajectories

```bash
# Qwen-Plus teacher (PD + BoN, retail + airline)
python -m src.data_generation.generate_trajectories   # PD
python -m src.data_generation.generate_bon            # BoN

# Qwen3-32B teacher via local vLLM
bash scripts/start_vllm.sh /path/to/Qwen3-32B 8000
python -m src.data_generation.generate_trajectories \
    --teacher openai/Qwen3-32B --api-base http://localhost:8000/v1 \
    --domains retail airline telecom \
    --output-dir data/raw_trajectories_qwen3_32b
```

### Step 2 — Build SFT / DPO datasets

```bash
# Qwen-Plus data (retail + airline)
python -m src.data_generation.build_dataset \
    --raw-dir data/raw_trajectories \
    --sft-output data/sft_dataset/train.jsonl \
    --dpo-output data/dpo_dataset/train.jsonl

# Qwen3-32B data (retail + airline + telecom)
python -m src.data_generation.build_dataset \
    --raw-dir data/raw_trajectories_qwen3_32b \
    --sft-output data/sft_dataset_32b_bon/train.jsonl \
    --dpo-output data/dpo_dataset_32b/train.jsonl \
    --source bon --domains retail airline telecom
```

### Step 3 — Train student model

```bash
# E3c: BoN-SFT → PD-DPO (best configuration)
bash experiments/run_e3c_32b_bon_dpo.sh

# Or run all Phase B experiments
for exp in e3a e3b e3c e3d; do
    bash experiments/run_${exp}_32b_*.sh
done
```

### Step 4 — Evaluate

```bash
# Start vLLM server with trained model
bash scripts/start_vllm.sh outputs/models/e3c_sft_32b_bon_dpo_dpo/merged 8002

# Run evaluation on test split
python -m src.evaluation.eval_on_tau_bench \
    --domain retail --split test \
    --agent-model openai/finetuned \
    --vllm-url http://localhost:8002/v1 \
    --user-model openai/qwen-plus \
    --num-trials 3 \
    --output-dir data/results/e3c
```

---

## Citation

If you use this code, please cite the underlying benchmark and method:

```bibtex
@misc{tau2bench2025,
  title={τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment},
  author={Sierra Research},
  year={2025},
  eprint={2506.07982},
  archivePrefix={arXiv}
}

@misc{ma2024predictive,
  title={Predictive Decoding},
  author={Ma, ...},
  year={2024},
  eprint={2410.17195},
  archivePrefix={arXiv}
}
```

---

## License

Academic use only. The τ²-bench submodule is subject to its own license.
