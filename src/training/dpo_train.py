"""
DPO training script (Phase B — RTX 5090 32GB, after SFT).

Uses PEFT with ref_model=None: TRL computes reference logprobs by disabling LoRA
adapters on the same model instance, so no second copy is loaded — fits in 32 GB.

Run:
    python -m src.training.dpo_train \
        --sft-model outputs/models/sft_pd/final \
        --dataset data/dpo_dataset/train.jsonl \
        --output outputs/models/dpo_pd

    # Raise score-gap threshold to keep only high-quality pairs
    python -m src.training.dpo_train \
        --sft-model outputs/models/sft_pd/final \
        --dataset data/dpo_dataset/train.jsonl \
        --output outputs/models/dpo_pd \
        --min-score-gap 0.10
"""

import argparse
import os


def train_dpo(
    sft_model_path: str,
    dpo_dataset_path: str,
    output_dir: str,
    min_score_gap: float = 0.05,
    eval_fraction: float = 0.05,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    from peft import LoraConfig
    from datasets import load_dataset

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ─────────────────────────────────────────────────────────────────
    # ref_model=None: with PEFT, TRL computes reference logprobs by disabling LoRA,
    # so no second model copy is needed — saves ~16 GB VRAM.
    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA (re-initialised for DPO on top of SFT weights) ───────────────────
    # Match SFT LoRA exactly: r=8, q/v only, dropout=0.1.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    # Format from build_dataset.py:
    #   {"prompt": [...msgs], "chosen": {msg_dict}, "rejected": {msg_dict}, "score_gap": float}
    # TRL DPOTrainer expects plain strings: {"prompt": str, "chosen": str, "rejected": str}
    dataset = load_dataset("json", data_files=dpo_dataset_path, split="train")
    dataset = dataset.filter(lambda x: x["score_gap"] >= min_score_gap)
    if len(dataset) == 0:
        raise ValueError(f"Empty DPO dataset after filtering score_gap >= {min_score_gap}")
    print(f"DPO dataset: {len(dataset)} pairs (score_gap >= {min_score_gap})")

    def preprocess_dpo(examples):
        prompts, chosens, rejecteds = [], [], []
        for prompt_msgs, chosen_msg, rejected_msg in zip(
            examples["prompt"], examples["chosen"], examples["rejected"]
        ):
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            chosen_full = tokenizer.apply_chat_template(
                prompt_msgs + [chosen_msg],
                tokenize=False,
                add_generation_prompt=False,
            )
            rejected_full = tokenizer.apply_chat_template(
                prompt_msgs + [rejected_msg],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompts.append(prompt_text)
            chosens.append(chosen_full[len(prompt_text):])
            rejecteds.append(rejected_full[len(prompt_text):])
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    dataset = dataset.map(
        preprocess_dpo,
        batched=True,
        remove_columns=["score_gap"],
    )

    # Train / eval split
    if eval_fraction > 0 and len(dataset) >= 20:
        split = dataset.train_test_split(test_size=eval_fraction, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset) if eval_dataset else 0}")

    # ── Training ──────────────────────────────────────────────────────────────
    # Small dataset regime: lr=1e-6 (very conservative, DPO is sensitive),
    # beta=0.3 (larger KL penalty than default 0.1 — prevents model drifting
    # too far from SFT checkpoint with limited data).
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        beta=0.3,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=8192,
        max_prompt_length=6144,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT handles this — no second model copy loaded
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"Model saved to {output_dir}/final")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--min-score-gap",
        type=float,
        default=0.05,
        help="Minimum score gap to keep a DPO pair (default: 0.05)",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.05,
        help="Fraction of data to use for eval (0 to disable)",
    )
    args = parser.parse_args()
    train_dpo(
        args.sft_model,
        args.dataset,
        args.output,
        args.min_score_gap,
        args.eval_fraction,
    )


if __name__ == "__main__":
    main()
