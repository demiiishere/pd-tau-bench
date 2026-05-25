"""
SFT training script for on-policy data (Qwen3-8B generated with thinking ON).

Key differences from sft_train.py:
  1. enable_thinking=False removed from apply_chat_template — training data
     contains <think>...</think> blocks so they must NOT be stripped.
  2. max_length raised to 16384 — thinking chains make trajectories longer.
  3. Default model path points to local /user/zhujiatong/models/Qwen3-8B.

Run:
    # E3-v2: SFT on on-policy PD trajectories
    python -m src.training.sft_train_onpolicy \
        --model /user/zhujiatong/models/Qwen3-8B \
        --dataset data/sft_dataset_onpolicy/train.jsonl \
        --output /user/zhujiatong/outputs_pd/sft_pd_onpolicy

    # E2-v2: SFT on on-policy BoN trajectories
    python -m src.training.sft_train_onpolicy \
        --model /user/zhujiatong/models/Qwen3-8B \
        --dataset data/sft_dataset_onpolicy/train_bon.jsonl \
        --output /user/zhujiatong/outputs_pd/sft_bon_onpolicy
"""

import argparse
import os


def train_sft(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    source_filter: str | None = None,
    eval_fraction: float = 0.05,
    general_data_path: str | None = None,
    max_steps: int = -1,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig
    from datasets import load_dataset

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    if source_filter:
        dataset = dataset.filter(lambda x: x["source"] == source_filter)
    if len(dataset) == 0:
        raise ValueError(f"Empty dataset after filtering source='{source_filter}'")
    print(f"Loaded {len(dataset)} examples (source_filter={source_filter!r})")

    if general_data_path:
        general_ds = load_dataset("json", data_files=general_data_path, split="train")
        if "messages" not in general_ds.column_names:
            raise ValueError(f"General data must have a 'messages' column: {general_data_path}")
        general_ds = general_ds.map(lambda x: {"task_id": "", "source": "general"})
        from datasets import concatenate_datasets
        dataset = concatenate_datasets([dataset, general_ds])
        print(f"Mixed in {len(general_ds)} general examples → total {len(dataset)}")

    # NOTE: enable_thinking is NOT passed here.
    # On-policy trajectories were generated with thinking ON, so assistant turns
    # already contain <think>...</think> blocks. apply_chat_template with
    # add_generation_prompt=False renders them as-is — no suppression needed.
    def apply_template(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(
        apply_template,
        batched=True,
        remove_columns=["messages", "task_id", "source"],
    )

    if eval_fraction > 0 and len(dataset) >= 20:
        split = dataset.train_test_split(test_size=eval_fraction, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset) if eval_dataset else 0}")

    # ── Training ──────────────────────────────────────────────────────────────
    # max_length raised to 16384: thinking chains extend assistant turn length.
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=16384,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"Model saved to {output_dir}/final")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/user/zhujiatong/models/Qwen3-8B")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source", default=None, choices=["pd", "bon", "baseline"],
        help="Filter by source field (default: use all)",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--general-data", default=None)
    parser.add_argument(
        "--max-steps", type=int, default=-1,
        help="Max training steps (-1 = full epoch). Use 2-5 for smoke test.",
    )
    args = parser.parse_args()
    train_sft(
        args.model, args.dataset, args.output,
        args.source, args.eval_fraction, args.general_data, args.max_steps,
    )


if __name__ == "__main__":
    main()
