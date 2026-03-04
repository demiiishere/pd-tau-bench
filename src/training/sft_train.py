"""
SFT training script (Phase B — RTX 5090 32GB, no quantization).

Qwen3-8B bf16 (~16 GB weights) + LoRA params/optimizer (~2 GB) leaves ~14 GB
for activations — no quantization needed on 32 GB VRAM.
gradient_checkpointing is still enabled to safely handle 8192-token sequences.

Run:
    # Train on PD trajectories (E3)
    python -m src.training.sft_train \
        --model Qwen/Qwen3-8B \
        --dataset data/sft_dataset/train.jsonl \
        --output outputs/models/sft_pd

    # Train on BoN trajectories (E2)
    python -m src.training.sft_train \
        --model Qwen/Qwen3-8B \
        --dataset data/sft_dataset/train_bon.jsonl \
        --output outputs/models/sft_bon
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

    # ── Model (bf16, no quantization — 32 GB is sufficient for Qwen3-8B) ──────
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required when gradient_checkpointing=True

    # ── LoRA ──────────────────────────────────────────────────────────────────
    # Small dataset regime: r=8 (fewer trainable params = less overfitting),
    # only q_proj/v_proj (skip k/o to further reduce params), higher dropout.
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
    print(f"Loaded {len(dataset)} domain examples (source_filter={source_filter!r})")

    # Optional: mix in general function-calling data for regularization.
    # Recommended: ~3000 samples from glaive-function-calling-v2.
    # This prevents overfitting to τ-bench idioms and stabilises training.
    # All experiments (E1/E2/E3) must use the SAME general_data_path so the
    # only variable is the ~200 domain-specific samples.
    if general_data_path:
        general_ds = load_dataset("json", data_files=general_data_path, split="train")
        # Ensure 'messages' column exists; drop other columns to align schema
        if "messages" not in general_ds.column_names:
            raise ValueError(f"General data must have a 'messages' column: {general_data_path}")
        # Add dummy task_id/source so apply_template can drop them uniformly
        general_ds = general_ds.map(lambda x: {"task_id": "", "source": "general"})
        from datasets import concatenate_datasets
        dataset = concatenate_datasets([dataset, general_ds])
        print(f"Mixed in {len(general_ds)} general examples → total {len(dataset)}")

    # Apply Qwen3 chat template to the 'messages' column → produce a 'text' column.
    # Qwen3's built-in template handles system/user/assistant/tool messages
    # including tool_calls in OpenAI format.
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
    # Small dataset regime: 1 epoch (multiple epochs = memorisation),
    # slightly higher lr (5e-5) to get signal from limited data,
    # smaller effective batch (4 vs 8) for more gradient updates.
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
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
        max_length=8192,
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
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source",
        default=None,
        choices=["pd", "bon", "baseline"],
        help="Filter examples by source field (default: use all)",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.05,
        help="Fraction of data to use for eval (0 to disable)",
    )
    parser.add_argument(
        "--general-data",
        default=None,
        help="Path to general function-calling JSONL (e.g. glaive-function-calling-v2). "
             "If provided, mixed in for regularisation. Must be the SAME across E1/E2/E3.",
    )
    args = parser.parse_args()
    train_sft(args.model, args.dataset, args.output, args.source, args.eval_fraction,
              args.general_data)


if __name__ == "__main__":
    main()
