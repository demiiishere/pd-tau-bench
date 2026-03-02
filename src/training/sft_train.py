"""
SFT training script (Phase B — run on GPU machine).

Run:
    python -m src.training.sft_train \
        --model Qwen/Qwen3-8B \
        --dataset data/sft_dataset/train.jsonl \
        --output outputs/models/sft_pd
"""

import argparse


def train_sft(model_name: str, dataset_path: str, output_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig
    from datasets import load_dataset

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=8192,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(output_dir + "/final")
    print(f"Model saved to {output_dir}/final")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train_sft(args.model, args.dataset, args.output)


if __name__ == "__main__":
    main()
