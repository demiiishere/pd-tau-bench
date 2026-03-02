"""
DPO training script (Phase B — run on GPU machine, after SFT).

Run:
    python -m src.training.dpo_train \
        --sft-model outputs/models/sft_pd/final \
        --dataset data/dpo_dataset/train.jsonl \
        --output outputs/models/dpo_pd
"""

import argparse


def train_dpo(sft_model_path: str, dpo_dataset_path: str, output_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
    from peft import LoraConfig
    from datasets import load_dataset

    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype="bfloat16",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)

    ref_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype="bfloat16",
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("json", data_files=dpo_dataset_path, split="train")

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        beta=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=8192,
        max_prompt_length=4096,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
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
    parser.add_argument("--sft-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train_dpo(args.sft_model, args.dataset, args.output)


if __name__ == "__main__":
    main()
