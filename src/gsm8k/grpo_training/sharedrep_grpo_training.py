import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import re
import argparse
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from src.models.sr_gpt2.sr_gpt2_modeling import SharedRepGPT2RM
from src.helper.gsm8k_utils import format_prompt

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pluralistic Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    parser.add_argument("--proportion", required=True, type=float, help="Proportion of minority group.")
    parser.add_argument("--k", type=int, required=True, help="Inner hidden dimension size.")
    return parser.parse_args()

if __name__ == "__main__":


    accelerator = Accelerator()
    device = accelerator.device

    args = parse_args()
    user_id = args.user_id
    seed = args.seed
    proportion = args.proportion
    k = args.k
    

    # Dataset
    batch_size = 16
    max_completion_length = 256
    dataset_name = "openai/gsm8k"
    dataset = load_dataset(dataset_name, 'main')["train"]
    with accelerator.main_process_first():
        dataset = dataset.map(format_prompt)
        dataset = dataset.rename_column("formatted_prompts", "prompt")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != "prompt"])
        dataset = dataset.train_test_split(test_size=0.05, seed=seed)

    # Reward Model
    reward_model_name = f"{user_id}/sharedrep-gsm8k-reward-prop{proportion}-seed{seed}-k{k}"
    reward_model = SharedRepGPT2RM.from_pretrained(reward_model_name)
    reward_model.enable_maxmin()
    
    # Reward Tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    reward_model.eval()

    # Policy Model
    policy_model_name = "Qwen/Qwen2.5-Math-1.5B"
    policy_hub_id = f"{user_id}/sharedrep-gsm8k-grpo-prop{proportion}-seed{seed}-k{k}"
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name, padding_side="left")
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

    # Training Arguments
    training_args = GRPOConfig(
        logging_steps=10,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_completion_length=max_completion_length,
        num_generations=2,
        bf16=True,
        log_completions=True,
        learning_rate= 1e-5,
        push_to_hub=True,
        save_strategy="no",
        hub_private_repo=True,
        hub_model_id=policy_hub_id,
        ddp_find_unused_parameters=False,
        report_to=["wandb"],
        run_name=policy_hub_id,
        beta = 0.05,
    )

    # Train
    trainer = GRPOTrainer(
        model=policy_model_name,
        processing_class=policy_tokenizer,
        reward_funcs=reward_model,
        reward_processing_classes=reward_tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()
    
    # Save the model
    trainer.push_to_hub(dataset_name=dataset_name)