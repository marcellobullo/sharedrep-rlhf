import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import torch
import argparse
import numpy as np
from peft import PeftConfig
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator, PartialState
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.helper.gsm8k_utils import FEW_SHOT_PREFIX, extract_final_answer, format_prompt, extract_hash_answer, add_stats

def parse_args():
    parser = argparse.ArgumentParser(description="Train GOLD REWARD model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    return parser.parse_args()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prepare_reward_fn(reward_model, reward_tokenizer, user_id):

    # Compute Normalization Stats
    dataset = load_dataset(f"{user_id}/qwen2.5-math-1.5b-gsm8k-raw", split="train")
    dataset = add_stats(dataset, keys=["socratic", "length"])
    mean_len = dataset[0]["mean_length_0"]
    std_len = dataset[0]["std_length_0"]
    mean_socratic = dataset[0]["mean_socratic_0"]
    std_socratic = dataset[0]["std_socratic_0"]

    def reward_fn(prompts, completions, ground_truth, **kwargs):

        state = PartialState()

        state.print("\n\n", ground_truth)

        rewards = []
        for prompt, completion, gt in zip(prompts, completions, ground_truth):
            
            # Correctness
            preds, gt = np.array(extract_final_answer(prompt+completion)), np.array(gt)
            correct = (preds==gt).astype(int)

            # Length normalization
            length = len(completion.split())
            length = (length - mean_len) / std_len if std_len > 0 else 0.0
            conciseness = 1-sigmoid(length)

            state.print("\n\n")
            state.print("-"*20)
            state.print("PROMPT:\n")
            state.print(prompt)

            state.print("\n")
            state.print("-"*20)
            state.print("COMPLETION:\n")
            state.print(completion)
            state.print("\n Pred: ", preds)

            # Socratic score
            with torch.no_grad():
                inputs = reward_tokenizer(" "+completion, return_tensors="pt", padding=True, truncation=True).to(reward_model.device)
                score = reward_model(**inputs).logits[0].item()
            score = (score - mean_socratic) / std_socratic if std_socratic > 0 else 0.0
            socratic_score = sigmoid(score)

            # Compute both group scores
            group_score_majority = 0.2 * correct + 0.8 * conciseness
            group_score_minority = 0.2 * correct + 0.8 * socratic_score

            reward = min(group_score_majority, group_score_minority)
            rewards.append(reward)

        return rewards
    return reward_fn


if __name__ == "__main__":

    accelerator = Accelerator()

    args = parse_args()
    user_id = args.user_id
    seed = args.seed

    # Dataset
    batch_size = 16
    max_completion_length = 256
    dataset_name = "openai/gsm8k"
    dataset = load_dataset(dataset_name, 'main')["train"]
    dataset = dataset.shuffle().select(range(10))
    with accelerator.main_process_first():
        dataset = dataset.map(format_prompt)
        dataset = dataset.rename_column("formatted_prompts", "prompt")
        dataset = dataset.map(lambda x: {"ground_truth": extract_hash_answer(x["answer"])})
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["prompt", "ground_truth"]])
        dataset = dataset.train_test_split(test_size=0.05, seed=seed)

    # Socratic Reward Model
    peft_model_id = f"{user_id}/gpt2-gsm8k-lora"
    config = PeftConfig.from_pretrained(peft_model_id)
    reward_model = AutoModelForSequenceClassification.from_pretrained(peft_model_id, num_labels=1)

    # Reward Tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    # Reward function
    reward_function = prepare_reward_fn(reward_model, reward_tokenizer, user_id)

    # Policy Model
    policy_model_name = "Qwen/Qwen2.5-Math-1.5B"
    policy_hub_id = f"{user_id}/gold-gsm8k-grpo-seed{seed}"
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name, padding_side="left")
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

    # Training Arguments
    training_args = GRPOConfig(
        logging_steps=10,
        num_train_epochs=2,
        per_device_train_batch_size=2,
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
        run_name=policy_hub_id,
        ddp_find_unused_parameters=False,
        report_to=["wandb"],
        beta = 0.05,
    )

    # Train
    trainer = GRPOTrainer(
        model=policy_model_name,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()

    # Save the model
    trainer.push_to_hub(dataset_name=dataset_name)

