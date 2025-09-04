from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset, Dataset
from accelerate import PartialState
import wandb
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pluralistic Reward Model with configurable parameters.")

    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    parser.add_argument("--minority_proportion", required=True, type=float, help="Proportion of minority group.")
    parser.add_argument("--group", required=True, type=str, help="Group ID.")

    return parser.parse_args()

def format_pref_dataset(dataset):
    """Converts preference data into RewardTrainer-compatible format."""
    chosen_responses = []
    rejected_responses = []
    for ex in dataset:
        if ex["label"] == 0:
            chosen_responses.append(ex["response_0"])
            rejected_responses.append(ex["response_1"])
        else:
            chosen_responses.append(ex["response_1"])
            rejected_responses.append(ex["response_0"])
    return Dataset.from_dict({
        "prompt": dataset["prompt"],
        "chosen": chosen_responses,
        "rejected": rejected_responses
    })


def train_reward_model(
    model_name: str,
    dataset_id: str,
    proportion: float, 
    seed: int, 
    group_id: str,
):
    # Load and filter dataset
    full_dataset = load_dataset(dataset_id, split="train")
    group_dataset = full_dataset.filter(lambda ex: ex["group_id"] == group_id)

    # Format preference data
    reward_dataset = format_pref_dataset(group_dataset)
    reward_dataset = reward_dataset.train_test_split(test_size=0.05, seed=seed)
    train_dataset = reward_dataset["train"]
    eval_dataset = reward_dataset["test"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Freeze Backbone
    for param in model.transformer.parameters():
        param.requires_grad = False

    # print("\n\n\nTrainable parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.requires_grad)
    # print("\n\n\n")

    # Output dir
    output_dir = f"./gpt2_gsm8k_{group_id}_prop{proportion}_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    # Training config
    training_args = RewardConfig(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="no",
        save_total_limit=1,
        report_to=["wandb"],
        run_name=f"gpt2-socratic-{group_id}-prop{proportion}_seed{seed}",
        push_to_hub=True,
        hub_model_id=f"mukhea5/gpt2-socratic-reward-{group_id}-prop{proportion}-seed{seed}",
        hub_private_repo=True,
        ddp_find_unused_parameters=False,
    )

    # Reward trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Evaluation
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    # Save the model
    trainer.push_to_hub(dataset_name=dataset_id)


if __name__ == "__main__":

    state = PartialState()
    args = parse_args()

    proportion = args.minority_proportion
    seed = args.seed
    group_id = args.group

    dataset_id = f"mukhea5/gsm8k_qwen-qwen2.5-math-1.5b_proportion_{proportion}_seed_{seed}_socratic"
    
    train_reward_model(
        proportion=proportion,
        seed=seed,
        group_id=group_id,
        model_name="gpt2-large",
        dataset_id=dataset_id,
    )