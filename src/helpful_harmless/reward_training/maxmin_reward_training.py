import argparse
from trl import RewardConfig, RewardTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Train Maxmin Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    parser.add_argument("--proportion", type=float, required=True, help="Proportions of helpful and harmless responses.")
    parser.add_argument("--group", required=True, type=int, help="Group ID.")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    user_id = args.user_id
    seed = args.seed
    proportion = args.proportion
    group_id = args.group

    # Dataset
    dataset_name = f"{user_id}/tinyllama-hh-10K-seed{seed}-prop{proportion}"
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.filter(lambda example: example["group_id"] == group_id)
    dataset = dataset.train_test_split(test_size=0.05, seed=seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Reward Params
    num_labels = 1
    model_name = "gpt2-large"
    freeze_backbone = True
    group_name = "majority" if group_id == 1 else "minority"
    model_hub_id = f"{user_id}/maxmin-hh-reward-prop{proportion}-seed{seed}-{group_name}"

    # Reward Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Reward Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Freeze Backbone
    for param in model.transformer.parameters():
        param.requires_grad = False

    # Training config
    training_args = RewardConfig(
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="no",
        report_to=["wandb"],
        run_name=model_hub_id,
        push_to_hub=True,
        hub_model_id=model_hub_id,
        hub_private_repo=True,
        ddp_find_unused_parameters=False,
    )

    # Reward trainer
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    # Evaluation
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    trainer.train()

    trainer.push_to_hub(
        dataset_name=dataset_name,
    )