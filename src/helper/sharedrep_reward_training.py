import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

from trl import RewardConfig
from accelerate import Accelerator
from transformers import AutoTokenizer
from datasets import load_dataset
from src.models.sr_gpt2.sr_gpt2_modeling import SharedRepGPT2RM
from src.helper.sharedrep_reward_trainer import SharedRepRMTrainer, SharedRepRewardDataCollatorWithPadding

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train SharedRep Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    parser.add_argument("--proportion", type=float, required=True, help="Proportions of helpful and harmless responses.")
    parser.add_argument("--k", type=int, required=True, help="Inner hidden dimension size.")
    return parser.parse_args()


if __name__ == "__main__":

    accelerator = Accelerator()
    
    # Arguments
    args = parse_args()
    user_id = args.user_id
    seed = args.seed
    proportion = args.proportion
    k = args.k

    # Dataset
    dataset_num_proc = 8
    dataset_batch_size = 128
    dataset_name = f"{user_id}/gpt2-imdb-seed{seed}-prop{proportion}"
    with accelerator.main_process_first():
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.train_test_split(test_size=0.05, seed=seed)

    # Reward Params
    n_groups=2
    use_tanh = True
    model_name = "lvwerra/gpt2-imdb"
    freeze_backbone = True
    model_hub_id = f"{user_id}/sharedrep-imdb-reward-prop{proportion}-seed{seed}-k{k}"
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token  

    # Reward Model
    model = SharedRepGPT2RM.from_pretrained(model_name, k=k, n_heads=n_groups)
    model.disable_maxmin()
    if freeze_backbone: model.freeze_backbone()
    model.unfreeze_heads()
    model.config.pad_token_id = tokenizer.pad_token_id

    # Create the data collator
    data_collator = SharedRepRewardDataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

        # Training config
    training_args = RewardConfig(
        per_device_train_batch_size=16,
        learning_rate=5e-4,
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="no",
        report_to=["wandb"],
        run_name=model_hub_id,
        push_to_hub=True,
        hub_model_id=model_hub_id,
        hub_private_repo=True,
    )

    # Trainer
    trainer = SharedRepRMTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        use_tanh=use_tanh,
        data_collator=data_collator,
    )
    trainer.train()

    # Evaluation
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    trainer.push_to_hub(
        dataset_name=dataset_name,
    )





