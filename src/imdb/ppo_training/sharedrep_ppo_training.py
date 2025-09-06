import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import argparse
from trl import PPOConfig, PPOTrainer
from datasets import load_dataset
from accelerate import Accelerator
from src.helper.imdb_utils import SharedRepPPOTrainer
from src.models.sr_gpt2.sr_gpt2_modeling import SharedRepGPT2RM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification


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
    dataset_name = f"{user_id}/gpt2-imdb-raw"
    dataset = load_dataset(dataset_name, split="train")
    with accelerator.main_process_first():
        dataset = dataset.train_test_split(test_size=0.05, seed=seed)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]


    # Reward Model
    reward_model_name = f"{user_id}/sharedrep-imdb-reward-prop{proportion}-seed{seed}-k{k}"
    reward_model = SharedRepGPT2RM.from_pretrained(reward_model_name)
    reward_model.enable_maxmin()
    
    # Reward Tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    reward_model.eval()

    # Policy Model
    policy_model_name = "lvwerra/gpt2-imdb"
    policy_hub_id = f"{user_id}/sharedrep-imdb-ppo-prop{proportion}-seed{seed}-k{k}"
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name, padding_side="left")
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

    # Policy model
    policy = AutoModelForCausalLM.from_pretrained( policy_model_name, trust_remote_code=True)

    # Ref policy
    ref_policy = AutoModelForCausalLM.from_pretrained(policy_model_name, trust_remote_code=True)

    # Value model
    value_model = AutoModelForSequenceClassification.from_pretrained( policy_model_name, trust_remote_code=True, num_labels=1)

    # Training Arguments
    training_args = PPOConfig(
        logging_steps=10,
        num_train_epochs=5,
        num_ppo_epochs=3,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=1,
        bf16=True,
        learning_rate= 3e-06,
        eval_on_start= True,
        push_to_hub=True,
        save_strategy="no",
        hub_private_repo=True,
        hub_model_id=policy_hub_id,
        report_to=["wandb"],
        run_name=policy_hub_id,
        whiten_rewards= True,
        missing_eos_penalty= 1.0,
        total_episodes = 50000,
        lr_scheduler_type= "cosine",
        stop_token= "eos",
        kl_coef=0.05

    )

    # Train
    trainer = PPOTrainer(
        args=training_args,
        model=policy,
        processing_class=policy_tokenizer,
        ref_model=ref_policy,
        reward_model=reward_model,  
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    # trainer = SharedRepPPOTrainer(
    #     pluralistic_reward_model=reward_model,#.to(device),
    #     args=training_args,
    #     processing_class=policy_tokenizer,
    #     model=policy,
    #     ref_model=ref_policy,
    #     reward_model=reward_model,  # place-holder
    #     value_model=value_model,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     use_tanh=True,
    #     pointwise=True
    # )
    trainer.train()
    
    # Save the model
    trainer.push_to_hub(dataset_name=dataset_name)