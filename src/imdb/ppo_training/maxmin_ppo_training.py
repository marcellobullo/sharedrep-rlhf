import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

from torch import nn
import torch
from datasets import load_dataset
from trl import PPOConfig, PPOTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from accelerate import Accelerator

from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import argparse

class MaxMinRewardWrapper(nn.Module):
    def __init__(self, reward_model, fc_maj, fc_min):
        super().__init__()
        self.reward_model = reward_model
        self.fc_maj = fc_maj
        self.fc_min = fc_min
        self.config = reward_model.config

    def forward(self, input_ids, **kwargs):
        
        # Majority score
        self.reward_model.score = self.fc_maj
        maj_outputs = self.reward_model(input_ids, **kwargs)

        # Minority score
        self.reward_model.score = self.fc_min
        min_outputs = self.reward_model(input_ids, **kwargs)

        rewards = torch.min(min_outputs.logits, maj_outputs.logits)

        return SequenceClassifierOutputWithPast(
            loss=None,
            logits=rewards,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pluralistic Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    parser.add_argument("--proportion", required=True, type=float, help="Proportion of minority group.")
    return parser.parse_args()


if __name__ == "__main__":

    accelerator = Accelerator()
    device = accelerator.device

    args = parse_args()
    user_id = args.user_id
    seed = args.seed
    proportion = args.proportion

    # Dataset
    batch_size = 16
    dataset_name = f"{user_id}/gpt2-imdb-raw"
    dataset = load_dataset(dataset_name, split="train")
    with accelerator.main_process_first():
        dataset = dataset.train_test_split(test_size=0.05, seed=seed)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]


    # Reward Model Majority
    reward_model_name_majority = f"{user_id}/maxmin-imdb-reward-prop{proportion}-seed{seed}-majority"
    reward_model_majority = AutoModelForSequenceClassification.from_pretrained(reward_model_name_majority)
    reward_model_majority.eval()
    fc_maj = reward_model_majority.score
    del reward_model_majority
    
    # Reward Model Minority
    reward_model_name_minority = f"{user_id}/maxmin-imdb-reward-prop{proportion}-seed{seed}-minority"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name_minority)
    reward_model.eval()
    fc_min = reward_model.score
    
    # Reward Tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name_majority)
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    # MaxMin Reward Wrapper
    fc_maj.to(device)
    fc_min.to(device)
    reward_model.to(device)
    maxmin_reward_model = MaxMinRewardWrapper(
        reward_model=reward_model,
        fc_maj=fc_maj,
        fc_min=fc_min
    )
    maxmin_reward_model.eval()

    # Policy Model
    policy_model_name = "lvwerra/gpt2-imdb"
    policy_hub_id = f"{user_id}/maxmin-imdb-ppo-prop{proportion}-seed{seed}"
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
        reward_model=maxmin_reward_model,  
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    # Save the model
    trainer.push_to_hub(dataset_name=dataset_name)