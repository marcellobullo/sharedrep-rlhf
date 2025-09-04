import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from torch import nn
import json
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from accelerate import Accelerator, PartialState

from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import argparse

FEW_SHOT_PREFIX = """These are examples of how to solve math problems step by step:

    Q: If there are 3 apples and you eat one, how many are left?
    A: Let's think step by step. There are 3 apples. You eat one. So 3 - 1 = 2.
    #### The final answer is 2.

    Q: Tom had 4 pencils. He gave 1 to Sarah and bought 3 more. How many does he have now?
    A: Let's think step by step. He had 4 and gave away 1, so 4 - 1 = 3. Then he bought 3, so 3 + 3 = 6.
    #### The final answer is 6.

    The following is the only question you need to answer:
    """

def format_prompt(example: str) -> str:
        return {
            "formatted_prompts": FEW_SHOT_PREFIX + f"\nQ: {example['question']}. Let's think step by step and provide the final answer prefixed by ####.\nA: "
        }

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
    parser.add_argument("--user_id", type=int, required=True, help="Hugging Face user ID.")
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
    max_completion_length = 256 
    dataset_name = "openai/gsm8k"
    dataset = load_dataset(dataset_name, 'main')["train"]
    with accelerator.main_process_first():
        dataset = dataset.map(format_prompt)
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != "prompt"])
        dataset = dataset.train_test_split(test_size=0.05, seed=seed)


    # Reward Model Majority
    reward_model_name_majority = f"{user_id}/maxmin-gsm8k-reward-prop{proportion}-seed{seed}-majority"
    reward_model_majority = AutoModelForSequenceClassification.from_pretrained(reward_model_name_majority)
    reward_model_majority.eval()
    fc_maj = reward_model_majority.score
    del reward_model_majority
    
    # Reward Model Minority
    reward_model_name_minority = f"{user_id}/maxmin-gsm8k-reward-prop{proportion}-seed{seed}-minority"
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
    policy_model_name = "Qwen/Qwen2.5-Math-1.5B"
    policy_hub_id = f"{user_id}/maxmin-gsm8k-grpo-prop{proportion}-seed{seed}"
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
        run_name=policy_hub_id,
        ddp_find_unused_parameters=False,
        report_to=["wandb"],
        beta = 0.05,
    )

    # Train
    trainer = GRPOTrainer(
        model=policy_model_name,
        reward_funcs=maxmin_reward_model,
        reward_processing_classes=reward_tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()

    # Save the model
    trainer.push_to_hub(dataset_name=dataset_name)