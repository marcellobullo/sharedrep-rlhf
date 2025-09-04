import re
import torch
import argparse
from torch import nn
from datasets import load_dataset
from accelerate import Accelerator
from trl import GRPOConfig, GRPOTrainer
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import AutoTokenizer, AutoModelForSequenceClassification


CHAT_TEMPLATE ="""{% for message in messages %}
    {% if message['role'] == 'user' %}
    <|user|>
    {{ message['content'] }}</s>
    {% elif message['role'] == 'assistant' %}
    <|assistant|>
    {% generation %}
    {{ message['content'] }}</s>
    {% endgeneration %}
    {% endif %}
    {% endfor %}
    {% if messages[-1]['role'] != 'assistant' %}
    <|assistant|>
    {% generation %}
    {% endgeneration %}
    {% endif %}
    """

def prepare_for_chat(example):
    parts = re.split(r"(Human:|Assistant:)", example["prompt"])
    parts = [p.strip() for p in parts if p.strip()]
    messages = []
    for i in range(0, len(parts) - 1, 2):
        role = "user" if parts[i] == "Human:" else "assistant"
        messages.append({"role": role, "content": parts[i + 1]})
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    return {"prompt": messages}

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
    dataset_name = f"{user_id}/tinyllama-hh-raw-10K"
    dataset = load_dataset(dataset_name, split="train")
    with accelerator.main_process_first():
        dataset = dataset.map(prepare_for_chat, num_proc=32)
        dataset = dataset.train_test_split(test_size=0.05, seed=seed)

    # Reward Model Majority
    reward_model_name_majority = f"{user_id}/maxmin-hh-reward-prop{proportion}-seed{seed}-majority"
    reward_model_majority = AutoModelForSequenceClassification.from_pretrained(reward_model_name_majority)
    reward_model_majority.eval()
    fc_maj = reward_model_majority.score
    del reward_model_majority
    
    # Reward Model Minority
    reward_model_name_minority = f"{user_id}/maxmin-hh-reward-prop{proportion}-seed{seed}-minority"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name_minority)
    reward_model.eval()
    fc_min = reward_model.score
    
    # Reward Tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name_majority)
    reward_tokenizer.chat_template = CHAT_TEMPLATE
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
    policy_model_name = f"{user_id}/tinyllama-sft-hh"
    policy_hub_id = f"{user_id}/maxmin-hh-grpo-prop{proportion}-seed{seed}"
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.chat_template = CHAT_TEMPLATE

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


    