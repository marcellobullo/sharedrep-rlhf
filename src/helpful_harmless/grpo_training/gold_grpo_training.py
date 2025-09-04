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
    def __init__(self, helpful_reward_model, harmless_reward_model):
        super().__init__()
        self.helpful_reward_model = helpful_reward_model
        self.harmless_reward_model = harmless_reward_model
        self.config = helpful_reward_model.config

    def forward(self, input_ids, **kwargs):

        # Helpfulness score
        help_outputs = self.helpful_reward_model(input_ids, **kwargs)

        # Minority score
        harm_outputs = self.harmless_reward_model(input_ids, **kwargs)

        # combination
        maj_score = help_outputs.logits
        min_score = 0.3*help_outputs.logits + 0.7*harm_outputs.logits

        rewards = torch.min(maj_score, min_score)
        return SequenceClassifierOutputWithPast(
            loss=None,
            logits=rewards,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Train GOLD REWARD model with configurable parameters.")
    parser.add_argument("--user_id", type=int, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    return parser.parse_args()


if __name__ == "__main__":

    accelerator = Accelerator()

    args = parse_args()
    user_id = args.user_id
    seed = args.seed

    # Dataset
    batch_size = 16
    max_completion_length = 256
    dataset_name = f"{user_id}/tinyllama-hh-raw-10K-test"
    dataset = load_dataset(dataset_name, split="train")
    with accelerator.main_process_first():
        dataset = dataset.map(prepare_for_chat, num_proc=32)
        dataset = dataset.train_test_split(test_size=0.05, seed=seed)

    # Reward Tokenizer
    rm_tokenizer = AutoTokenizer.from_pretrained('Ray2333/gpt2-large-harmless-reward_model')
    rm_tokenizer.chat_template = CHAT_TEMPLATE
    rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_tokenizer.pad_token_id = rm_tokenizer.eos_token_id

    # Reward Model Harmless
    reward_model_harmless = AutoModelForSequenceClassification.from_pretrained(
        'Ray2333/gpt2-large-harmless-reward_model',
        num_labels=1,
    )
    reward_model_harmless.config.pad_token_id = rm_tokenizer.pad_token_id

    # Reward Model Helpful
    reward_model_helpful = AutoModelForSequenceClassification.from_pretrained(
        'Ray2333/gpt2-large-helpful-reward_model',
        num_labels=1, 
    )
    reward_model_helpful.config.pad_token_id = rm_tokenizer.pad_token_id

    # MaxMin Reward Wrapper
    maxmin_reward_model = MaxMinRewardWrapper(
        helpful_reward_model=reward_model_helpful,
        harmless_reward_model=reward_model_harmless
    )

    # Policy Model
    policy_model_name = f"{user_id}/tinyllama-sft-hh"
    policy_hub_id = f"{user_id}/gold-hh-grpo-seed{seed}-test"
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
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
        processing_class=policy_tokenizer,
        reward_funcs=maxmin_reward_model,
        reward_processing_classes=rm_tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()

    # Save the model
    trainer.push_to_hub(dataset_name=dataset_name)