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
    dataset_name = f"{user_id}/tinyllama-hh-raw-10K"
    dataset = load_dataset(dataset_name, split="train")
    with accelerator.main_process_first():
        dataset = dataset.map(prepare_for_chat, num_proc=32)
        dataset = dataset.train_test_split(test_size=0.05, seed=seed)


    # Reward Model
    reward_model_name = f"{user_id}/sharedrep-hh-reward-prop{proportion}-seed{seed}-k{k}"
    reward_model = SharedRepGPT2RM.from_pretrained(reward_model_name)
    reward_model.enable_maxmin()
    
    # Reward Tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    reward_tokenizer.chat_template = CHAT_TEMPLATE
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    reward_model.eval()

    # Policy Model
    policy_model_name = f"{user_id}/tinyllama-sft-hh"
    policy_hub_id = f"{user_id}/sharedrep-hh-grpo-prop{proportion}-seed{seed}-k{k}"
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

