import re
import torch
import argparse
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator

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
    parts = re.split(r"(Human:|Assistant:)", example["chosen"])
    parts = [p.strip() for p in parts if p.strip()]
    messages = []
    for i in range(0, len(parts) - 1, 2):
        role = "user" if parts[i] == "Human:" else "assistant"
        messages.append({"role": role, "content": parts[i + 1]})
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    return {"messages": messages}

def convert_messages_to_text(messages):
    formatted_prompts = "\n".join(
        f"{'Human' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
    )
    return formatted_prompts

def tokenize(example, policy_tokenizer, max_new_tokens=256, apply_chat_template=True):
        if apply_chat_template:
            chat_prompt = policy_tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        else:
            chat_prompt = convert_messages_to_text(example["messages"])
        
        policy_inputs = policy_tokenizer(
            chat_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2048 - max_new_tokens - 1,
            padding_side="left"
        )
        return {
            "prompt": convert_messages_to_text(example["messages"]),
            "formatted_prompt": chat_prompt,
            "input_ids": policy_inputs["input_ids"],
            "attention_mask": policy_inputs["attention_mask"],
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating SharedRep-RLHF...")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--proportion", type=float, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["gold", "maxmin", "sharedrep"])
    parser.add_argument("--disable_chat_template", action="store_true", help="Whether to disable the chat template.")
    return parser.parse_args()

if __name__ == "__main__":

    # Accelerator
    accelerator = Accelerator()

    # Arguments
    args = parse_args()
    k = args.k
    seed = args.seed
    method = args.method
    user_id = args.user_id
    proportion = args.proportion
    apply_chat_template = not args.disable_chat_template
    
    # Dataset
    batch_size = 32
    max_new_tokens = 256
    num_eval_samples = 2000 
    dataset = load_dataset("Anthropic/hh-rlhf", split="test").shuffle(seed=seed).select(range(num_eval_samples))

    # Policy Model
    if method=="gold":
        # TODO: Remove this comment --> Use the other line if you want to test
        policy_model_id = f"{user_id}/{method}-hh-grpo-seed{seed}"
        #policy_model_id = f"mukhea5/HH-GRPO-gold-seed{seed}" # only for test
    elif method == "maxmin":
        policy_model_id = f"{user_id}/{method}-hh-grpo-prop{proportion}-seed{seed}"
    elif method == "sharedrep":
        policy_model_id = f"{user_id}/{method}-hh-grpo-prop{proportion}-seed{seed}-k{k}"
    policy_model = AutoModelForCausalLM.from_pretrained(policy_model_id)
    policy_model.eval()
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_id)
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.chat_template = CHAT_TEMPLATE
    policy_model.config.pad_token_id = policy_tokenizer.pad_token_id

    # Prepare dataset
    with accelerator.main_process_first():
        dataset = dataset.map(prepare_for_chat, num_proc=32, desc="Extracting structured prompts")
        dataset = dataset.map(lambda x: tokenize(x, policy_tokenizer, max_new_tokens, apply_chat_template=apply_chat_template), desc="Tokenizing", num_proc=32)
        eval_dataset = {
            "prompt": dataset["prompt"],
            "formatted_prompt": dataset["formatted_prompt"],
            "completion": []
        }
        dataset.set_format(type="torch")

        columns_to_keep = ["input_ids", "attention_mask"]
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        dataset = dataset.remove_columns(columns_to_remove)
        accelerator.print(dataset)

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare model and dataloader with accelerator
    policy_model, dataloader = accelerator.prepare(policy_model, dataloader)

    
    for batch in tqdm(dataloader, desc="Generating"):
        with torch.no_grad():
            # Generate responses
            outputs = accelerator.unwrap_model(policy_model).generate(
                input_ids=batch["input_ids"].squeeze(),
                attention_mask=batch["attention_mask"].squeeze(),
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=policy_tokenizer.pad_token_id
            )
            completions = policy_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_completions = accelerator.gather_for_metrics(completions)
            eval_dataset["completion"].extend(all_completions)

    if accelerator.is_main_process:
        print("Pushing evaluation dataset to the hub...")
        if method=="gold":
            dataset_hub_id = f"{user_id}/{method}-hh-grpo-eval-seed{seed}"
        elif method == "maxmin":
            dataset_hub_id = f"{user_id}/{method}-hh-grpo-eval-prop{proportion}-seed{seed}"
        elif method == "sharedrep":
            dataset_hub_id = f"{user_id}/{method}-hh-grpo-eval-prop{proportion}-seed{seed}-k{k}"
        Dataset.from_dict(eval_dataset).push_to_hub(
            dataset_hub_id,
            private=True
        )