import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import torch
import argparse
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.helper.gsm8k_utils import format_prompt

def tokenize(example, tokenizer, max_new_tokens=256):  
        return tokenizer(
            example["formatted_prompts"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length= 2048 - max_new_tokens - 1,
            padding_side="left"
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating SharedRep-RLHF...")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--proportion", type=float, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["gold", "maxmin", "sharedrep"])
    return parser.parse_args()


if __name__ == "__main__":

    accelerator = Accelerator()

     # Arguments
    args = parse_args()
    k = args.k
    seed = args.seed
    method = args.method
    user_id = args.user_id
    proportion = args.proportion

    # Dataset
    batch_size = 32
    num_responses = 1
    max_new_tokens = 256
    dataset = load_dataset("openai/gsm8k", "main")["test"]

    # Policy Model
    if method=="gold":
        policy_model_id = f"{user_id}/{method}-gsm8k-grpo-seed{seed}"
    elif method == "maxmin":
        policy_model_id = f"{user_id}/{method}-gsm8k-grpo-prop{proportion}-seed{seed}"
    elif method == "sharedrep":
        policy_model_id = f"{user_id}/{method}-gsm8k-grpo-prop{proportion}-seed{seed}-k{k}"
    policy_model = AutoModelForCausalLM.from_pretrained(policy_model_id)
    policy_model.eval()
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_id)
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_model.config.pad_token_id = policy_tokenizer.pad_token_id
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare dataset
    with accelerator.main_process_first():
        dataset = dataset.map(format_prompt, num_proc=32, desc="Formatting prompts")
        dataset = dataset.map(lambda x: tokenize(x, tokenizer, max_new_tokens), desc="Tokenizing", num_proc=32)
        generation_dataset = {
            "completion": [],
            "prompt": dataset["question"],
            "answer": dataset["answer"]
        }
        dataset.set_format(type="torch")

        columns_to_keep = ["input_ids", "attention_mask"]
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        dataset = dataset.remove_columns(columns_to_remove)

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare model and dataloader with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    for batch in tqdm(dataloader, desc="Generating"):
        with torch.no_grad():
            # Generate responses
            outputs = accelerator.unwrap_model(model).generate(
                input_ids=batch["input_ids"].squeeze(),
                attention_mask=batch["attention_mask"].squeeze(),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_completions = accelerator.gather_for_metrics(completions)
            generation_dataset[f"completion"].extend(all_completions)
    
    if accelerator.is_main_process:
        print("Pushing evaluation dataset to the hub...")
        if method=="gold":
            dataset_hub_id = f"{user_id}/{method}-gsm8k-grpo-eval-seed{seed}"
        elif method == "maxmin":
            dataset_hub_id = f"{user_id}/{method}-gsm8k-grpo-eval-prop{proportion}-seed{seed}"
        elif method == "sharedrep":
            dataset_hub_id = f"{user_id}/{method}-gsm8k-grpo-eval-prop{proportion}-seed{seed}-k{k}"
        Dataset.from_dict(eval_dataset).push_to_hub(
            dataset_hub_id,
            private=True
        )