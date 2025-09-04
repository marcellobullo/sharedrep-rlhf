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
    parser = argparse.ArgumentParser(description="Train Pluralistic Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":

    accelerator = Accelerator()

    # Arguments
    args = parse_args()
    user_id = args.user_id

    # Dataset
    batch_size = 32
    num_responses = 2
    max_new_tokens = 256
    dataset = load_dataset("openai/gsm8k", "main")["train"]

    # Model
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare dataset
    with accelerator.main_process_first():
        dataset = dataset.map(format_prompt, num_proc=32, desc="Formatting prompts")
        dataset = dataset.map(lambda x: tokenize(x, tokenizer, max_new_tokens), desc="Tokenizing", num_proc=32)
        generation_dataset = {
            f"response_{i}": [] for i in range(num_responses)
        }
        generation_dataset["prompt"] = dataset["question"]
        generation_dataset["answer"] = dataset["answer"]
        dataset.set_format(type="torch")

        columns_to_keep = ["input_ids", "attention_mask"]
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        dataset = dataset.remove_columns(columns_to_remove)

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare model and dataloader with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    for batch in tqdm(dataloader, desc="Generating"):
        for i in range(num_responses):
            with torch.no_grad():
                # Generate responses
                outputs = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"].squeeze(),
                    attention_mask=batch["attention_mask"].squeeze(),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    #num_return_sequences=num_responses
                )
                completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_completions = accelerator.gather_for_metrics(completions)
                generation_dataset[f"response_{i}"].extend(all_completions)
    
    if accelerator.is_main_process:
        print("Pushing completions to hub...")
        dataset_hub_id = f"{user_id}/qwen2.5-math-1.5b-gsm8k-raw"
        Dataset.from_dict(generation_dataset).push_to_hub(
            dataset_hub_id,
            private=True
        )