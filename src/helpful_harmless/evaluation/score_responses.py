import re
import torch
import argparse
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating SharedRep-RLHF...")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--proportion", type=float, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--method", type=str, required=True, choices=["maxmin", "sharedrep"])
    return parser.parse_args()

if __name__ == "__main__":

    accelerator = Accelerator()
    device = accelerator.device

    # Arguments
    args = parse_args()
    k = args.k
    seed = args.seed
    method = args.method
    user_id = args.user_id
    proportion = args.proportion

    # Dataset
    batch_size = 32
    if method=="gold":
        dataset_name = f"{user_id}/{method}-hh-grpo-eval-seed{seed}"
    elif method == "maxmin":
        dataset_name = f"{user_id}/{method}-hh-grpo-eval-prop{proportion}-seed{seed}"
    elif method == "sharedrep":
        dataset_name = f"{user_id}/{method}-hh-grpo-eval-prop{proportion}-seed{seed}-k{k}"

    # Reward Tokenizer
    rm_tokenizer = AutoTokenizer.from_pretrained('Ray2333/gpt2-large-harmless-reward_model')
    rm_tokenizer.pad_token = rm_tokenizer.eos_token

    # Reward Model - Harmless
    reward_model_harmless = AutoModelForSequenceClassification.from_pretrained(
        'Ray2333/gpt2-large-harmless-reward_model',
        num_labels=1,  
    )
    reward_model_harmless.eval()
    reward_model_harmless.config.pad_token_id = rm_tokenizer.pad_token_id

    # Reward Model - Helpful
    reward_model_helpful = AutoModelForSequenceClassification.from_pretrained(
        'Ray2333/gpt2-large-helpful-reward_model',
        num_labels=1,
    )
    reward_model_helpful.eval()
    reward_model_helpful.config.pad_token_id = rm_tokenizer.pad_token_id


    def tokenize(example):
        return rm_tokenizer(
            example["completion"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
    
    with accelerator.main_process_first():
        dataset = load_dataset(dataset_name)
        dataset = dataset.map(tokenize, desc="Tokenizing", num_proc=32, batched=True, batch_size=batch_size)
        dataset.set_format(type="torch")

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare model and dataloader with accelerator
    reward_model_harmless, reward_model_helpful, dataloader = accelerator.prepare(reward_model_harmless, reward_model_helpful, dataloader)

    harmless_scores = []
    helpful_scores = []
    for batch in tqdm(dataloader, desc="Scoring"):
        with torch.no_grad():

            # Helpful Score
            outputs = accelerator.unwrap_model(reward_model_helpful)(
                input_ids=batch[f"input_ids"].squeeze(),
                attention_mask=batch[f"attention_mask"].squeeze(),
            )
            helpful_score = outputs.logits.squeeze().tolist()
            all_helpful_scores = accelerator.gather_for_metrics(helpful_score)
            helpful_scores.extend(all_helpful_scores)

            # Harmless Score
            outputs = accelerator.unwrap_model(reward_model_harmless)(
                input_ids=batch[f"input_ids"].squeeze(),
                attention_mask=batch[f"attention_mask"].squeeze(),
            )
            harmless_score = outputs.logits.squeeze().tolist()
            all_harmless_scores = accelerator.gather_for_metrics(harmless_score)
            harmless_scores.extend(all_harmless_scores)

    if accelerator.is_main_process:
        print("Pushing scores to hub...")
        
        # Add scores to dataset
        dataset = dataset.remove_columns(["input_ids", "attention_mask"])
        dataset = dataset.add_column("harmless_score", harmless_scores)
        dataset = dataset.add_column("helpful_score", helpful_scores)

        # Push to hub
        dataset.push_to_hub(
            dataset_name,
            private=True,
        )
