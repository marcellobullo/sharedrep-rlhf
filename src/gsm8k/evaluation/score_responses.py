import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import torch
import argparse
from tqdm import tqdm
from peft import PeftConfig
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.helper.gsm8k_utils import compute_correctness_score, compute_length_score


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
    if method=="gold":
        dataset_name = f"{user_id}/{method}-gsm8k-grpo-eval-seed{seed}"
    elif method == "maxmin":
        dataset_name = f"{user_id}/{method}-gsm8k-grpo-eval-prop{proportion}-seed{seed}"
    elif method == "sharedrep":
        dataset_name = f"{user_id}/{method}-gsm8k-grpo-eval-prop{proportion}-seed{seed}-k{k}"

    # Socratic Reward Model
    peft_model_id = f"{user_id}/gpt2-gsm8k-lora"
    config = PeftConfig.from_pretrained(peft_model_id)
    reward_model = AutoModelForSequenceClassification.from_pretrained(peft_model_id, num_labels=1)

    # Reward Tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    def tokenize(example):
        return reward_tokenizer(
            example["completion"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2048
        )

    with accelerator.main_process_first():
        
        # Scoring
        accelerator.print("Scoring...")
        dataset = dataset.map(lambda x: compute_length_score(x, key="completion", num_responses=num_responses), num_proc=32, desc="Computing length scores")
        dataset = dataset.map(lambda x: compute_correctness_score(x, key="completion", num_responses=num_responses), num_proc=32, desc="Computing correctness scores")

        # Tokenization
        accelerator.print("Tokenizing dataset...")
        dataset = dataset.map(tokenize, desc="Tokenizing", num_proc=32)
        dataset.set_format(type="torch")

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare model and dataloader with accelerator
    reward_model, dataloader = accelerator.prepare(reward_model, dataloader)

    socratic_scores = []
    for batch in tqdm(dataloader, desc="Scoring"):
        with torch.no_grad():
            # Socratic score
            outputs = accelerator.unwrap_model(reward_model)(
                input_ids=batch[f"input_ids"],
                attention_mask=batch[f"attention_mask"],
            )
            score = outputs.logits.squeeze().tolist() # .logits.item()
            all_scores = accelerator.gather_for_metrics(score)
            socratic_scores.extend(all_scores)

    if accelerator.is_main_process:
        print("Pushing scores to hub...")
        
        # Add scores to dataset
        dataset = dataset.remove_columns([f"input_ids", f"attention_mask"])
        dataset = dataset.add_column("socratic", socratic_scores)
        
        if accelerator.is_main_process:
            print("Pushing evaluation dataset to the hub...")
            if method=="gold":
                dataset_hub_id = f"{user_id}/{method}-gsm8k-grpo-eval-seed{seed}"
            elif method == "maxmin":
                dataset_hub_id = f"{user_id}/{method}-gsm8k-grpo-eval-prop{proportion}-seed{seed}"
            elif method == "sharedrep":
                dataset_hub_id = f"{user_id}/{method}-gsm8k-grpo-eval-prop{proportion}-seed{seed}-k{k}"
            dataset.push_to_hub(
                dataset_hub_id,
                private=True
            )