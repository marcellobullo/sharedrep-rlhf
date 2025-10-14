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
from src.helper.imdb_utils import pre_processing_imdb
from transformers import AutoTokenizer, AutoModelForCausalLM

def tokenize(examples, tokenizer):
    return tokenizer(
        examples["query"],
        padding="max_length",
        max_length=8,
        truncation=True,
        padding_side="left",
        return_tensors="pt"
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating RLHF methods...")
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

    # Parameters
    batch_size = 128
    max_new_tokens = 100
    dataset_name = "stanfordnlp/imdb"
    model_name = "lvwerra/gpt2-imdb"
    min_review_length = 200
    input_min_text_length = 2
    input_max_text_length = 8

    # Dataset
    dataset = load_dataset(dataset_name, split="test")

    # Policy Model
    if method=="gold":
        policy_model_id = f"{user_id}/{method}-imdb-ppo-clustering-seed{seed}"
    elif method == "maxmin":
        policy_model_id = f"{user_id}/{method}-imdb-ppo-clustering-prop{proportion}-seed{seed}"
    elif method == "sharedrep":
        policy_model_id = f"{user_id}/{method}-imdb-ppo-clustering-prop{proportion}-seed{seed}-k{k}"
    policy_model = AutoModelForCausalLM.from_pretrained(policy_model_id)
    policy_model.eval()
    
    # Tokenizer
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_id, padding_side="left")
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_model.config.pad_token_id = policy_tokenizer.pad_token_id
    eos_token_id = policy_tokenizer.eos_token_id
    pad_token_id = policy_tokenizer.pad_token_id

    # Prepare dataset
    with accelerator.main_process_first():
        
        dataset = pre_processing_imdb(
            dataset,
            model_name=model_name,
            min_review_length=min_review_length,
            input_min_text_length=input_min_text_length,
            input_max_text_length=input_max_text_length
        )

        generation_dataset = {
            "completion": [],
            "prompt": dataset["query"]
        }
        dataset = dataset.map(lambda x: tokenize(x, policy_tokenizer), batched=True, batch_size=batch_size, num_proc=32, desc="Tokenizing")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
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
                max_length=max_new_tokens,
                do_sample=True,
                top_k=0,
                temperature=0.7,
                top_p=0.95,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            completions = policy_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_completions = accelerator.gather_for_metrics(completions)
            generation_dataset[f"completion"].extend(all_completions)
    
    if accelerator.is_main_process:
        print("Pushing evaluation dataset to the hub...")
        if method=="gold":
            dataset_hub_id = f"{user_id}/{method}-imdb-ppo-clustering-eval-seed{seed}"
        elif method == "maxmin":
            dataset_hub_id = f"{user_id}/{method}-imdb-ppo-clustering-eval-prop{proportion}-seed{seed}"
        elif method == "sharedrep":
            dataset_hub_id = f"{user_id}/{method}-imdb-ppo-clustering-eval-prop{proportion}-seed{seed}-k{k}"
        Dataset.from_dict(generation_dataset).push_to_hub(
            dataset_hub_id,
            private=True
        )