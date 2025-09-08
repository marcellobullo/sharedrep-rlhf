import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.helper.imdb_utils import compute_length_score, compute_normalized_length_score


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
    if method=="gold":
        dataset_name = f"{user_id}/{method}-imdb-ppo-eval-seed{seed}"
    elif method == "maxmin":
        dataset_name = f"{user_id}/{method}-imdb-ppo-eval-prop{proportion}-seed{seed}"
    elif method == "sharedrep":
        dataset_name = f"{user_id}/{method}-imdb-ppo-eval-prop{proportion}-seed{seed}-k{k}"
    dataset = load_dataset(dataset_name)

    # Positiveness Model
    reward_model_name = "lvwerra/distilbert-imdb"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
    pos_id = reward_model.config.label2id["POSITIVE"]
    reward_model.eval()

    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    def tokenize(example):
        return reward_tokenizer(
            example["completion"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=100,
        )

    with accelerator.main_process_first():

        # Tokenization
        accelerator.print("Tokenizing dataset...")
        dataset = dataset.map(tokenize, desc="Tokenizing", num_proc=32)
        
        # Scoring
        accelerator.print("Scoring...")
        dataset = dataset.map(lambda x: compute_length_score(x, num_responses=1), num_proc=32, desc="Computing length scores")
        dataset = dataset.map(lambda x: compute_normalized_length_score(x, num_responses=1), num_proc=32, desc="Computing normalized length scores")
        
        dataset.set_format(type="torch")

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare model and dataloader with accelerator
    reward_model, dataloader = accelerator.prepare(reward_model, dataloader)

    sentiment_scores = []
    for batch in tqdm(dataloader, desc="Scoring"):
        with torch.no_grad():
            # Socratic score
            outputs = accelerator.unwrap_model(reward_model)(
                input_ids=batch[f"input_ids"],
                attention_mask=batch[f"attention_mask"],
            )
            score = outputs.logits.squeeze().tolist()
            all_scores = accelerator.gather_for_metrics(score)
            sentiment_scores.extend(all_scores)

    if accelerator.is_main_process:
        print("Pushing scores to hub...")
        
        # Add scores to dataset
        dataset = dataset.remove_columns([f"input_ids", f"attention_mask"])
        dataset = dataset.add_column("positiveness", sentiment_scores)
        
        if accelerator.is_main_process:
            print("Pushing evaluation dataset to the hub...")
            dataset.push_to_hub(
                dataset_name,
                private=True
            )