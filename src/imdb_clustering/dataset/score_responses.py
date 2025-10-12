import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import torch
import argparse
import numpy as np
from tqdm import tqdm
from peft import PeftConfig
from datasets import load_dataset, Dataset, DatasetDict
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.helper.imdb_utils import compute_length_score, compute_normalized_length_score

def parse_args():
    parser = argparse.ArgumentParser(description="Scoring.")
    parser.add_argument("--user_id", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":

    accelerator = Accelerator()

    # Arguments
    args = parse_args()
    user_id = args.user_id

    # Dataset
    batch_size = 64
    num_responses = 2
    dataset_name = f"{user_id}/gpt2-imdb-raw"
    dataset = load_dataset(dataset_name)

    # Positiveness Model
    reward_model_name = "lvwerra/distilbert-imdb"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
    pos_id = reward_model.config.label2id["POSITIVE"]
    reward_model.eval()

    # Reward Tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    # for split_name, split_dataset in dataset.items():


    def tokenize(example, num_responses=2):
        result = {}

        for i in range(num_responses):
            key = f"response_{i}"
            inputs = reward_tokenizer(
                example[key],
                return_tensors="pt",
                padding="max_length",
                max_length=100,
                truncation=True,
                )
            result[f"input_ids_{i}"] = inputs["input_ids"]
            result[f"attention_mask_{i}"] = inputs["attention_mask"]
        return result

    with accelerator.main_process_first():

        # Tokenization
        accelerator.print("Tokenizing dataset...")
        dataset = dataset.map(lambda x: tokenize(x, num_responses=num_responses), desc="Tokenizing", num_proc=32)
        dataset.set_format(type="torch")
        
        # Scoring
        accelerator.print("Scoring...")
        dataset = dataset.map(lambda x: compute_length_score(x, num_responses=num_responses), num_proc=32, desc="Computing length scores")
        dataset = dataset.map(lambda x: compute_normalized_length_score(x, num_responses=num_responses), num_proc=32, desc="Computing normalized length scores")

        

    # For collecting all split results
    #all_score_datasets = DatasetDict()

    for split_name in dataset.keys():

        # Dataloader
        dataloader = DataLoader(dataset[split_name], batch_size=batch_size, shuffle=False)

        # Prepare model and dataloader with accelerator
        reward_model, dataloader = accelerator.prepare(reward_model, dataloader)

        scores = {f"positiveness_{i}": [] for i in range(num_responses)}
        for batch in tqdm(dataloader, desc="Scoring"):
            with torch.no_grad():
                for i in range(num_responses):

                    # Positiveness score
                    outputs = accelerator.unwrap_model(reward_model)(
                        input_ids=batch[f"input_ids_{i}"].squeeze(),
                        attention_mask=batch[f"attention_mask_{i}"].squeeze(),
                    )
                    probs = torch.softmax(outputs.logits, dim=-1)
                    score = probs[:, pos_id]
                    #score = outputs.logits.squeeze().tolist()
                    all_scores = accelerator.gather_for_metrics(score.detach().cpu().tolist())
                    scores[f"positiveness_{i}"].extend(all_scores)

        if accelerator.is_main_process:
            # Add scores to dataset
            for i in range(num_responses):
                dataset[split_name] = dataset[split_name].remove_columns([f"input_ids_{i}", f"attention_mask_{i}"])
                dataset[split_name] = dataset[split_name].add_column(f"positiveness_{i}", scores[f"positiveness_{i}"])

    if accelerator.is_main_process:
        print("Pushing scores to hub...")
        dataset.push_to_hub(
            f"{user_id}/gpt2-imdb-raw",
            private=True,
        )