import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import argparse
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Create chosen and rejected responses.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    parser.add_argument("--proportion", type=float, required=True, help="Proportions of helpful and harmless responses.")
    return parser.parse_args()

def assign_group_id(dataset, seed, proportion):
    """
    Assigns group IDs to the dataset based on the seed and proportion.
    """
    import numpy as np
    np.random.seed(seed)
    
    # Create a random array of 0s and 1s based on the proportion
    group_ids = np.random.choice([0, 1], size=len(dataset), p=[proportion, 1 - proportion])
    group_names = {0: "minority", 1: "majority"}
    
    # Assign group IDs to the dataset
    dataset = dataset.add_column("group_id", group_ids.tolist())
    
    # Assign group names based on the group IDs
    dataset = dataset.map(lambda x: {"group_name": group_names[x["group_id"]]})
    return dataset

def compute_scores(example):

    if example["group_name"] == "minority":
        score0 = (1-example["normalized_length_0"])*0.3 + 0.7*example["positiveness_0"]
        score1 = (1-example["normalized_length_1"])*0.3 + 0.7*example["positiveness_1"]

    if example["group_name"] == "majority":
        score0 = (1-example["normalized_length_0"])
        score1 = (1-example["normalized_length_1"])
    
    return {"score_0": score0, "score_1": score1}


def create_chosen_rejected(example):
    if example["score_0"] >= example["score_1"]:
        return {"chosen": example["response_0"], "rejected": example["response_1"]}
    else:
        return {"chosen": example["response_1"], "rejected": example["response_0"]}
    
def create_chosen_rejected_dataset(
    dataset,
    seed,
    proportion,
):
    """
    Helper function to create a dataset with chosen and rejected responses based on scores.
    """
    dataset = assign_group_id(dataset, seed, proportion)
    dataset = dataset.map(compute_scores, num_proc=32, desc="Computing scores")
    dataset = dataset.map(create_chosen_rejected, num_proc=32, desc="Creating chosen and rejected responses")
    dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in ["chosen", "rejected", "group_id"]]
        )
    return dataset

if __name__ == "__main__":
    
    # Arguments
    args = parse_args()
    user_id = args.user_id
    seed = args.seed
    proportion = args.proportion
    
    # Dataset
    dataset = load_dataset(f"{user_id}/gpt2-imdb-raw", split="train")
    dataset = assign_group_id(dataset, seed, proportion)

    dataset = dataset.map(compute_scores, num_proc=32, desc="Computing scores")
    dataset = dataset.map(create_chosen_rejected, num_proc=32, desc="Creating chosen and rejected responses")
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["chosen", "rejected", "group_id"]]
    )
    
    # Push to hub
    dataset.push_to_hub(
        f"{user_id}/gpt2-imdb-seed{seed}-prop{proportion}",
    )