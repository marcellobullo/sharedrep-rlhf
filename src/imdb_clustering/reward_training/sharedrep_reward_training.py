import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)
os.environ["HF_HOME"] = "/hdd/mb1921/"

import torch
import random
from trl import RewardConfig
from datasets import load_dataset
from accelerate import Accelerator
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.models.sr_gpt2.sr_gpt2_modeling import SharedRepGPT2RM
from src.helper.sharedrep_reward_trainer import SharedRepRMTrainer, SharedRepRewardDataCollatorWithPadding

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Train SharedRep Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    parser.add_argument("--k", type=int, required=True, help="Inner hidden dimension size.")
    parser.add_argument("--em_iters", type=int, default=10, help="Number of EM iterations.")
    parser.add_argument("--num_users", type=int, default=30, help="Number of synthetic users.")
    return parser.parse_args()

# Partition the dataset into 'num_users' synthetic users using round robin
def partition_data_round_robin(dataset, num_users: int, seed: int):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)  # Shuffle to avoid ordering bias
    assignments = {i: [] for i in range(num_users)}  # Dict to hold user-specific sample indices
    for idx, data_idx in enumerate(indices):
        user_id = idx % num_users  # Round-robin assignment
        assignments[user_id].append(data_idx)

    user_ids = [0] * len(dataset)
    for user_id, data_indices in assignments.items():
        for i in data_indices:
            user_ids[i] = user_id

    return dataset.add_column("user_id", user_ids)

def score_dataset(dataset, model, batch_size):

    accelerator = Accelerator()

    num_responses=2
    
    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare model and dataloader with accelerator
    model.eval()
    n_groups = model.n_heads
    model, dataloader = accelerator.prepare(model, dataloader)

    # Prepare scores
    scores = {f"score_r{i}_group{g}": [] 
              for i in range(num_responses) for g in range(n_groups)}

    # Compute scores
    for batch in tqdm(dataloader, desc="Scoring"):
        with torch.no_grad():
            for i in range(num_responses):
                for group_id in range(n_groups):
                    
                    # Create group_ids tensor
                    group_ids = torch.ones(batch[f"input_ids_{i}"].shape[0], dtype=torch.long).to(model.device) * group_id
                    
                    # Get scores
                    outputs = accelerator.unwrap_model(model)(
                        input_ids=batch[f"input_ids_{i}"].squeeze(),
                        attention_mask=batch[f"attention_mask_{i}"].squeeze(),
                        group_ids=group_ids,
                    )
                    score = outputs.logits.squeeze().tolist()
                    all_scores = accelerator.gather_for_metrics(score)
                    scores[f"score_r{i}_group{group_id}"].extend(all_scores)
    
    accelerator.wait_for_everyone()       
    # Add scores to dataset
    for i in range(num_responses):
        for group_id in range(n_groups):
            dataset = dataset.add_column(f"score_r{i}_group{group_id}", scores[f"score_r{i}_group{group_id}"])

    return dataset

def assign_group_ids_fast(dataset, model, num_users):
    """
    Vectorized version (runs in parallel under the hood).
    Assumes dataset has columns:
      user_id (ints in [0, num_users-1])
      score_r0_group{g}, score_r1_group{g} for g in [0, n_groups-1]
    """
    n_groups = int(model.n_heads)
    num_responses = 2  # kept for clarity; not used below

    # Build tensors
    user_ids = torch.as_tensor(dataset["user_id"], dtype=torch.long)            # [N]
    N = user_ids.numel()

    # Stack scores per group -> shapes [N, G]
    s0 = torch.stack(
        [torch.as_tensor(dataset[f"score_r0_group{g}"], dtype=torch.float32) for g in range(n_groups)],
        dim=1
    )
    s1 = torch.stack(
        [torch.as_tensor(dataset[f"score_r1_group{g}"], dtype=torch.float32) for g in range(n_groups)],
        dim=1
    )

    # log_w = log( exp(s0) / (exp(s0) + exp(s1)) ) computed stably
    # = s0 - logsumexp([s0, s1])
    stacked = torch.stack([s0, s1], dim=-1)                     # [N, G, 2]
    log_w = s0 - torch.logsumexp(stacked, dim=-1)               # [N, G]

    # Sum per user for each group: [U, G]
    per_user_loglik = torch.zeros((num_users, n_groups), dtype=torch.float32)
    per_user_loglik.index_add_(0, user_ids, log_w)              # scatter-add rows by user_id

    # Handle users with no rows: keep -1
    counts = torch.bincount(user_ids, minlength=num_users)      # [U]
    group_ids_per_user = per_user_loglik.argmax(dim=1)          # [U]
    group_ids_per_user[counts == 0] = -1

    # Assign per-sample group id via gather
    group_id_per_sample = group_ids_per_user[user_ids]          # [N]

    # Write column and return
    if "group_id" in dataset.column_names:
        dataset = dataset.remove_columns("group_id")
    dataset = dataset.add_column("group_id", group_id_per_sample.tolist())
    return dataset

def assign_group_ids(dataset, model, num_users):

    num_responses=2

    # Add sample index to dataset
    dataset = dataset.add_column("sample_index", list(range(len(dataset))))
    
    # Prepare group_ids
    n_groups = model.n_heads
    group_ids = [-1] * len(dataset)

    # Update group id
    for user_id in range(num_users):
        user_data = dataset.filter(lambda ex: ex["user_id"] == user_id)
        indices = user_data["sample_index"]

        log_likelihoods = torch.zeros(n_groups)
        for u in range(n_groups):
            total_log_prob = 0.0
            for example in user_data:
                score_0 = example[f"score_r0_group{u}"]
                score_1 = example[f"score_r1_group{u}"]
                w = torch.exp(score_0) / (torch.exp(score_0) + torch.exp(score_1))
                log_w = torch.log(w + 1e-8)
                total_log_prob += log_w
            log_likelihoods[u] = total_log_prob

        group_id = torch.argmax(log_likelihoods)

        for idx in indices:
            group_ids[idx] = group_id

    # Assign group id
    dataset = dataset.map(lambda ex, idx: {"group_id": group_ids[idx]}, with_indices=True)
    dataset = dataset.remove_columns("sample_index")
    return dataset

def compute_gold_scores(example):

    if example["group_id"] == 0:
        score0 = example["positiveness_0"]
        score1 = example["positiveness_1"]

    if example["group_id"] == 1:
        score0 = (1-example["normalized_length_0"])
        score1 = (1-example["normalized_length_1"])

    return {"gold_score_0": score0, "gold_score_1": score1}

def create_chosen_rejected(example):
    if example["gold_score_0"] >= example["gold_score_1"]:
        return {"chosen": example["response_0"], "rejected": example["response_1"]}
    else:
        return {"chosen": example["response_1"], "rejected": example["response_0"]}
    

def clustering(dataset, model, batch_size, num_users):
    dataset = score_dataset(dataset, model, batch_size)
    #dataset = assign_group_ids(dataset, model, num_users)
    dataset = assign_group_ids_fast(dataset, model, num_users)
    dataset = dataset.map(compute_gold_scores, num_proc=32, desc="Computing gold scores")
    dataset = dataset.map(create_chosen_rejected, num_proc=32, desc="Creating chosen and rejected responses")
    dataset = dataset.remove_columns([f"score_r{i}_group{g}" for i in range(2) for g in range(model.n_heads)])
    return dataset

def update_reward_model(model, dataset, tokenizer, data_collator, use_tanh, seed, training_args):

    # Split dataset
    dataset = dataset.train_test_split(test_size=0.05, seed=seed)

    # Train
    trainer = SharedRepRMTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        use_tanh=use_tanh,
        data_collator=data_collator,
    )
    trainer.train()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    return model, trainer

if __name__ == "__main__":

    accelerator = Accelerator()
    
    # Arguments
    args = parse_args()
    user_id = args.user_id
    seed = args.seed
    k = args.k
    em_iters = args.em_iters                  
    num_users = args.num_users
    batch_size = 256
    
    # Reward Params
    n_groups=2
    use_tanh = True
    num_responses = 2
    freeze_backbone = True
    model_name = "lvwerra/gpt2-imdb"
    model_hub_id = f"{user_id}/sharedrep-imdb-reward-clustering-seed{seed}-k{k}"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token  

    # Reward Model
    model = SharedRepGPT2RM.from_pretrained(model_name, k=k, n_heads=n_groups)
    model.disable_maxmin()
    if freeze_backbone: model.freeze_backbone()
    model.unfreeze_heads()
    model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenization
    def tokenize(example, num_responses=2):
        result = {}
        for i in range(num_responses):
            key = f"response_{i}"
            inputs = tokenizer(
                example[key],
                return_tensors="pt",
                padding="max_length",
                max_length=100,
                truncation=True,
            )
            result[f"input_ids_{i}"] = inputs["input_ids"]
            result[f"attention_mask_{i}"] = inputs["attention_mask"]
        return result
    
    # Dataset
    dataset_name = f"{user_id}/gpt2-imdb-raw"
    with accelerator.main_process_first():
        # Dataset Processing
        dataset = load_dataset(dataset_name, split="train")
        dataset = partition_data_round_robin(dataset, num_users=num_users, seed=seed)
        dataset = dataset.map(lambda x: tokenize(x, num_responses=num_responses), desc="Tokenizing", num_proc=32)
        dataset.set_format(type="torch")

    
    # Preparing Training
    data_collator = SharedRepRewardDataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    training_args = RewardConfig(
        output_dir=f"/hdd/mb1921/imdb_clustering/sharedrep-imdb-reward-clustering-seed{seed}-k{k}",
        per_device_train_batch_size=64,
        learning_rate=5e-4,
        num_train_epochs=2,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        run_name=f"EM-training",
        push_to_hub=False,
        hub_model_id=model_hub_id,
    )

    # Learning Rewards with EM Algorithm
    for em_iter in trange(em_iters):
        # E-step
        dataset = clustering(dataset, model, batch_size, num_users)
        # M-step
        model, trainer = update_reward_model(model, dataset, tokenizer, data_collator, use_tanh, seed, training_args)


    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.push_to_hub(dataset_name=dataset_name)
    

        



