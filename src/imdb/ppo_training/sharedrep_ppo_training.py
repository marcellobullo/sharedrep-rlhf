import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import torch
import argparse
import importlib
from datasets import load_dataset
from accelerate import Accelerator
from trl import PPOConfig, PPOTrainer
from trl.trainer.utils import get_reward
from trl.trainer.utils import first_true_indices
from src.models.sr_gpt2.sr_gpt2_modeling import SharedRepGPT2RM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from src.helper.imdb_utils import pre_processing_imdb



def get_sharedrep_reward( model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the reward logits and the rewards for a given model and query responses.

    Args:
        model (`torch.nn.Module`):
            The model used to compute the reward logits.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`int`):
            The length of the context in the query responses.

    Returns:
        tuple:
            - `reward_logits` (`torch.Tensor`):
                The logits for the reward model.
            - `final_rewards` (`torch.Tensor`):
                The final rewards for each query response.
            - `sequence_lengths` (`torch.Tensor`):
                The lengths of the sequences in the query responses.
    """
    if isinstance(model, SharedRepGPT2RM):
        assert model.maxmin, "The model must be in maxmin mode"

        attention_mask = query_responses != pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

        # Compute rewards
        with torch.no_grad():
            rewards = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
    
        sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
        
        return (
            rewards.logits,
            rewards.logits.squeeze(-1),
            sequence_lengths,
        )
    else:
        return get_reward(model, query_responses, pad_token_id, context_length)

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
    parser = argparse.ArgumentParser(description="Train Pluralistic Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    parser.add_argument("--proportion", required=True, type=float, help="Proportion of minority group.")
    parser.add_argument("--k", type=int, required=True, help="Inner hidden dimension size.")
    return parser.parse_args()

if __name__ == "__main__":


    accelerator = Accelerator()
    device = accelerator.device

    args = parse_args()
    user_id = args.user_id
    seed = args.seed
    proportion = args.proportion
    k = args.k
    

    # Dataset
    batch_size = 16
    dataset_name = f"stanfordnlp/imdb"
    dataset = load_dataset(dataset_name, split="test")


    # Reward Model
    reward_model_name = f"{user_id}/sharedrep-imdb-reward-prop{proportion}-seed{seed}-k{k}"
    reward_model = SharedRepGPT2RM.from_pretrained(reward_model_name)
    reward_model.enable_maxmin()
    
    # Reward Tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name, padding_side="left")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    reward_model.eval()

    # Policy Model
    policy_model_name = "lvwerra/gpt2-imdb"
    policy_hub_id = f"{user_id}/sharedrep-imdb-ppo-prop{proportion}-seed{seed}-k{k}"
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name, padding_side="left")
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

    # Policy model
    policy = AutoModelForCausalLM.from_pretrained( policy_model_name, trust_remote_code=True)

    # Ref policy
    ref_policy = AutoModelForCausalLM.from_pretrained(policy_model_name, trust_remote_code=True)

    # Value model
    value_model = AutoModelForSequenceClassification.from_pretrained( policy_model_name, trust_remote_code=True, num_labels=1)

    with accelerator.main_process_first():

        min_review_length = 200
        input_min_text_length = 2
        input_max_text_length = 8
        
        dataset = pre_processing_imdb(
            dataset,
            model_name=policy_model_name,
            min_review_length=min_review_length,
            input_min_text_length=input_min_text_length,
            input_max_text_length=input_max_text_length
        )
        dataset = dataset.map(lambda x: tokenize(x, policy_tokenizer), batched=True, batch_size=batch_size, num_proc=32, desc="Tokenizing")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataset = dataset.train_test_split(test_size=0.05, seed=seed)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    # Training Arguments
    training_args = PPOConfig(
        logging_steps=10,
        num_train_epochs=5,
        num_ppo_epochs=3,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        bf16=True,
        learning_rate= 3e-06,
        eval_on_start= True,
        push_to_hub=True,
        save_strategy="no",
        hub_private_repo=True,
        hub_model_id=policy_hub_id,
        report_to=["wandb"],
        run_name=policy_hub_id,
        whiten_rewards= True,
        missing_eos_penalty= 1.0,
        total_episodes = 50000,
        lr_scheduler_type= "cosine",
        stop_token= "eos",
        kl_coef=0.05,
    )

    m = importlib.import_module(PPOTrainer.__module__)  
    m.get_reward = get_sharedrep_reward

    # Train
    trainer = PPOTrainer(
        args=training_args,
        model=policy,
        processing_class=policy_tokenizer,
        ref_model=ref_policy,
        reward_model=reward_model,  
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    
    # Save the model
    trainer.push_to_hub(dataset_name=dataset_name)