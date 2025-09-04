import torch
import argparse
from tqdm.auto import tqdm
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Score responses.")
    parser.add_argument("--user_id", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":

    accelerator = Accelerator()
    device = accelerator.device

    args = parse_args()
    user_id = args.user_id

    # Dataset
    batch_size = 16
    max_new_tokens = 128
    num_responses = 2  # Number of responses to score
    dataset = load_dataset(f"{user_id}/tinyllama-hh-raw-10K", split="train")

    # Reward models
    rm_tokenizer = AutoTokenizer.from_pretrained('Ray2333/gpt2-large-harmless-reward_model')
    rm_tokenizer.pad_token = rm_tokenizer.eos_token

    reward_model_harmless = AutoModelForSequenceClassification.from_pretrained(
        'Ray2333/gpt2-large-harmless-reward_model',
        num_labels=1,  
    ).to(device).eval()
    reward_model_harmless.config.pad_token_id = rm_tokenizer.pad_token_id
    
    reward_model_helpful = AutoModelForSequenceClassification.from_pretrained(
        'Ray2333/gpt2-large-helpful-reward_model',
        num_labels=1,
    ).to(device).eval()
    reward_model_helpful.config.pad_token_id = rm_tokenizer.pad_token_id


    def tokenize(example, num_responses=2):
        result = {}

        for i in range(num_responses):
            key = f"response_{i}"
            inputs = rm_tokenizer(
                example[key],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=reward_model_harmless.config.max_position_embeddings - max_new_tokens - 1
            )
            result[f"input_ids_{i}"] = inputs["input_ids"][0].tolist()
            result[f"attention_mask_{i}"] = inputs["attention_mask"][0].tolist()
        return result
    
    with accelerator.main_process_first():
        accelerator.print("Tokenizing dataset...")
        dataset = dataset.map(lambda x: tokenize(x, num_responses=num_responses), desc="Tokenizing", num_proc=32)
        dataset.set_format(type="torch")

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare model and dataloader with accelerator
    reward_model_harmless, reward_model_helpful, dataloader = accelerator.prepare(reward_model_harmless, reward_model_helpful, dataloader)

    harmless_scores = {f"harmless_score_{i}": [] for i in range(num_responses)}
    helpful_scores = {f"helpful_score_{i}": [] for i in range(num_responses)}
    for batch in tqdm(dataloader, desc="Scoring"):
        with torch.no_grad():
            for i in range(num_responses):
                
                # Helpful Score
                outputs = accelerator.unwrap_model(reward_model_helpful)(
                    input_ids=batch[f"input_ids_{i}"],
                    attention_mask=batch[f"attention_mask_{i}"],
                )
                helpful_score = outputs.logits.squeeze().tolist()
                all_helpful_scores = accelerator.gather_for_metrics(helpful_score)
                helpful_scores[f"helpful_score_{i}"].extend(all_helpful_scores)

                # Harmless Score
                outputs = accelerator.unwrap_model(reward_model_harmless)(
                    input_ids=batch[f"input_ids_{i}"],
                    attention_mask=batch[f"attention_mask_{i}"],
                )
                harmless_score = outputs.logits.squeeze().tolist()
                all_harmless_scores = accelerator.gather_for_metrics(harmless_score)
                harmless_scores[f"harmless_score_{i}"].extend(all_harmless_scores)

    if accelerator.is_main_process:
        print("Pushing scores to hub...")
        
        # Add scores to dataset
        for i in range(num_responses):
            dataset = dataset.remove_columns([f"input_ids_{i}", f"attention_mask_{i}"])
            dataset = dataset.add_column(f"harmless_score_{i}", harmless_scores[f"harmless_score_{i}"])
            dataset = dataset.add_column(f"helpful_score_{i}", helpful_scores[f"helpful_score_{i}"])
        
        dataset.push_to_hub(
            f"{user_id}/tinyllama-hh-raw-10K",
            private=True,
        )
