import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.helper.imdb_utils import pre_processing_imdb
from datasets import load_dataset, Dataset

import argparse

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
    parser = argparse.ArgumentParser(description="Generate IMDb responses using bert-base-uncased model.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    return parser.parse_args()



if __name__ == "__main__":

    accelerator = Accelerator()

    # Arguments
    args = parse_args()
    user_id = args.user_id
    #seed = args.seed

    # Parameters
    batch_size = 128
    num_responses = 2
    max_new_tokens = 100
    dataset_name = "stanfordnlp/imdb"
    model_name = "lvwerra/gpt2-imdb"
    min_review_length = 200
    input_min_text_length = 2
    input_max_text_length = 8

    # Dataset
    dataset = load_dataset(dataset_name, split="train")

    # Model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.pad_token_id = tokenizer.pad_token_id

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
            f"response_{i}": [] for i in range(num_responses)
        }
        generation_dataset["prompt"] = dataset["query"]
        dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True, batch_size=batch_size, num_proc=32, desc="Tokenizing")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
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
                    max_length=max_new_tokens,
                    do_sample=True,
                    top_k=0,
                    temperature=0.7,
                    top_p=0.95,
                    eos_token_id=model.config.eos_token_id,
                    pad_token_id=model.pad_token_id,
                )
                completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_completions = accelerator.gather_for_metrics(completions)
                generation_dataset[f"response_{i}"].extend(all_completions)
    
    if accelerator.is_main_process:
        print("Pushing completions to hub...")
        dataset_hub_id = f"{user_id}/gpt2-imdb-raw"
        Dataset.from_dict(generation_dataset).push_to_hub(
            dataset_hub_id,
            private=True
        )