import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, DatasetDict
from src.helper.imdb_utils import pre_processing_imdb
import argparse



# Tokenization function
def tokenize(examples, tokenizer):
    return tokenizer(
        examples["query"],
        padding="max_length",
        max_length=8,
        truncation=True,
        padding_side="left",
        return_tensors="pt"
    )

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Generate IMDb responses using gpt2-imdb model.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    return parser.parse_args()

if __name__ == "__main__":
    accelerator = Accelerator()
    args = parse_args()
    user_id = args.user_id

    # Parameters
    batch_size = 128
    num_responses = 2
    max_new_tokens = 100
    dataset_name = "stanfordnlp/imdb"
    model_name = "lvwerra/gpt2-imdb"
    min_review_length = 200
    input_min_text_length = 2
    input_max_text_length = 8

    # Load dataset (all splits) 
    dataset = load_dataset(dataset_name)  # returns DatasetDict

    # Load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.pad_token_id = tokenizer.pad_token_id
    eos_token_id = model.config.eos_token_id
    pad_token_id = model.pad_token_id

    # For collecting all split results
    all_generation_datasets = DatasetDict()

    for split_name, split_dataset in dataset.items():
        if accelerator.is_main_process:
            print(f"\nProcessing split: {split_name}...")

        # Preprocessing
        with accelerator.main_process_first():
            split_dataset = pre_processing_imdb(
                split_dataset,
                model_name=model_name,
                min_review_length=min_review_length,
                input_min_text_length=input_min_text_length,
                input_max_text_length=input_max_text_length
            )

            generation_dataset = {f"response_{i}": [] for i in range(num_responses)}
            generation_dataset["prompt"] = split_dataset["query"]

            split_dataset = split_dataset.map(
                lambda x: tokenize(x, tokenizer),
                batched=True,
                batch_size=batch_size,
                num_proc=8,
                desc=f"Tokenizing {split_name}"
            )
            split_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Dataloader
        dataloader = DataLoader(split_dataset, batch_size=batch_size, shuffle=False)
        model, dataloader = accelerator.prepare(model, dataloader)

        # Generation
        for batch in tqdm(dataloader, desc=f"🚀 Generating for {split_name}"):
            for i in range(num_responses):
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
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
                    completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    all_completions = accelerator.gather_for_metrics(completions)
                    generation_dataset[f"response_{i}"].extend(all_completions)

        # Add results to DatasetDict
        if accelerator.is_main_process:
            all_generation_datasets[split_name] = Dataset.from_dict(generation_dataset)

    # Push final DatasetDict to Hugging Face Hub
    if accelerator.is_main_process:
        print("\nPushing all completions to the Hugging Face Hub...")
        dataset_hub_id = f"{user_id}/gpt2-imdb-raw"
        all_generation_datasets.push_to_hub(dataset_hub_id, private=True)
        print(f"Upload complete: https://huggingface.co/datasets/{dataset_hub_id}")