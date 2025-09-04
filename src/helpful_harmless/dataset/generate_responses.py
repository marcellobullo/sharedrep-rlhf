import re
import torch
import argparse
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Generating responses")
    parser.add_argument("--user_id", type=str, required=True)
    parser.add_argument("--disable_chat_template", action="store_true", help="Whether to disable the chat template.")
    return parser.parse_args()

CHAT_TEMPLATE ="""{% for message in messages %}
    {% if message['role'] == 'user' %}
    <|user|>
    {{ message['content'] }}</s>
    {% elif message['role'] == 'assistant' %}
    <|assistant|>
    {% generation %}
    {{ message['content'] }}</s>
    {% endgeneration %}
    {% endif %}
    {% endfor %}
    {% if messages[-1]['role'] != 'assistant' %}
    <|assistant|>
    {% generation %}
    {% endgeneration %}
    {% endif %}
    """

def prepare_for_chat(example):
    parts = re.split(r"(Human:|Assistant:)", example["chosen"])
    parts = [p.strip() for p in parts if p.strip()]
    messages = []
    for i in range(0, len(parts) - 1, 2):
        role = "user" if parts[i] == "Human:" else "assistant"
        messages.append({"role": role, "content": parts[i + 1]})
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    return {"messages": messages}

def convert_messages_to_text(messages):
    formatted_prompts = "\n".join(
        f"{'Human' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
    )
    return formatted_prompts

def tokenize(example, tokenizer, max_new_tokens=256, apply_chat_template=True):
        if apply_chat_template:
            chat_prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        else:
            chat_prompt = convert_messages_to_text(example["messages"])
        
        inputs = tokenizer(
            chat_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2048 - max_new_tokens - 1,
            padding_side="left"
        )
        return {
            "prompt": convert_messages_to_text(example["messages"]),
            "formatted_prompt": chat_prompt,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

if __name__ == "__main__":

    accelerator = Accelerator()
    device = accelerator.device

    args = parse_args()
    user_id = args.user_id
    apply_chat_template = not args.disable_chat_template

    # Dataset
    batch_size = 32
    max_new_tokens = 256
    num_eval_samples = 50   
    num_responses = 2
    dataset = load_dataset("Anthropic/hh-rlhf", split="train").shuffle().select(range(num_eval_samples))

    # Model
    model_name = f"{user_id}/tinyllama-sft-hh"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare dataset
    with accelerator.main_process_first():
        dataset = dataset.map(prepare_for_chat, num_proc=32, desc="Extracting structured prompts")
        dataset = dataset.map(lambda x: tokenize(x, tokenizer, max_new_tokens, apply_chat_template=apply_chat_template), desc="Tokenizing", num_proc=32)
        generation_dataset = {
            f"response_{i}": [] for i in range(num_responses)
        }
        generation_dataset["prompt"] = dataset["prompt"]
        dataset.set_format(type="torch")

        columns_to_keep = ["input_ids", "attention_mask"]
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        dataset = dataset.remove_columns(columns_to_remove)

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
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_completions = accelerator.gather_for_metrics(completions)
                generation_dataset[f"response_{i}"].extend(all_completions)

    if accelerator.is_main_process:
        print("Pushing completions to hub...")
        dataset_hub_id = f"{user_id}/tinyllama-hh-raw-10K"
        Dataset.from_dict(generation_dataset).push_to_hub(
            dataset_hub_id,
            private=True
        )