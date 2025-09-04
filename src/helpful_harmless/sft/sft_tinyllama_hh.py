import re
import torch
import argparse
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama on HH dataset")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    return parser.parse_args()

def parse_dialogue(text: str):
    segments = re.split(r"(Human:|Assistant:)", text)
    segments = [seg.strip() for seg in segments if seg.strip()]
    messages = []
    for i in range(0, len(segments) - 1, 2):
        role = "user" if segments[i] == "Human:" else "assistant"
        content = segments[i + 1].strip()
        messages.append({"role": role, "content": content})
    return messages

def to_conversational(example):
    return {"messages": parse_dialogue(example["chosen"])}

def has_valid_assistant(example):
    return any(m["role"] == "assistant" and m["content"].strip() for m in example["messages"])

CHAT_TEMPLATE = """{% for message in messages %}
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

if __name__ == "__main__":
    args = parse_args() 
    user_id = args.user_id

    # Configurations
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    repo_name = f"{user_id}/tinyllama-sft-hh"
    use_bf16 = True

    # Load and preprocess dataset
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    dataset = dataset.map(to_conversational, remove_columns=dataset.column_names)
    dataset = dataset.filter(has_valid_assistant)

    # Split into train/test
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Load tokenizer and patch template
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATE

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # SFT Config
    sft_config = SFTConfig(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=500,
        eval_steps=250,
        learning_rate=2e-5,
        fp16=not use_bf16,
        bf16=use_bf16,
        remove_unused_columns=False,
        push_to_hub=True,
        hub_model_id=repo_name,
        hub_strategy="every_save",
        run_name="tinyllama-sft-hh",
        packing=False,
        dataset_text_field="messages",
        max_seq_length=512,
        assistant_only_loss=True
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    trainer.train()
    trainer.push_to_hub()
