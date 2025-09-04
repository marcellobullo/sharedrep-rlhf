import argparse
import pandas as pd
from accelerate import Accelerator
from peft import LoraConfig, TaskType
from datasets import load_dataset, Dataset
from trl import RewardTrainer, RewardConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pluralistic Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True)
    return parser.parse_args()

# Create the chosen/rejected dataset
def create_chosen_rejected(dataset_name):

    main_ds = load_dataset("openai/gsm8k", "main")["train"]
    socratic_ds = load_dataset("openai/gsm8k", "socratic")["train"]

    # Dataset
    num_samples = 10000
    main_ds = main_ds.shuffle(seed=14)#.select(range(num_samples))
    socratic_ds = socratic_ds.shuffle(seed=14)#.select(range(num_samples))

    # Format preference data
    pairs = [{
        "prompt": m["question"],
        "chosen": s["answer"],
        "rejected": m["answer"]
    } for m, s in zip(main_ds, socratic_ds)]

    dataset = Dataset.from_pandas(pd.DataFrame(pairs))
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return dataset

if __name__ == "__main__":

    # Initialize accelerator
    accelerator = Accelerator()

    # Arguments
    args = parse_args()
    user_id = args.user_id

    # Tokenizer
    model_name = "gpt2-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Reward model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id

    # PEFT config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    # Dataset
    dataset_name = "openai/gsm8k"
    with accelerator.main_process_first():
        dataset = create_chosen_rejected(dataset_name)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    
    # Training config
    training_args = RewardConfig(
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=["wandb"],
        push_to_hub=True,
        hub_model_id = f"{user_id}/gpt2-gsm8k-lora-new",
        hub_private_repo = True
    )

    # Reward trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    # Push to Hub
    if accelerator.is_main_process:
        print("Pushing model to hub...")
        trainer.push_to_hub(
            dataset_name=dataset_name
        )


