import os
import numpy as np
from datasets import load_dataset, DatasetDict
from typing import Optional, Union
from datasets import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from torch.utils.data import DataLoader

class LengthSampler:
    """
    Samples a length
    """

    def __init__(self, min_value: int, max_value: int, seed: Optional[int] = None):
        self.values = list(range(min_value, max_value+1))
        if seed is not None:
            np.random.seed(seed)

    def __call__(self) -> int:
        return np.random.choice(self.values)
    
def compute_length_score(example, num_responses, key="response"):
    if num_responses==1:
        return {"length": len(example[key].split()), "normalized_length": len(example[key].split())}
    return {
        f"length_{i}": len(example[f"{key}_{i}"].split()) for i in range(num_responses)
    }

def compute_normalized_length_score(example, num_responses, key="response"):
    if num_responses==1:
        return {"length": len(example[key].split()), "normalized_length": len(example[key].split())/100}
    return {
        f"normalized_length_{i}": len(example[f"{key}_{i}"].split())/100 for i in range(num_responses)
    }

def sample_fragment(
        sentence,
        tokenizer,
        in_size
):
    tokens = tokenizer(sentence, truncation=True)
    input_ids = tokens['input_ids']
    word_ids = tokens.word_ids()

    if in_size >= len(input_ids):
        in_size = len(input_ids) - 1  # avoid overflow

    # Adjust the endpoint to ensure ending on a word boundary
    end = in_size - 1

    # If the token at `end` shares a word with the next token, expand until word ends
    while (end + 1 < len(word_ids)) and (word_ids[end] == word_ids[end + 1]):
        end += 1

    # Decode the selected tokens
    sampled_query = tokenizer.decode(
        input_ids[0: end + 1],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return sampled_query

def pre_processing_imdb(
        ds: Dataset,
        model_name: str,
        min_review_length: Optional[None] = None,
        input_min_text_length: int = 2,
        input_max_text_length: int = 8
):
    """
    Preprocess the IMDB dataset for the text generation task.
    :param ds: The dataset to be pre-processed. It is expected to be a Hugging Face `datasets.Dataset` object.
    :param tokenizer: The tokenizer to be used for encoding the text reviews. It should be compatible with the dataset.
    :param min_review_length: The minimum length of reviews to be included in the dataset. Reviews shorter than this length will be filtered out.
    :param input_min_text_length: The minimum length of the tokenized input sequences. This value is used by the `LengthSampler` to determine the range of lengths for the input sequences.
    :param input_max_text_length: The maximum length of the tokenized input sequences. This value is used by the `LengthSampler` to determine the range of lengths for the input sequences.
    :return: The pre-processed dataset with tokenized reviews and set format to "torch". Specifically each sample will have the following keys:
        - `"review"`: the whole film review from IMDB dataset,
        - `"label"`: positive or negative according wih IMDB ground-truth,
        - `"query"`: a subset of the review with random length between 2 and 8,
        - `"input_ids"`: tokenized query.
    """

    # Rename the column "text" to "review"
    ds = ds.rename_columns({"text": "review"})

    # Filter for comments that are at least `min_review_length` characters long
    if min_review_length is not None:
        ds = ds.filter(lambda x: len(x["review"]) > min_review_length, batched=False)

    # Set the length sampler
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Choose randomly a length for the input and tokenize the review
    def sample_length(example):
        in_size = input_size()
        example["query"] = sample_fragment(example["review"], tokenizer, in_size)
        return example
    ds = ds.map(sample_length, batched=False, num_proc=32, desc="Sampling length")
    return ds

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def create_dataset(
        dataset_name: str,
        model_name: str,
        num_return_sequences: int,
        device: Optional[str] = "cuda",
        min_review_length: Optional[None] = None,
        input_min_text_length: int = 2,
        input_max_text_length: int = 8,
        processing_batch_size = 16,
        num_proc=2,
        batched = True
):
    """
    Create a new dataset for text generation using a pre-trained language model.
    :param dataset_name: Hugging Face dataset name
    :param model_name: Name of the model checkpoint to use
    :param sentiment_classifier_name: Name of the sentiment classifier model
    :param num_return_sequences: Number of sequences to generate for each prompt
    :param min_review_length: Minimum review length to filter out shorter samples
    :param input_min_text_length: Minimum length for the input text subset
    :param input_max_text_length: Maximum length for the input text subset
    :param processing_batch_size: Batch size for processing and text generation
    :param num_proc: Number of processes to use when mapping data
    :param batched: Whether to enable batched processing in the dataset map
    :return: A dataset containing the original samples along with generated responses.
    """

    # Load the IMDB dataset
    dataset = load_dataset(dataset_name)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Apply your custom preprocessing function
    preprocessed_dataset = pre_processing_imdb(
        dataset,
        tokenizer=tokenizer,
        min_review_length=min_review_length,
        input_min_text_length=input_min_text_length,
        input_max_text_length=input_max_text_length
    )

    # Create the model
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device).to(device)
    model.pad_token_id = tokenizer.pad_token_id

    # # Create the sentiment classifier -- "lvwerra/distilbert-imdb"
    # discriminator = pipeline(
    #     task="sentiment-analysis",
    #     model=sentiment_classifier_name
    # )

    def tokenize_fn(examples):
        return tokenizer(
            examples["query"],
            padding="max_length",
            max_length=8,
            truncation=True,
            padding_side="left",
            return_tensors="pt"
        )

    tokenized_dataset = preprocessed_dataset.map(tokenize_fn, batched=True, num_proc=4)

    def response_generation_fn(examples):#,rank):
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())
        # Model generation on each batch (examples['input_ids'])
        outputs = model.generate(
            examples['input_ids'].to(device),
            attention_mask=examples['attention_mask'].to(device),
            do_sample=True,
            top_k=0,
            temperature=0.7,
            top_p=0.95,
            max_length=100,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.pad_token_id,
            num_return_sequences=num_return_sequences
        )
        query_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {"responses": [query_responses[i:i + num_return_sequences] for i in
                              range(0, len(query_responses), num_return_sequences)]}

    mapped_dataset = tokenized_dataset.map(
        response_generation_fn,
        batched=batched,
        batch_size=processing_batch_size,
        num_proc=1,
        #with_rank=True
    )

    return mapped_dataset

def create_sentiment_dataset(
        dataset_name: str,
        sentiment_classifier_name: str,
        device: Optional[Union[torch.device, str]] = "cuda",
        processing_batch_size: int = 16,
        num_proc: int =1,
        batched: bool = True,
        pos_proportion: float = 0.8
):
    discriminator = pipeline(
        task="sentiment-analysis",
        model=sentiment_classifier_name,
        torch_dtype="auto",
        device=device
    )

    def choose_best_response_fn(examples):
        batched_indices = []
        batched_responses = []

        # Collect all chosen responses
        for i, resp_list in enumerate(examples["responses"]):
            selected = np.random.choice(resp_list, 2, replace=False)
            batched_responses.extend(selected)
            batched_indices.append((i, len(batched_responses) - 2))

        # Single pipeline call over the entire batch
        feedbacks = discriminator(batched_responses)

        # Store results
        chosen_list = []
        rejected_list = []
        group_ids = []
        for (i, start_idx) in batched_indices:
            # Extract feedback for the 2 responses
            row_feedbacks = feedbacks[start_idx:start_idx + 2]
            selected = batched_responses[start_idx:start_idx + 2]

            if np.random.random() < pos_proportion:
                group_ids.append(0)

                # Compute POSITIVE score
                scores = [
                    f["score"] if f["label"] == "POSITIVE" else 1 - f["score"]
                    for f in row_feedbacks
                ]
                chosen_idx = np.argmax(scores)
            else:
                group_ids.append(1)
                lengths = [len(r) for r in selected]
                chosen_idx = np.argmin(lengths)

            chosen_list.append(selected[chosen_idx])
            rejected_list.append(selected[1 - chosen_idx])

        return {"chosen": chosen_list, "rejected": rejected_list, "group_id": group_ids}

    dataset = load_dataset(dataset_name)
    mapped_dataset = dataset.map(choose_best_response_fn, batched=batched, batch_size=processing_batch_size, num_proc=num_proc)

    return mapped_dataset

