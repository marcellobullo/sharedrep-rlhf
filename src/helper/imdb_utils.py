import numpy as np
from datasets import Dataset
from typing import Optional
from transformers import AutoTokenizer

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
    
# def compute_length_score_old(example, num_responses, key="response"):
#     if num_responses==1:
#         return {"length": len(example[key].split())}
#     return {
#         f"length_{i}": len(example[f"{key}_{i}"].split()) for i in range(num_responses)
#     }

def compute_length_score(example, num_responses, key="response"):
    if num_responses==1:
        return {"length": example["attention_mask"].sum(dim=-1).squeeze().tolist()}
    return {
        f"length_{i}": example[f"attention_mask_{i}"].sum(dim=-1).squeeze().tolist() for i in range(num_responses)
    }

def compute_normalized_length_score(example, num_responses, key="response"):
    if num_responses==1:
        return {"normalized_length": (example["attention_mask"].sum(dim=-1).squeeze()/100).tolist()}
    return {
        f"normalized_length_{i}": (example[f"attention_mask_{i}"].sum(dim=-1).squeeze()/100).tolist() for i in range(num_responses)
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