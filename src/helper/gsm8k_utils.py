import re
import numpy as np


# SYSTEM_PROMPT = "You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>"

# def prepare_chat_template(example):
#     return {
#         "prompt": [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user",   "content": example["formatted_prompts"]},
#             {"role": "assistant", "content": "\nA: Let's think step by step."},
#         ],
#         "answer": extract_hash_answer(example["solution"]),
#     }

# def apply_chat_template(example, tokenizer):
#     return tokenizer.apply_chat_template(
#         example["prompts"],
#         add_generation_prompt=True,
#         tokenize=True,
# 	    return_dict=True,
# 	    return_tensors="pt",
#         continue_final_message=True # We want the mode to continue from "Let's think step by step."
#     )


FEW_SHOT_PREFIX = """These are examples of how to solve math problems step by step:

    Q: If there are 3 apples and you eat one, how many are left?
    A: Let's think step by step. There are 3 apples. You eat one. So 3 - 1 = 2.
    #### The final answer is 2.

    Q: Tom had 4 pencils. He gave 1 to Sarah and bought 3 more. How many does he have now?
    A: Let's think step by step. He had 4 and gave away 1, so 4 - 1 = 3. Then he bought 3, so 3 + 3 = 6.
    #### The final answer is 6.

    The following is the only question you need to answer:
    """

def format_prompt(example: str) -> str:
        return {
            "formatted_prompts": FEW_SHOT_PREFIX + f"\nQ: {example['question']}. Think step by step and provide the final answer prefixed by ####.\nA: "
        }

def extract_sentence_components(text):
    marker = "The following is the only question you need to answer:"
    qnr = text.split(marker, 1)[-1].strip()
    parts = qnr.split("\nA: ", 1)
    question = parts[0]
    response = parts[1] if len(parts) > 1 else " "
    return question.strip(), response.strip()

def extract_hash_answer(text):
    if "####" not in text: return None
    return float(text.split("####")[1].rstrip(".,").replace(",", ""))

def extract_final_answer(text):
    _, main = extract_sentence_components(text)

    # Regular expression to capture content inside \boxed{}
    boxed_pattern = r"\\boxed\{(.*?)\}"
    match = re.search(boxed_pattern, main)
    if match:
        num_str =  match.group(1).rstrip(".,").replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            return None

    # Try to extract using explicit patterns first
    for pattern in [
        r"The answer is ([\d.,\-]+)", 
        r"Final answer: ([\d.,\-]+)", 
        r"The final answer is ([\d.,\-]+)"
    ]:
        match = re.search(pattern, main, re.IGNORECASE | re.DOTALL)
        if match:
            num_str = match.group(1).rstrip(".,").replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                continue  # fall through to regex fallback if conversion fails

    # Fallback: last number in the text
    numbers = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", main)
    if numbers:
        num_str = numbers[-1].rstrip(".,").replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            return None

def compute_length_score(example, num_responses, key="response"):
    if num_responses==1:
        return {"length": len(example[key].split())}
    return {
        f"length_{i}": len(example[f"{key}_{i}"].split()) for i in range(num_responses)
    }

def compute_correctness_score(example, ground_truth_key="answer", key="response", num_responses=2):
    true = extract_hash_answer(example[ground_truth_key])
    
    if num_responses == 1:
        pred = extract_final_answer(example[key])
        return {"correctness": int(pred == true)}
    
    correctness = {}
    for i in range(num_responses):
        pred = extract_final_answer(example[f"{key}_{i}"])
        correctness[f"correctness_{i}"] = int(pred == true)
    return correctness


def add_stats(dataset, keys=("socratic", "length")):
    """
    Compute global mean and std for all keys_i across the dataset,
    and add them as new columns (mean_key_i, std_key_i).
    
    Args:
        dataset: HuggingFace Dataset
        keys (list/tuple): base names to process, e.g. ["socratic", "length"]
    """
    # Find all indices present for these keys
    indices = sorted({
        int(col.split("_")[-1])
        for col in dataset.column_names
        for key in keys
        if col.startswith(f"{key}_")
    })

    for key in keys:
        for i in indices:
            colname = f"{key}_{i}"
            if colname not in dataset.column_names:
                continue  # skip if this (key, index) pair is missing
            
            mean_val = np.mean(dataset[colname])
            std_val  = np.std(dataset[colname]) or 1.0
            
            dataset = dataset.add_column(f"mean_{key}_{i}", [mean_val] * len(dataset))
            dataset = dataset.add_column(f"std_{key}_{i}",  [std_val]  * len(dataset))
    
    return dataset


# def extract_final_answer_V1(text):
#     marker = "Let's think step by step and provide the final answer prefixed by ####."
#     text = text.split(marker, 1)[-1].strip()
#     patterns = [
#         r"#+.*?(-?[0-9][0-9,]*\.?[0-9]*)",                             # preferred: any number of # markers
#         r"(?:final answer is|answer is)\s*(-?[0-9][0-9,]*\.?[0-9]*)"   # fallback: phrases
#     ]
#     match = None
#     for pat in patterns:
#         match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
#         if match:
#             break
#     if match:
#         # clean up number
#         raw_number = match.group(1).rstrip(".,!?%")   # strip trailing punctuation
#         clean_number = raw_number.replace(",", "")
#         return float(clean_number) if "." in clean_number else int(clean_number)
#     return None

# def extract_final_answer_V2(text):
#     _, response = extract_sentence_components(text)
#     patterns = [
#         # r"#+.*?(-?[0-9][0-9,]*\.?[0-9]*)",                             # preferred: any number of # markers
#         r"#{3,}.*?(-?[0-9][0-9,]*\.?[0-9]*)",
#         r"(?:final answer is|answer is)\s*(-?[0-9][0-9,]*\.?[0-9]*)"   # fallback: phrases
#     ]
#     match = None
#     for pat in patterns:
#         match = re.search(pat, response, re.IGNORECASE | re.DOTALL)
#         if match:
#             break
#     if match:
#         # clean up number
#         raw_number = match.group(1).rstrip(".,!?%")   # strip trailing punctuation
#         clean_number = raw_number.replace(",", "")
#         return float(clean_number) if "." in clean_number else int(clean_number)
#     return None

# def extract_final_answer_V3(text):
#     _, main = extract_sentence_components(text)
#     for pattern in [r"answer is ([\d\.\-]+)", r"final answer is: ([\d\.\-]+)"]:
#         match = re.search(pattern, main, re.IGNORECASE | re.DOTALL)
#         if match:
#             return float(match.group(1).replace(",", ""))
        
#         numbers = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", main)
#         return float(numbers[-1].replace(",", "")) if numbers else None