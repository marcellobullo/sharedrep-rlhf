from datasets import Dataset
from datasets import load_dataset
from typing import Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import gc
import math
import time
import torch
import numpy as np
import pandas as pd
from trl import PPOTrainer
from transformers import (
    GenerationConfig,
)
from trl.trainer.utils import (
    batch_generation,
    first_true_indices,
    forward,
    get_reward,
    selective_log_softmax,
    truncate_response,
    print_rich_table,
    log_table_to_comet_experiment,
)
from collections import defaultdict
from accelerate.utils import gather_object
from trl.core import masked_mean, masked_whiten
from trl.trainer.ppo_trainer import INVALID_LOGPROB
from trl.models.utils import unwrap_model_for_generation

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
        return {"length": len(example[key].split())}
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


class SharedRepPPOTrainer(PPOTrainer):
    def __init__(self, pluralistic_reward_model, use_tanh=False, pointwise=True,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pluralistic_reward_model = pluralistic_reward_model
        self.n_groups = pluralistic_reward_model.n_heads
        self.use_tanh = use_tanh
        self.pointwise = pointwise

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_model
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            repetition_penalty=1.,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []
                with unwrap_model_for_generation(
                        self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i: i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i: i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i: i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    torch.cuda.empty_cache()

                    if ref_policy is None:
                        with self.null_ref_context():
                            ref_output = forward(model.module.policy, query_response, processing_class.pad_token_id)
                    else:
                        ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1: -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    full_value, _, _ = get_reward(
                        unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
                    )
                    value = full_value[:, context_length - 1: -1].squeeze(-1)
                    group_scores = []
                    attention_mask = (postprocessed_query_response != processing_class.pad_token_id).to(
                        postprocessed_query_response.device)
                    position_ids = attention_mask.cumsum(1) - attention_mask.long()
                    input_ids = torch.masked_fill(postprocessed_query_response, ~attention_mask, 0)
                    for group_id in range(self.n_groups):
                        group_ids = group_id*torch.ones_like(postprocessed_query_response[:, 0], device=postprocessed_query_response.device)
                        group_score = self.pluralistic_reward_model(
                            group_ids=group_ids,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids = position_ids,
                            return_dict=True,
                        )["logits"]
                        if self.use_tanh:
                            group_score = torch.tanh(group_score)
                        group_scores.append(group_score)

                    group_scores = torch.stack(group_scores, 0).transpose(0,1)

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(group_scores)
                    values.append(value)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, group_score, unwrapped_model)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1)
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)


                # 4. compute rewards
                kl = logprobs - ref_logprobs
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                ## 4.1 Compute `group_id` as in https://arxiv.org/abs/2402.08925 equation (13)
                # Fs = torch.zeros(rewards.shape[0], rewards.shape[1], self.n_groups, device=rewards.device)
                # Fs[[actual_start, actual_end]] += scores.squeeze(-1)

                if self.pointwise:
                    group_id = torch.argmin(scores, dim=1).unsqueeze(2)
                    rewards[[actual_start, actual_end]] += torch.gather(scores, dim=1, index=group_id).squeeze(1, 2)
                else:
                    group_id = torch.argmin(torch.mean(scores, 0))
                    rewards[[actual_start, actual_end]] += scores[:, group_id].squeeze(-1)

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]

                            output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1: -1]
                            logits /= args.temperature + 1e-7
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            vpred = vpred_temp[:, context_length - 1: -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff ** 2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2,
                        pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather_for_metrics(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.5 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            repetition_penalty=1.,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model.policy,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if self.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            self.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    # _, score, _ = get_reward(
                    #     self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    # )
                    attention_mask = (postprocessed_query_response != processing_class.pad_token_id).to(
                        postprocessed_query_response.device)
                    position_ids = attention_mask.cumsum(1) - attention_mask.long()
                    input_ids = torch.masked_fill(postprocessed_query_response, ~attention_mask, 0)
                    for group_id in range(self.n_groups):
                        group_ids = group_id * torch.ones_like(postprocessed_query_response[:, 0],
                                                               device=postprocessed_query_response.device)
                        score = self.pluralistic_reward_model(
                            group_ids=group_ids,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            return_dict=True,
                        )["logits"]
                        if self.use_tanh:
                            score = torch.tanh(score)

                        table[f"score {group_id}"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )


