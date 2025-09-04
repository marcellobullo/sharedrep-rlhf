import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)

import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from src.models.sr_gpt2.sr_gpt2_configuration import SharedRepGPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel


class SharedRepGPT2RM(GPT2PreTrainedModel):
    config_class =  SharedRepGPT2Config
    base_model_prefix = "transformer"

    def __init__(self, config: SharedRepGPT2Config):
        super().__init__(config)
        self.config = config
        self.n_heads = config.n_heads
        self.k = config.k
        self.transformer = GPT2Model(config)
        self.num_labels = config.n_heads

        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, self.k),
            nn.Linear(self.k, self.n_heads, bias=False)
        )

        self.maxmin = False

        # Initialize weights and apply final processing
        self.post_init()

    def enable_maxmin(self):
        self.maxmin = True

    def disable_maxmin(self):
        self.maxmin = False

    def freeze_backbone(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.transformer.parameters():
            param.requires_grad = True

    def unfreeze_heads(self):
        for param in self.score.parameters():
            param.requires_grad = True

    def forward(self, input_ids, group_ids=None, **kwargs):

        # Argument validation
        if not self.maxmin and group_ids is None:
            raise ValueError("`group_ids` must be provided when not in `maxmin` mode")
        batch_size, sequence_length = input_ids.shape[:2]
        if group_ids is not None:
            assert group_ids.shape[
                       0] == batch_size, "Size mismatch: `group_ids.shape[0]` and `batch_size` need to be the same"

        # Identify the index of the last non pad token (to select the reward value from logits)
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = sequence_length - 1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(input_ids.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = sequence_length - 1
            print(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        # Forward through backbone
        kwargs.pop("labels", None)
        outputs = self.transformer(input_ids, **kwargs)
        hidden = outputs.last_hidden_state

        # Pick out the single hidden vector at the last real token
        last_states = hidden[torch.arange(batch_size), last_non_pad_token]

        # Apply the two-stage head just once per example
        head_scores = self.score(last_states)

        if self.maxmin:
            rewards, _ = head_scores.min(dim=1)  # pick the minimum over heads
            rewards = rewards.unsqueeze(-1)

        else:
            # Create one-hot mask M: (batch_size, n_heads)
            M = torch.nn.functional.one_hot(group_ids, num_classes=self.config.n_heads).float()

            # Apply mask to correctly route the inputs
            rewards = (head_scores * M).sum(dim=-1).unsqueeze(-1)

        return SequenceClassifierOutputWithPast(
            loss=None,
            logits=rewards,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )