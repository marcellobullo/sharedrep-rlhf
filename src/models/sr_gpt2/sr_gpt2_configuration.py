from typing import Optional
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class SharedRepGPT2Config(GPT2Config):
    model_type = "sharedrep-gpt2"

    def __init__(
        self,
        k: Optional[int] = 2,
        n_heads: Optional[int] = 2,
        **kwargs
    ):
        self.k = k
        self.n_heads = n_heads
        super().__init__(**kwargs)