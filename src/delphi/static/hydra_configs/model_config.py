from dataclasses import dataclass
from typing import Optional

from beartype import beartype


@dataclass
class Llama2HfConfig:
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int
    eos_token_id: int
    hidden_act: str
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    model_type: str
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    pretraining_tp: int
    rms_norm_eps: float
    rope_scaling: Optional[float]
    rope_theta: float
    tie_word_embeddings: bool
    transformers_version: str
    use_cache: bool
    vocab_size: int


@dataclass
class Config:
    max_epochs: int
    eval_interval: int
    eval_iters: int
    llama2hf_config: Llama2HfConfig


# Instantiate the configs with hydra


@beartype
def _instantiate_llama2hf_config(llama2hf_config_dict: dict) -> Llama2HfConfig:
    # This function is responsible for converting the llama2hf_config dictionary
    # into an instance of Llama2HfConfig data class.
    return Llama2HfConfig(**llama2hf_config_dict)


@beartype
def instantiate_config(cfg_dict: dict) -> Config:
    # Ensure llama2hf_config is properly instantiated as Llama2HfConfig
    # before instantiating the Config object.
    if "llama2hf_config" in cfg_dict:
        cfg_dict["llama2hf_config"] = _instantiate_llama2hf_config(
            cfg_dict["llama2hf_config"]
        )
    return Config(**cfg_dict)
