from dataclasses import dataclass


@dataclass
class Llama2ConfigData:
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = -1
    eos_token_id: int = -2
    hidden_act: str = "silu"
    hidden_size: int = 288
    initializer_range: float = 0.02
    intermediate_size: int = 288
    max_position_embeddings: int = 513
    model_type: str = "llama"
    num_attention_heads: int = 6
    num_hidden_layers: int = 6
    num_key_value_heads: int = 6
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-06
    rope_scaling: None = None
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    transformers_version: str = "4.36.2"
    use_cache: bool = True
    vocab_size: int = 4096
