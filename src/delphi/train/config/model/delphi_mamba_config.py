from dataclasses import dataclass
from typing import Union


@dataclass
class DelphiMambaConfig:
    # model shape
    vocab_size: int = 4096
    hidden_size: int = 768
    state_size: int = 16
    num_hidden_layers: int = 32
    conv_kernel: int = 4
    expand: int = 2
    use_bias: bool = False
    use_conv_bias: bool = True
    # tokens
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: int = 0
    # time step
    time_step_rank: Union[int, str] = "auto"
    time_step_scale: float = 1.0
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_init_scheme: str = "random"  # "random" or "uniform"
    time_step_floor: float = 0.0001
    # misc
    layer_norm_epsilon: float = 1e-05
    hidden_act: str = "silu"
    initializer_range: float = 0.1
    residual_in_fp32: bool = True
    rescale_prenorm_residual: bool = False
    use_cache: bool = True
    tie_word_embeddings: bool = True
