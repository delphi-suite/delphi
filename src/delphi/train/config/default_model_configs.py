import transformers
from transformers import LlamaConfig, MambaConfig

from delphi.constants import ModelTypes

llama_config = LlamaConfig(
    attention_bias=False,
    attention_dropout=0.0,
    bos_token_id=-1,
    eos_token_id=-2,
    hidden_act="silu",
    hidden_size=288,
    initializer_range=0.02,
    intermediate_size=288,
    max_position_embeddings=513,
    model_type="llama",
    num_attention_heads=6,
    num_hidden_layers=6,
    num_key_value_heads=6,
    pretraining_tp=1,
    rms_norm_eps=1e-06,
    rope_scaling=None,
    rope_theta=10000.0,
    tie_word_embeddings=False,
    use_cache=True,
    vocab_size=4096,
    transformers_version=transformers.__version__,
)

mamba_config = MambaConfig(
    # metadata
    model_type="mamba",
    transformers_version=transformers.__version__,
    # model shape
    vocab_size=50280,
    hidden_size=768,
    state_size=16,
    num_hidden_layers=32,
    conv_kernel=4,
    expand=2,
    use_bias=False,
    use_conv_bias=True,
    # tokens
    bos_token_id=0,
    eos_token_id=0,
    pad_token_id=0,
    # time step
    time_step_rank="auto",
    time_step_scale=1.0,
    time_step_min=0.001,
    time_step_max=0.1,
    time_step_init_scheme="random",  # "random" or "uniform"
    time_step_floor=0.0001,
    # misc
    layer_norm_epsilon=1e-05,
    hidden_act="silu",
    initializer_range=0.1,
    residual_in_fp32=True,
    rescale_prenorm_residual=False,
    use_cache=True,
    tie_word_embeddings=True,
)

default_config_dicts = {
    ModelTypes.LLAMA: llama_config.to_dict(),
    ModelTypes.MAMBA: mamba_config.to_dict(),
}
