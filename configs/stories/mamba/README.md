pad_token_id - we're not using pad tokens, do we don't set it
layer_norm_eps - different than rms norm eps in mamba
initializer_range - different in mamba & llama
residual_in_fp32 - mamba specific parameter
time_step_* - mamba specific, sane defaults
there is no way to untie embeddings and unembeddings in mamba, they're tied by default
https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/mamba/modeling_mamba.py#L602-L610
rescale_prenorm_residual was True in original paper, so we set it to True, despite HF default being false
using default for use_cache
state_size is default