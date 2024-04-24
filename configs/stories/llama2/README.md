not using padding, so pad_token_id not set
use_cache - using default
pretraining_tp - experimental parallelization we're not using, which is the default
tie_word_embeddings - llama2 used False and this is better for interpretability, note that llama2.c is using True by default, which is probably more efficient use of parameters for very small models
rope settings are widely used defaults
attention_bias - no biases on QKV and output projection is the default and that's what we're using
attention_dropout - this is the only dropout llama2 can use, it's set to prob=0 by default and that's what we're using