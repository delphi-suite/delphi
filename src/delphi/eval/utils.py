from collections.abc import Callable

import torch
from jaxtyping import Float, Int


def get_all_logprobs(
    model: Callable, input_ids: Int[torch.Tensor, "batch seq"]
) -> Float[torch.Tensor, "batch seq vocab"]:
    # batch, seq, vocab
    logits = model(input_ids).logits
    return torch.log_softmax(logits, dim=-1)


def gather_logprobs(
    logprobs: Float[torch.Tensor, "batch seq vocab"],
    tokens: Int[torch.Tensor, "batch seq"],
) -> Float[torch.Tensor, "batch seq"]:
    return torch.gather(logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)


def get_next_logprobs(
    model: Callable, input_ids: Int[torch.Tensor, "batch seq"]
) -> Float[torch.Tensor, "batch shorter_seq"]:
    logprobs = get_all_logprobs(model, input_ids[:, :-1])
    next_tokens = input_ids[:, 1:]
    return gather_logprobs(logprobs, next_tokens)
