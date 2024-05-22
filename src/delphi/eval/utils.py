from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int
from transformers import PreTrainedModel


def get_all_logprobs(
    model: Callable, input_ids: Int[torch.Tensor, "batch seq"]
) -> Float[torch.Tensor, "batch seq vocab"]:
    # batch, seq, vocab
    logits = model(input_ids).logits
    return torch.log_softmax(logits, dim=-1)


# convenience wrapper for calling on a single sample
def get_single_logprobs(
    model: Callable, input_ids: Int[torch.Tensor, "seq"]
) -> Float[torch.Tensor, "seq vocab"]:
    return get_all_logprobs(model, input_ids.unsqueeze(0))[0]


def gather_logprobs(
    logprobs: Float[torch.Tensor, "batch seq vocab"],
    tokens: Int[torch.Tensor, "batch seq"],
) -> Float[torch.Tensor, "batch seq"]:
    return torch.gather(logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)


def get_all_and_next_logprobs(
    model: Callable,
    input_ids: Int[torch.Tensor, "batch seq"],
) -> tuple[
    Float[torch.Tensor, "batch shorter_seq vocab"],
    Float[torch.Tensor, "batch shorter_seq"],
]:
    logprobs = get_all_logprobs(model, input_ids[:, :-1])
    next_tokens = input_ids[:, 1:]
    return logprobs, gather_logprobs(logprobs, next_tokens)


def get_all_and_next_logprobs_single(
    model: Callable,
    input_ids: Int[torch.Tensor, "seq"],
) -> tuple[
    Float[torch.Tensor, "shorter_seq vocab"],
    Float[torch.Tensor, "shorter_seq"],
]:
    all_logprobs, next_logprobs = get_all_and_next_logprobs(
        model, input_ids.unsqueeze(0)
    )
    return all_logprobs[0], next_logprobs[0]


def get_next_and_top_k_probs(
    model: PreTrainedModel, input_ids: Int[torch.Tensor, "seq"], k: int = 3
) -> tuple[Float[torch.Tensor, "shorter_seq"], torch.return_types.topk,]:
    all_logprobs, next_logprobs = get_all_and_next_logprobs_single(model, input_ids)
    all_probs = torch.exp(all_logprobs)
    next_probs = torch.exp(next_logprobs)
    top_k = torch.topk(all_probs, k, dim=-1)
    return next_probs, top_k


def dict_filter_quantile(
    d: dict[Any, float], q_start: float, q_end: float
) -> dict[Any, float]:
    if not (0 <= q_start < q_end <= 1):
        raise ValueError("Invalid quantile range")
    q_start_val = np.nanquantile(list(d.values()), q_start)
    q_end_val = np.nanquantile(list(d.values()), q_end)
    return {
        k: v for k, v in d.items() if q_start_val <= v <= q_end_val and not np.isnan(v)
    }
