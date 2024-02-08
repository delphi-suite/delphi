from collections.abc import Callable
from typing import cast

import torch
from datasets import Dataset, load_dataset
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


def load_validation_dataset(dataset_name: str) -> Dataset:
    if "/" not in dataset_name:
        dataset_name = f"delphi-suite/{dataset_name}"
    data_str = f"data/validation-*.parquet"
    dataset = load_dataset(
        dataset_name,
        data_files=data_str,
        verification_mode="no_checks",
        # this seems to be the only split when using data_files
        # regardless of the files we're actually loading
        split="train",
    )
    return cast(Dataset, dataset)
