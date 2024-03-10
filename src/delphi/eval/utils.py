from collections.abc import Callable
from typing import cast

import torch
from datasets import Dataset, load_dataset
from jaxtyping import Float, Int
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from delphi.eval import constants


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


def load_validation_dataset(dataset_name: str, split_slice: str = "") -> Dataset:
    if "/" not in dataset_name:
        dataset_name = f"delphi-suite/{dataset_name}"
    data_str = f"data/validation-*.parquet"
    dataset = load_dataset(
        dataset_name,
        data_files=data_str,
        verification_mode="no_checks",
        # this seems to be the only split when using data_files
        # regardless of the files we're actually loading
        split=f"train{split_slice}",
    )
    return cast(Dataset, dataset)


def tokenize(
    tokenizer: PreTrainedTokenizerBase, sample_txt: str
) -> Int[torch.Tensor, "seq"]:
    # supposedly this can be different than prepending the bos token id
    return cast(
        Int[torch.Tensor, "seq"],
        tokenizer.encode(tokenizer.bos_token + sample_txt, return_tensors="pt")[0],
    )


def load_logprob_dataset(model: str) -> Dataset:
    return load_dataset(f"dephi-suite/v0-next-logprobs-{model}")  # type: ignore


def load_logprob_datasets(split: str = "validation") -> dict[str, list[list[float]]]:
    return {
        model: cast(dict, load_logprob_dataset(model)[split])["logprobs"]
        for model in constants.LLAMA2_MODELS
    }
