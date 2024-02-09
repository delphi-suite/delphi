from collections.abc import Callable
from typing import List, cast

import torch
from datasets import Dataset, load_dataset
from jaxtyping import Float, Int
from transformers import PreTrainedModel, PreTrainedTokenizerBase

ALLOWED_CHARS = set(
    " \n\"'(),.:?!0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


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


def load_text_from_dataset(dataset: Dataset) -> list[str]:
    text = []
    for sample_txt in dataset["story"]:
        # encoding issues and rare weird prompts
        if not set(sample_txt).issubset(ALLOWED_CHARS):
            continue
        text.append(sample_txt)
    return text


def tokenize(
    tokenizer: PreTrainedTokenizerBase, sample_txt: str
) -> Int[torch.Tensor, "seq"]:
    # supposedly this can be different than prepending the bos token id
    return cast(
        Int[torch.Tensor, "seq"],
        tokenizer.encode(tokenizer.bos_token + sample_txt, return_tensors="pt")[0],
    )


def get_correct_and_all_probs(
    model: PreTrainedModel, sample_tok: Int[torch.Tensor, "seq"]
) -> tuple[Float[torch.Tensor, "next_seq"], Float[torch.Tensor, "next_seq vocab"]]:
    """Get probabilities for the actual next token and for all predictions"""
    # remove the first token (the bos token)
    probs = get_single_logprobs(model, sample_tok)[1:]
    correct_probs = probs[range(len(probs)), sample_tok[1:]]
    return correct_probs, probs


def get_correct_and_top_probs(
    model: PreTrainedModel, sample_tok: Int[torch.Tensor, "seq"], top_k: int = 3
) -> tuple[Float[torch.Tensor, "next_seq"], torch.return_types.topk]:
    """Get probabilities for the actual next token and for top k predictions"""
    correct_probs, probs = get_correct_and_all_probs(model, sample_tok)
    top_k_probs = torch.topk(probs, top_k, dim=-1)
    return correct_probs, top_k_probs
