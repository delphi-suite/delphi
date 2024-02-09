from collections.abc import Callable
from typing import cast

import torch
from datasets import Dataset, load_dataset
from jaxtyping import Float, Int

ALLOWED_CHARS = set(
    " \n\"'(),.:?!0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


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


# TODO: replace this with HF dataset instance using function (see #25)
def load_orig_ds_txt(split: str) -> list[str]:
    # checking just startswith, because you can include slice like "train[:1000]"
    assert split.startswith("train") or split.startswith("validation")
    hf_ds = load_dataset(f"roneneldan/TinyStories", split=split)
    dataset = []
    # hf_ds by type could be an IterableDataset, which is not subscriptable
    # in practice, it's not
    for sample_txt in hf_ds["text"]:  # type: ignore
        # encoding issues and rare weird prompts
        if not set(sample_txt).issubset(ALLOWED_CHARS):
            continue
        dataset.append(sample_txt)
    return dataset


def tokenize(tokenizer, sample_txt: str) -> Int[torch.Tensor, "pos"]:
    # supposedly this can be different than prepending the bos token id
    return tokenizer.encode(tokenizer.bos_token + sample_txt, return_tensors="pt")[0]


def get_logits(model, sample_tok):
    sample_tok = sample_tok.unsqueeze(0)
    return model(sample_tok).logits[0]


def get_probs(model, sample_tok):
    logits = get_logits(model, sample_tok)
    # drop the value for the last position, as we don't know
    # what is the correct next token there
    # pos, d_vocab
    return torch.softmax(logits, dim=-1)[:-1]


def get_correct_probs(model, sample_tok):
    probs = get_probs(model, sample_tok)
    # out of d_vocab values, take the one that corresponds to the correct next token
    return probs[range(len(probs)), sample_tok[1:]]


def get_correct_and_all_probs(model, sample_tok):
    """Get probabilities for the actual next token and for all predictions"""
    probs = get_probs(model, sample_tok)
    correct_probs = probs[range(len(probs)), sample_tok[1:]]
    return correct_probs, probs


def get_correct_and_top_probs(model, sample_tok, top_k=3):
    """Get probabilities for the actual next token and for top k predictions"""
    probs = get_probs(model, sample_tok)
    correct_probs = probs[range(len(probs)), sample_tok[1:]]
    top_k_probs = torch.topk(probs, top_k, dim=-1)
    return correct_probs, top_k_probs
