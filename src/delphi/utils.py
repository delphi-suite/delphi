from collections.abc import Callable
from typing import cast

import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset
from jaxtyping import Float, Int


def hf_split_to_split_name(split: str) -> str:
    return split.split("[")[0]


def load_dataset_split_features(
    repo_id: str,
    split: str,
    features: Features,
) -> Dataset:
    dataset = load_dataset(
        repo_id,
        split=split,
        features=features,
    )
    dataset = cast(Dataset, dataset)
    return dataset


def load_dataset_split_string_feature(
    repo_id: str,
    split: str,
    feature_name: str,
) -> Dataset:
    print("Loading string dataset")
    print(f"{repo_id=}, {split=}, {feature_name=}")
    return load_dataset_split_features(
        repo_id,
        split,
        Features({feature_name: Value("string")}),
    )


def load_dataset_split_sequence_int32_feature(
    repo_id: str,
    split: str,
    feature_name: str,
) -> Dataset:
    print("Loading sequence int32 dataset")
    print(f"{repo_id=}, {split=}, {feature_name=}")
    return load_dataset_split_features(
        repo_id,
        split,
        Features({feature_name: Sequence(Value("int32"))}),
    )


def get_all_hf_branch_names(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    refs = api.list_repo_refs(repo_id)
    return [branch.name for branch in refs.branches]


def gather_logprobs(
    logprobs: Float[torch.Tensor, "batch seq vocab"],
    tokens: Int[torch.Tensor, "batch seq"],
) -> Float[torch.Tensor, "batch seq"]:
    return torch.gather(logprobs, -1, tokens.unsqueeze(-1)).squeeze(-1)


def get_all_logprobs(
    model: Callable, input_ids: Int[torch.Tensor, "batch seq"]
) -> Float[torch.Tensor, "batch seq vocab"]:
    # batch, seq, vocab
    logits = model(input_ids).logits
    return torch.log_softmax(logits, dim=-1)


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
