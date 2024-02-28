import os
from typing import cast

from datasets import Dataset


def token_map(
    tokenized_dataset: Dataset,
    tokenizer_size: int,
) -> list[list[tuple[int, int]]]:
    """Return a mapping of tokens to their (prompt_idx, token_idx) locations in the tokenized_dataset."""

    mapping = [[] for _ in range(tokenizer_size)]
    for prompt_idx, prompt in enumerate(tokenized_dataset):
        prompt = cast(dict, prompt)
        for position_idx, token in enumerate(prompt["tokens"]):
            mapping[token].append((prompt_idx, position_idx))
    return mapping
