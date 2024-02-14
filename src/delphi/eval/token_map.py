import os
from typing import cast

from datasets import Dataset


def token_map(
    tokenized_dataset: Dataset,
) -> dict[int, list[tuple[int, int]]]:
    """Return a mapping of tokens to their (prompt_idx, token_idx) locations in the tokenized_dataset."""

    mapping = {}
    for prompt_idx, prompt in enumerate(tokenized_dataset):
        prompt = cast(dict, prompt)
        for token_idx, token in enumerate(prompt["tokens"]):
            mapping.setdefault(token, []).append((prompt_idx, token_idx))

    return mapping
