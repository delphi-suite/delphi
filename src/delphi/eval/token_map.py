import os
from pickle import dump
from typing import cast

from datasets import Dataset


def token_map(
    tokenized_dataset: Dataset,
    output_path: str | None = None,
) -> dict[int, list[tuple[int, int]]]:
    """Return a mapping of tokens to their (prompt_idx, token_idx) locations in the tokenized_dataset."""

    mapping = {}
    for prompt_idx, prompt in enumerate(tokenized_dataset):
        prompt = cast(dict, prompt)
        for token_idx, token in enumerate(prompt["tokens"]):
            mapping.setdefault(token, []).append((prompt_idx, token_idx))

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            dump(mapping, f)

    return mapping
