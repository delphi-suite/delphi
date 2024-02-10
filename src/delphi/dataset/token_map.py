import os
from pickle import dump
from typing import cast

from datasets import Dataset


def token_map(
    tokenized_dataset: Dataset,
    output_path: str | None = None,
    file_name: str | None = None,
) -> dict[int, list[tuple[int, int]]]:
    """Return a mapping of tokens to their (prompt_idx, token_idx) locations in the tokenized_dataset.

    Args:
        tokenized_dataset (Dataset): A tokenized dataset.
        save_output (bool, optional): Whether to save the output to a file. Defaults to True.
        output_path (str, optional): The output file path. Defaults to "/data/token_map.pkl".

    Returns:
        dict[int, list[tuple[int, int]]]: A mapping of tokens to their (prompt_idx, token_idx)
            locations in the tokenized_dataset.
    """
    mapping = {}
    tokenized_dataset = cast(Dataset, tokenized_dataset)
    for prompt_idx, prompt in enumerate(tokenized_dataset):
        prompt = cast(dict, prompt)
        for token_idx, token in enumerate(prompt["tokens"]):
            mapping.setdefault(token, []).append((prompt_idx, token_idx))

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            dump(mapping, f)

    return mapping
