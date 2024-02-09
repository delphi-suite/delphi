from typing import cast

from datasets import Dataset


def token_map(
    tokenized_dataset: Dataset,
) -> dict[int, list[tuple[int, int]]]:
    """Return a mapping of tokens to their (prompt_idx, token_idx) locations in the tokenized_dataset.

    Args:
        tokenized_dataset (Dataset): A tokenized dataset.

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

    return mapping
