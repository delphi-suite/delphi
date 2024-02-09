from typing import cast

from datasets import Dataset


def token_map(
    tokenized_dataset: Dataset,
) -> dict[int, list[tuple[int, int]]]:
    mapping = {}
    tokenized_dataset = cast(Dataset, tokenized_dataset)
    for prompt_idx, prompt in enumerate(tokenized_dataset):
        prompt = cast(dict, prompt)
        for token_idx, token in enumerate(prompt["tokens"]):
            mapping.setdefault(token, []).append((prompt_idx, token_idx))

    return mapping
