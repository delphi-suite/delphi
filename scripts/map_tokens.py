from datasets import DatasetDict, IterableDataset, IterableDatasetDict
from datasets.arrow_dataset import Dataset


def token_map(
    tokenized_dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
) -> dict[int, list[tuple[int, int]]]:
    mapping = {}

    for prompt_idx, prompt in enumerate(tokenized_dataset):
        for token_idx, token in enumerate(prompt["tokens"]):
            mapping.setdefault(token, []).append((prompt_idx, token_idx))

    return mapping
