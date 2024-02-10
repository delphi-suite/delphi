import os
from pickle import dump

from datasets import load_dataset


def load_hf_dataset(
    dataset_name: str, output_path: str, split: str | None = None
) -> None:
    hf_ds = load_dataset(dataset_name, split=split)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        dump(hf_ds, f)
