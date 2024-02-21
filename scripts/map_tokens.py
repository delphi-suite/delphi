#!/usr/bin/env python3

import argparse
<<<<<<< HEAD
import os
import pickle
=======

import pandas as pd
from datasets import Dataset
>>>>>>> a5b5e63 (map_tokens from risky pickle to safe hf)

from delphi.constants import STATIC_ASSETS_DIR
from delphi.eval.token_map import token_map
from delphi.eval.utils import load_validation_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset from huggingface to run token_map on",
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Hugging Face API username",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token",
    )
    args = parser.parse_args()

    dataset = load_validation_dataset(args.dataset_name)

    mapping = token_map(
        dataset
    )  # outputs the dictionary: dict[int, list[tuple[int, int]]]

    df_dataset = pd.DataFrame.from_dict(
        {"token_id": list(mapping.keys()), "token_positions": list(mapping.values())}
    )
    df_dataset.set_index("token_id", inplace=True)  # Set the token_id as the index

    hf_dataset = Dataset.from_pandas(df_dataset)

    dataset_name = args.dataset_name.split("/")[-1]

    repo_id = f"{args.username}/{dataset_name}-token-map"  # location in to hf

    hf_dataset.push_to_hub(
        repo_id=repo_id,
        split="validation",
        private=False,
        token=args.token,
    )
