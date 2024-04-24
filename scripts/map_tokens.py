#!/usr/bin/env python3

import argparse

import pandas as pd
from datasets import Dataset

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
    parser.add_argument(
        "--tokenizer-size",
        type=int,
        default=4096,
        help="Size of the tokenizer",
    )
    args = parser.parse_args()

    dataset = load_validation_dataset(args.dataset_name)

    hf_dataset = Dataset.from_dict(
        {"prompt_pos_idx": token_map(dataset, args.tokenizer_size)}
    )

    repo_id = f"{args.username}/v0-token-map"  # location in to hf

    hf_dataset.push_to_hub(
        repo_id=repo_id,
        split="validation",
        private=False,
        token=args.token,
    )
