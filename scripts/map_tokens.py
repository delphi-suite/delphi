#!/usr/bin/env python3

import argparse
import pickle
from importlib.resources import files

from delphi.eval.token_map import token_map
from delphi.eval.utils import load_validation_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "dataset_name", help="Dataset from huggingface to run token_map on"
    )
    parser.add_argument("--output", help="Output file name", default="token_map.pkl")
    args = parser.parse_args()

    dataset = load_validation_dataset(args.dataset_name)

    mapping = token_map(dataset)

    with open(f"{files('delphi').joinpath('static')}/{args.output}", "wb") as f:
        pickle.dump(mapping, file=f)
