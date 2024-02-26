#!/usr/bin/env python3

import argparse
import os
import pickle

from delphi.constants import STATIC_ASSETS_DIR
from delphi.eval.token_map import token_map
from delphi.eval.utils import load_validation_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "dataset_name",
        help="Dataset from huggingface to run token_map on. Must be tokenized.",
        default="delphi-suite/v0-tinystories-v2-clean-tokenized",
    )
    parser.add_argument(
        "--output",
        help="Output path name. Must include at least output file name.",
        default="token_map.pkl",
    )
    args = parser.parse_args()

    print("\n", " MAP TOKENS TO POSITIONS ".center(50, "="), "\n")
    print(f"You chose the dataset: {args.dataset_name}\n")

    if os.path.split(args.output)[0] == "":
        filepath = STATIC_ASSETS_DIR.joinpath(args.output)
        print(f"Outputting file {args.output} to path\n\t{filepath}\n")
    else:
        filepath = os.path.expandvars(args.output)
        print(f"Outputting to path\n\t{filepath}\n")

    dataset = load_validation_dataset(args.dataset_name)

    mapping = token_map(dataset)

    with open(f"{filepath}", "wb") as f:
        pickle.dump(mapping, file=f)

    print(f"Token map saved to\n\t{filepath}\n")
    print("Sanity check ... ", end="")

    with open(f"{filepath}", "rb") as f:
        pickled = pickle.load(f)

    assert mapping == pickled
    print("completed.")
    print(" END ".center(50, "="))
