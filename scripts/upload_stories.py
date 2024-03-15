import argparse
import json

import pandas as pd
from datasets import Dataset

"""
example call:

python upload_stories.py \
 --train ../train/llama2c/data/TinyStoriesV2-GPT4-train-clean.json \
 --validation ../train/llama2c/data/TinyStoriesV2-GPT4-valid-clean.json
"""


def get_args() -> argparse.Namespace:
    # define argparse parser with --train and --validation mandatory arguments, both paths to json files
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=str, required=True, help="Path to train json file"
    )
    parser.add_argument(
        "--validation", type=str, required=True, help="Path to validation json file"
    )
    return parser.parse_args()


def load_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def push_to_hub(splits):
    for filename, split in splits:
        stories = load_dataset(filename)
        dataset = Dataset.from_pandas(pd.DataFrame(stories))
        dataset.push_to_hub(repo_id="", split=split, token="")


if __name__ == "__main__":
    args = get_args()
    splits = {
        "train": args.train,
        "validation": args.validation,
    }
    push_to_hub(splits)
