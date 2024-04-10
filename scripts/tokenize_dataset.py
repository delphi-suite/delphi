#!/usr/bin/env python3

import argparse
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from delphi.dataset.tokenization import tokenize_dataset
from delphi.eval.utils import load_validation_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--input-dataset-name",
        type=str,
        help="Text dataset from huggingface to tokenize",
    )
    parser.add_argument(
        "--output-dataset-name",
        type=str,
        help="Name of the tokenized dataset to upload to huggingface",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        help="Name of the tokenizer from huggingface",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=512,
        help="Context size of the tokenized dataset as input of the model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size of text inputs into the tokenizer",
    )
    parser.add_argument(
        "--column-name",
        type=str,
        help="Name of the column containing text documents in the input dataset",
    )
    args = parser.parse_args()

    input_dataset = load_dataset(args.input_dataset_name)
    input_dataset = cast(Dataset, input_dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    splits = list(input_dataset.keys())
    tokenized_datasets = {}  # dict that will hold tokenized vers. of each dataset split

    for i, split in enumerate(splits):
        print(f"Tokenizing {split = }", flush=True)
        text_docs = input_dataset[split]

        if args.column_name:
            text_docs = text_docs[args.column_name]
        else:
            if len(input_dataset.column_names) > 1:
                raise ValueError(
                    "There is more than one column in the specified dataset"
                )
            print(f"Using column {input_dataset.column_names[0]}")
            text_docs = input_dataset[input_dataset.column_names[0]]

        tokenized_dataset = tokenize_dataset(
            text_docs,
            tokenizer,
            context_size=args.context_size,
            batch_size=args.batch_size,
        )
        # Store the tokenized data in a new dataset for this split
        tokenized_datasets[split] = Dataset.from_dict({"tokens": tokenized_dataset})

    # Create a new dataset with the same structure (splits) as the original dataset, but with tokenized data
    output_dataset = DatasetDict(tokenized_datasets)

    print("Tokenizaton completed. Uploading dataset to Huggingface.", flush=True)

    output_dataset.push_to_hub(
        repo_id=args.output_dataset_name,
        private=False,
        token=args.token,
    )

    print("Done.", flush=True)
