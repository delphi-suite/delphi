#!/usr/bin/env python3

import argparse
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from delphi.dataset.tokenization import tokenize_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i",
        "--input-dataset",
        type=str,
        help="Text dataset from huggingface to tokenize",
    )
    parser.add_argument(
        "--column-name",
        type=str,
        help="Name of the column containing text documents in the input dataset",
    )
    parser.add_argument(
        "-o",
        "--output-dataset",
        type=str,
        help="Name of the tokenized dataset to upload to huggingface",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name of the tokenizer from huggingface",
    )
    parser.add_argument(
        "--hf-token",
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
    args = parser.parse_args()

    print(f"Loading dataset '{args.input_dataset}'...")
    input_dataset = load_dataset(args.input_dataset)
    input_dataset = cast(DatasetDict, input_dataset)
    print(f"Loading tokenizer '{args.tokenizer}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tokenizer.bos_token_id is not None, "Tokenizer must have a bos_token_id"
    assert tokenizer.eos_token_id is not None, "Tokenizer must have a eos_token_id"

    splits = list(input_dataset.keys())
    tokenized_datasets = {}  # dict that will hold tokenized vers. of each dataset split
    print(f"{splits=}")

    for i, split in enumerate(splits):
        text_docs = input_dataset[split]
        assert (
            args.column_name or len(text_docs.column_names) == 1
        ), "--column-name required when dataset has multiple columns"
        column_name = args.column_name or text_docs.column_names[0]
        print(f"Tokenizing {split=} {column_name=}")
        tokenized_dataset = tokenize_dataset(
            text_docs[column_name],
            tokenizer,
            context_size=args.context_size,
            batch_size=args.batch_size,
        )
        # Store the tokenized data in a new dataset for this split
        tokenized_datasets[split] = Dataset.from_dict({"tokens": tokenized_dataset})

    # Create a new dataset with the same structure (splits) as the original dataset, but with tokenized data
    output_dataset = DatasetDict(tokenized_datasets)

    print("Tokenizaton completed. Uploading dataset to Huggingface.")

    output_dataset.push_to_hub(
        repo_id=args.output_dataset,
        private=False,
        token=args.hf_token,
    )

    print("Done.", flush=True)
