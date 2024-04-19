#!/usr/bin/env python3

import argparse
import io
import math
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
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
        help="Size of input into batched tokenization",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200000,
        help="Size of the chunked datasets to upload to HuggingFace",
    )
    args = parser.parse_args()
    api = HfApi(token=args.hf_token)
    # api.create_repo(repo_id=args.output_dataset, repo_type="dataset")

    print(f"Loading dataset '{args.input_dataset}'...")
    input_dataset = load_dataset(args.input_dataset)
    input_dataset = cast(DatasetDict, input_dataset)
    print(f"Loading tokenizer '{args.tokenizer}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tokenizer.bos_token_id is not None, "Tokenizer must have a bos_token_id"
    assert tokenizer.eos_token_id is not None, "Tokenizer must have a eos_token_id"

    splits = list(input_dataset.keys())
    print(f"{splits=}")

    CHUNK_SIZE = args.chunk_size
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
        print(
            f"Dataset {split} split tokenization finished, length of {split} split: {len(tokenized_dataset)}. Starting to upload chunks to HF..."
        )

        n_chunks = math.ceil(len(tokenized_dataset) / CHUNK_SIZE)
        for chunk_idx in range(n_chunks):
            ds_chunk = Dataset.from_dict(
                {
                    "tokens": tokenized_dataset[
                        chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE
                    ]
                }
            )

            ds_parquet_chunk = io.BytesIO()
            ds_chunk.to_parquet(ds_parquet_chunk)
            api.upload_file(
                path_or_fileobj=ds_parquet_chunk,
                path_in_repo=f"data/{split}-{chunk_idx+1:05}-of-{n_chunks:05}.parquet",
                repo_id=args.output_dataset,
                repo_type="dataset",
            )

            print(
                f"Chunk {split}-{chunk_idx+1:05}-of-{n_chunks:05} uploaded to HuggingFace."
            )

    print("Done.", flush=True)
