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
        "-o",
        "--output-dataset",
        type=str,
        help="Name of the tokenized dataset to upload to huggingface",
    )
    parser.add_argument(
        "-t",
        "--hf-token",
        type=str,
        help="Hugging Face API token",
    )
    args = parser.parse_args()
    api = HfApi(token=args.hf_token)
    api.create_repo(repo_id=args.output_dataset, repo_type="dataset")

    n_chunks = 0
    CHUNK_SIZE = 200000
    input_dataset = load_dataset("delphi-suite/stories")
    input_dataset = cast(DatasetDict, input_dataset)
    splits = list(input_dataset.keys())
    for i, split in enumerate(splits):
        tokenized_dataset = tokenize_dataset(
            input_dataset[split]["story"],
            AutoTokenizer.from_pretrained("delphi-suite/stories-tokenizer"),
            context_size=512,
            batch_size=50,
        )[:300001]
        print(
            f"Dataset split tokenization finished, length of dataset split: {len(tokenized_dataset)}. Starting to upload chunks to HF..."
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
