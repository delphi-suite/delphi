#!/usr/bin/env python3

import argparse
import io
import random

from datasets import Dataset
from huggingface_hub import HfApi

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

    splits = ["train", "validation"]

    api = HfApi(token=args.hf_token)
    api.create_repo(repo_id=args.output_dataset, repo_type="dataset")

    N_CHUNKS = 3
    CHUNK_SIZE = 5
    for i, split in enumerate(splits):
        for chunk in range(N_CHUNKS):
            ds_chunk = Dataset.from_dict(
                {
                    "tokens": [
                        [chunk] + random.sample(range(10), 5) for _ in range(CHUNK_SIZE)
                    ]
                }
            )
            ds_parquet_chunk = io.BytesIO()
            ds_chunk.to_parquet(ds_parquet_chunk)
            api.upload_file(
                path_or_fileobj=ds_parquet_chunk,
                path_in_repo=f"data/{split}-{chunk:05}-of-{N_CHUNKS:05}.parquet",
                repo_id=args.output_dataset,
                repo_type="dataset",
            )

    print("Done.", flush=True)
