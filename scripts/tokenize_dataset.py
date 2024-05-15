#!/usr/bin/env python3
import argparse
import io
import os
from pathlib import Path

from datasets import Dataset, Features, Value, load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from delphi.dataset.tokenization import get_tokenized_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", allow_abbrev=False)

    parser.add_argument(
        "--in-repo-id",
        "-i",
        type=str,
        required=True,
        help="Text dataset from huggingface to tokenize",
    )
    parser.add_argument(
        "--feature",
        "-f",
        type=str,
        required=True,
        help="Name of the column containing text documents in the input dataset",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        required=True,
        help="Split of the dataset to be tokenized, supports slicing like 'train[:10%%]'",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=False,
        help="Local directory to save the resulting dataset",
    )
    parser.add_argument(
        "--out-repo-id",
        type=str,
        required=False,
        help="HF repo id to upload the resulting dataset",
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        type=str,
        required=True,
        help="HF repo id or local directory containing the tokenizer",
    )
    parser.add_argument(
        "--seq-len",
        "-l",
        type=int,
        required=True,
        help="Context size of the tokenized dataset as input of the model",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=50,
        help="Size of input into batched tokenization",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=200_000,
        help="Size of the parquet chunks uploaded to HuggingFace",
    )
    args = parser.parse_args()
    assert (
        args.out_repo_id or args.out_dir
    ), "You need to provide --out-repo-id or --out-dir"

    print(f"Loading dataset '{args.in_repo_id}'...")
    in_dataset_split = load_dataset(
        args.in_repo_id,
        split=args.split,
        features=Features({args.feature: Value("string")}),
    )
    assert isinstance(in_dataset_split, Dataset)
    print(f"Loading tokenizer from '{args.tokenizer}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tokenizer.bos_token_id is not None, "Tokenizer must have a bos_token_id"
    assert tokenizer.eos_token_id is not None, "Tokenizer must have a eos_token_id"

    api = None
    if args.out_repo_id:
        api = HfApi()
        api.create_repo(repo_id=args.out_repo_id, repo_type="dataset", exist_ok=True)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    ds_chunks_it = get_tokenized_chunks(
        dataset_split=in_dataset_split,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

    print(f"Tokenizing split='{args.split}'...")
    split_name = args.split.split("[")[0]
    for chunk_idx, ds_chunk in enumerate(ds_chunks_it):
        chunk_name = f"{split_name}-{chunk_idx:05}.parquet"
        if args.out_dir:
            ds_parquet_chunk = Path(args.out_dir) / chunk_name
            print(f"Saving '{ds_parquet_chunk}'...")
        else:
            ds_parquet_chunk = io.BytesIO()
        ds_chunk.to_parquet(ds_parquet_chunk)
        if api:
            print(f"Uploading '{chunk_name}' to '{args.out_repo_id}'...")
            api.upload_file(
                path_or_fileobj=ds_parquet_chunk,
                path_in_repo=f"data/{chunk_name}",
                repo_id=args.out_repo_id,
                repo_type="dataset",
            )
        print(f"Done saving/uploading '{chunk_name}'")
