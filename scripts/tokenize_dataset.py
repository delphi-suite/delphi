#!/usr/bin/env python3
import argparse
import io
import os
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from delphi import utils
from delphi.tokenization import get_tokenized_chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize a text dataset using a specified tokenizer",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--in-dataset",
        "-i",
        type=str,
        required=True,
        help="Dataset you want to tokenize. Local path or HF repo id",
    )
    parser.add_argument(
        "--feature",
        "-f",
        type=str,
        required=True,
        help="Name of the feature (column) containing text documents in the input dataset",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        required=True,
        help="Split of the dataset to be tokenized, supports slicing like 'train[:10%%]'",
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
        help="Length of the tokenized sequences",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=50,
        help="How many text documents to tokenize at once (default: 50)",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=200_000,
        help="Maximum number of tokenized sequences in a single parquet file (default: 200_000)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=False,
        help="Local directory to save the resulting dataset",
    )
    parser.add_argument(
        "--out-repo",
        type=str,
        required=False,
        help="HF repo id to upload the resulting dataset",
    )
    args = parser.parse_args()
    assert args.out_repo or args.out_dir, "You need to provide --out-repo or --out-dir"

    in_dataset_split = utils.load_dataset_split_string_feature(
        args.in_dataset, args.split, args.feature
    )
    assert isinstance(in_dataset_split, Dataset)
    print(f"Loading tokenizer from '{args.tokenizer}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tokenizer.bos_token_id is not None, "Tokenizer must have a bos_token_id"
    assert tokenizer.eos_token_id is not None, "Tokenizer must have a eos_token_id"

    api = None
    if args.out_repo:
        api = HfApi()
        api.create_repo(repo_id=args.out_repo, repo_type="dataset", exist_ok=True)
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
    split_name = utils.hf_split_to_split_name(args.split)
    for chunk_idx, ds_chunk in enumerate(ds_chunks_it):
        chunk_name = f"{split_name}-{chunk_idx:05}.parquet"
        if args.out_dir:
            ds_parquet_chunk = Path(args.out_dir) / chunk_name
            print(f"Saving '{ds_parquet_chunk}'...")
        else:
            ds_parquet_chunk = io.BytesIO()
        ds_chunk.to_parquet(ds_parquet_chunk)
        if api:
            print(f"Uploading '{chunk_name}' to '{args.out_repo}'...")
            api.upload_file(
                path_or_fileobj=ds_parquet_chunk,
                path_in_repo=f"data/{chunk_name}",
                repo_id=args.out_repo,
                repo_type="dataset",
            )
        print(f"Done saving/uploading '{chunk_name}'")
