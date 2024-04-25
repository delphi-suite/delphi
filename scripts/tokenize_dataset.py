#!/usr/bin/env python3
import argparse

from datasets import Dataset, Features, Value, load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from delphi.dataset.tokenization import tokenize_and_upload_split

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
        "--out-repo-id",
        "-o",
        type=str,
        required=True,
        help="Name of the tokenized dataset to upload to huggingface",
    )
    parser.add_argument(
        "--tokenizer",
        "-r",
        type=str,
        required=True,
        help="Name of the tokenizer from huggingface",
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

    print(f"Loading dataset '{args.in_repo_id}'...")
    in_dataset_split = load_dataset(
        args.in_repo_id,
        split=args.split,
        features=Features({args.feature: Value("string")}),
    )
    assert isinstance(in_dataset_split, Dataset)
    print(f"Loading tokenizer '{args.tokenizer}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    assert tokenizer.bos_token_id is not None, "Tokenizer must have a bos_token_id"
    assert tokenizer.eos_token_id is not None, "Tokenizer must have a eos_token_id"

    api = HfApi()
    api.create_repo(repo_id=args.out_repo_id, repo_type="dataset", exist_ok=True)
    tokenize_and_upload_split(
        dataset_split=in_dataset_split,
        split_name=args.split.split("[")[0],
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        out_repo_id=args.out_repo_id,
        api=api,
    )
