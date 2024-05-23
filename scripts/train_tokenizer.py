#!/usr/bin/env python3
import argparse

from datasets import Dataset, Features, Value
from tokenizers import ByteLevelBPETokenizer  # type: ignore
from transformers import PreTrainedTokenizerFast

from delphi import utils


def train_byte_level_bpe(
    dataset: Dataset, feature: str, vocab_size: int
) -> PreTrainedTokenizerFast:
    tokenizer = ByteLevelBPETokenizer()
    text_generator = (example[feature] for example in dataset)  # type: ignore
    tokenizer.train_from_iterator(
        text_generator,
        vocab_size=vocab_size,
        special_tokens=["<bos>", "<eos>", "<pad>"],
        show_progress=True,
        length=len(dataset),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", allow_abbrev=False)

    parser.add_argument(
        "--in-repo-id",
        "-i",
        type=str,
        required=True,
        help="Input dataset",
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
        help="Split of the dataset to be used for tokenizer training, supports slicing like 'train[:10%%]'",
    )
    parser.add_argument(
        "--vocab-size",
        "-v",
        type=int,
        required=True,
        help="Vocabulary size of the tokenizer",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=False,
        help="Local directory to save the resulting tokenizer",
    )
    parser.add_argument(
        "--out-repo-id",
        type=str,
        required=False,
        help="HF repo id to upload the resulting tokenizer",
    )
    args = parser.parse_args()
    assert (
        args.out_repo_id or args.out_dir
    ), "You need to provide out_repo_id or out_dir"

    in_dataset_split = utils.load_dataset_split_string_feature(
        args.in_repo_id, args.split, args.feature
    )
    assert isinstance(in_dataset_split, Dataset)
    tokenizer = train_byte_level_bpe(
        dataset=in_dataset_split,
        feature=args.feature,
        vocab_size=args.vocab_size,
    )
    if args.out_dir:
        print(f"Saving tokenizer to '{args.out_dir}' directory...")
        tokenizer.save_pretrained(args.out_dir)
        print("Done.")
    if args.out_repo_id:
        print(f"Pushing tokenizer to HF repo '{args.out_repo_id}'...")
        tokenizer.push_to_hub(
            repo_id=args.out_repo_id,
        )
        print("Done.")
