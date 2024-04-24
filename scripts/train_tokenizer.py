#!/usr/bin/env python3
import argparse

from datasets import Dataset, Features, Value, load_dataset
from tokenizers import ByteLevelBPETokenizer  # type: ignore
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast


def train_byte_level_bpe(
    dataset: Dataset, feature: str, vocab_size: int
) -> PreTrainedTokenizerFast:
    tokenizer = ByteLevelBPETokenizer()
    text_generator = (example[feature] for example in dataset)  # type: ignore
    tokenizer.train_from_iterator(
        text_generator,
        vocab_size=vocab_size,
        special_tokens=["<bos>", "<eos>"],
        show_progress=True,
        length=len(dataset),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
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
        "--out-repo-id",
        "-o",
        type=str,
        required=True,
        help="Where to push the resulting tokenizer",
    )
    parser.add_argument(
        "--hf-token",
        "-t",
        type=str,
        help="Hugging Face API token",
    )
    args = parser.parse_args()

    print(f"Loading dataset '{args.in_repo_id}'...")
    in_dataset_split = load_dataset(
        args.in_repo_id,
        split=args.split,
        features=Features({args.feature: Value("string")}),
    )
    assert isinstance(in_dataset_split, Dataset)
    tokenizer = train_byte_level_bpe(
        dataset=in_dataset_split,
        feature=args.feature,
        vocab_size=args.vocab_size,
    )
    tokenizer.push_to_hub(
        repo_id=args.out_repo_id,
        token=args.hf_token,
    )
