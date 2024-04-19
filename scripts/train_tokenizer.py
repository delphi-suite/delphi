#!/usr/bin/env python3
import argparse
import tempfile

from delphi.train.tokenizer import (
    hf_bpe_tokenizer_to_llama_tokenizer,
    hf_dataset_to_text,
    sp_processor_to_hf_bpe_tokenizer,
    train_sentence_piece,
)


def main(
    *,
    vocab_size: int,
    dataset_name: str,
    split: str,
    column: str,
    repo_id: str,
    hf_token: str,
):
    """Trains a SentencePiece tokenizer, converts it to LlamaTokenizerFast and pushes it to the Hugging Face Hub."""
    with tempfile.TemporaryFile(mode="w+") as text_file:
        print("Loading and writing dataset to text file...")
        hf_dataset_to_text(
            dataset_name=dataset_name,
            split=split,
            column=column,
            text_file=text_file,
        )
        text_file.seek(0)
        print("Training SentencePiece tokenizer...\n")
        sp_processor = train_sentence_piece(
            vocab_size=vocab_size,
            sentence_iterator=text_file,
        )
    print("\nConverting SentencePiece tokenizer Llama tokenizer...")
    hf_bpe_tokenizer = sp_processor_to_hf_bpe_tokenizer(sp_processor)
    llama_tokenizer = hf_bpe_tokenizer_to_llama_tokenizer(hf_bpe_tokenizer)
    print("Pushing Llama tokenizer to Hugging Face Hub...")
    llama_tokenizer.push_to_hub(
        repo_id=repo_id,
        token=hf_token,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer and convert to HF"
    )
    parser.add_argument(
        "--vocab-size",
        "-v",
        type=int,
        help="Vocabulary size of the tokenizer",
    )
    parser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        help="Dataset name with or without delphi-suite/ prefix",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="train",
        help="Split of the dataset to be used for training, supports slicing like 'train[:10%%]'",
    )
    parser.add_argument(
        "--column",
        "-c",
        type=str,
        help="Column of the dataset to be used for training",
    )
    parser.add_argument(
        "--repo-id",
        "-r",
        type=str,
        help="Hugging Face repository ID",
    )
    parser.add_argument(
        "--hf-token",
        "-t",
        type=str,
        help="Hugging Face API token",
    )
    args = parser.parse_args()
    main(
        vocab_size=args.vocab_size,
        dataset_name=args.dataset_name,
        split=args.split,
        column=args.column,
        repo_id=args.repo_id,
        hf_token=args.hf_token,
    )
