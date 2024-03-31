import os

import sentencepiece as spm
from datasets import Dataset


def get_tokenizer_model_path(
    vocab_size: int,
    cache_dir: str = "cache"
) -> str:
    """
    Returns path to the SentencePiece tokenizer model for a given vocab size.
    """
    if vocab_size == 0:
        return ""
    else:
        return os.path.join(cache_dir, f"tok{vocab_size}.model")

def train_vocab(
    vocab_size: int, 
    dataset: Dataset,
    cache_dir: str = "cache"
) -> None:
    """
    Trains a custom SentencePiece tokenizer.
    """
    assert vocab_size > 0, "Vocab size must be positive"
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # export text as a single text file
    text_file = os.path.join(cache_dir, "text.txt")
    with open(text_file, 'w', encoding='utf-8') as file:
        for item in dataset:
            text = item['story']
            text = text.strip()
            file.write(text + '\n')
    print(f"Size is: {os.path.getsize(text_file) / 1024 / 1024:.2f} MB")

    # train the tokenizer
    prefix = os.path.join(cache_dir, f"tok{vocab_size}")
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity"
    )

    # optional cleanup of the text file
    dec = input(f"Delete the temporary file {text_file}? [y/N] ")
    if dec.lower() == "y":
        os.remove(text_file)
        print(f"Deleted {text_file}")

    print(f"Trained tokenizer is in {prefix}.model")
