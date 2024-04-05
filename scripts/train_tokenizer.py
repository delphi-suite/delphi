import argparse
import os
import sentencepiece as spm
import tempfile

from datasets import load_dataset
from tqdm.auto import tqdm, trange
from transformers import LlamaTokenizerFast
from tokenizers import SentencePieceBPETokenizer

from delphi.train.tokenizer import train_vocab, get_tokenizer_model_path


def main(
    vocab_size: int,
    dataset_name: str,
    column: str,
    train_size: float,
    username: str,
    repo_id: str,
    token: str,
    seed: int,
    funct_test: bool = False,
):
    """
    Trains a SentencePiece tokenizer on a dataset.
    And uploads the resulting tokenizer to huggingface.
    Args:
    - vocab_size: The vocabulary size of the resulting tokenizer
    - dataset_name: The name of the dataset from which validation set will be loaded
    - train_size: The amount of the dataset that should be used for training
    - username: Hugging Face API username
    - repo_id: Hugging Face repository ID
    - token: Hugging Face API token
    """
    train_ds = load_dataset(dataset_name)["train"]
    if train_size < 1.0:
        train_ds = train_ds.train_test_split(
            train_size=train_size,
            seed=seed
        )["train"]
    
    assert vocab_size > 0, "vocab_size should be greater than 0"
    tokenizer_model_path = f"tok{vocab_size}.model"
    if not os.path.isfile(tokenizer_model_path):
        train_vocab(
            vocab_size=vocab_size,
            dataset=train_ds,
            column=column
        )
        
    sp_model = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
    
    # export 'vocab' and 'merges'
    vocab = {sp_model.id_to_piece(index): index for index in trange(sp_model.GetPieceSize())}
    merges = []
    for piece_l in tqdm(vocab.keys(), total=sp_model.GetPieceSize()):
        for piece_r in vocab.keys():
            merge = f"{piece_l}{piece_r}"
            piece_id = vocab.get(merge, None)
            if piece_id:
                merges += [(piece_l, piece_r, piece_id)]
    merges = sorted(merges, key=lambda val: val[2])
    merges = [(val[0], val[1]) for val in merges]
    
    # convert to BPE tokenizer
    bpe_tokenizer = SentencePieceBPETokenizer(vocab, merges)

    # Convert to LLaMA Tokenizer
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as tmpfile:
        bpe_tokenizer.save(tmpfile.name, pretty=True)
        tmpfile.seek(0)
        tokenizer = LlamaTokenizerFast(
            tokenizer_file=tmpfile.name,
            unk_token="<unk>",
            unk_token_id=0,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
            pad_token="<pad>",
            pad_token_id=3,
            padding_side="right",
        )
    print("Converted tokenizer to huggingface tokenizer.")
    
    # push tokenizer to the hub
    tokenizer.push_to_hub(
        repo_id=repo_id,
    )
    print("Pushed tokenizer to huggingface hub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer and convert to HF"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Vocabulary size of the tokenizer",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Dataset name with or without delphi-suite/ prefix",
    )
    parser.add_argument(
        "--column",
        type=str,
        help="Column of the dataset to be used for training",
    )
    parser.add_argument(
        "--train-size",
        help="Subset of the dataset to be used for training",
        default=1.0,
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Hugging Face API username",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        help="Hugging Face API username",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API token",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed",
    )
    parser.add_argument(
        "--test-funct", action="store_true", help="Enable test function mode"
    )

    args = parser.parse_args()

    main(
        args.vocab_size,
        args.dataset_name,
        args.train_size,
        args.column,
        args.username,
        args.repo_id,
        args.token,
        args.seed,
        args.test_funct,
    )
