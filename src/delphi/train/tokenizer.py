import io
import os
import tempfile
from typing import cast

from datasets import Dataset, load_dataset
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from tokenizers import SentencePieceBPETokenizer  # type: ignore
from transformers import LlamaTokenizerFast


def hf_dataset_to_text(
    dataset_name: str, split: str, column: str, text_file: io.TextIOBase
):
    dataset = cast(Dataset, load_dataset(dataset_name, split=split))
    for text in dataset[column]:
        text = text.strip()
        text_file.write(text + "\n")


def train_sentence_piece(
    vocab_size: int,
    sentence_iterator: io.TextIOBase,
) -> SentencePieceProcessor:
    """Trains a custom SentencePiece tokenizer."""
    model = io.BytesIO()
    SentencePieceTrainer.train(  # type: ignore
        sentence_iterator=sentence_iterator,
        model_writer=model,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity",
    )
    return SentencePieceProcessor(model_proto=model.getvalue())  # type: ignore


def sp_processor_to_hf_bpe_tokenizer(
    sp_processor: SentencePieceProcessor,
) -> SentencePieceBPETokenizer:
    """Converts a SentencePieceProcessor to a SentencePieceBPETokenizer."""
    vocab = {
        sp_processor.id_to_piece(index): index  # type: ignore
        for index in range(sp_processor.GetPieceSize())
    }
    merges = []
    for piece_l in vocab.keys():
        for piece_r in vocab.keys():
            merge = f"{piece_l}{piece_r}"
            piece_id = vocab.get(merge, None)
            if piece_id:
                merges += [(piece_l, piece_r, piece_id)]
    merges = sorted(merges, key=lambda val: val[2])
    merges = [(val[0], val[1]) for val in merges]

    return SentencePieceBPETokenizer(vocab, merges)


def hf_bpe_tokenizer_to_llama_tokenizer(
    hf_bpe_tokenizer: SentencePieceBPETokenizer,
) -> LlamaTokenizerFast:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as tmp_json_file:
        hf_bpe_tokenizer.save(tmp_json_file.name)
        return LlamaTokenizerFast(
            tokenizer_file=tmp_json_file.name,
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
