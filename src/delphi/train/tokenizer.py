import os
import sentencepiece as spm
import tempfile 

from datasets import Dataset


def train_vocab(
    vocab_size: int, 
    dataset: Dataset,
    column: str,
    cache_dir: str = "cache"
) -> None:
    """
    Trains a custom SentencePiece tokenizer.
    """
    assert vocab_size > 0, "Vocab size must be positive"
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as tmpfile:
        
        # export text as a single text file
        with open(tmpfile.name, 'w', encoding='utf-8') as file:
            for item in dataset:
                text = item[column]
                text = text.strip()
                file.write(text + '\n')
        print(f"Size is: {os.path.getsize(tmpfile.name) / 1024 / 1024:.2f} MB")

        # train the tokenizer
        prefix = os.path.join(cache_dir, f"tok{vocab_size}")
        spm.SentencePieceTrainer.train(
            input=tmpfile.name,
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
    print(f"Trained tokenizer is in {prefix}.model")
