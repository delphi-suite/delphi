import itertools
from collections import deque
from collections.abc import Iterator

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


def extend_deque(
    deq: deque[int],
    context_size: int,
    dataset: Dataset,
    doc_idx: int,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> int:
    """
    Extends the deque with tokenized text documents until the deque grows large
    enough to reach the context size, or until all text documents are processed.

    The usage of a deque here aims to save the memory as opposed to
    load all the documents and tokenize them at once.

    Args:
        dq: Deque to extend with tokenized tokens.
        context_size: Size of the context(input sequences).
        text_documents: List of (untokenized) text documents to be tokenized.
        doc_idx: Index of the current text story.
        tokenizer: Tokenizer to encode the text strings.
        batch_size: The size of input into batched tokenization.
    Returns:
        int: Updated index in the text documents dataset.
    """
    feature = dataset.column_names[0]
    while len(deq) < context_size and doc_idx < len(dataset):
        documents = dataset[doc_idx : doc_idx + batch_size][feature]
        batch_input_ids = tokenizer(
            documents, return_attention_mask=False, add_special_tokens=False
        )["input_ids"]
        for input_ids in batch_input_ids:  # type: ignore
            deq.extend(input_ids + [tokenizer.eos_token_id])
        doc_idx += batch_size
    return doc_idx


def make_new_sample(deq: deque[int], seq_len: int, bos_token_id: int) -> list[int]:
    """
    Generates new sample for training by creating sequence of tokens
    from the deque until the deque.

    Note: the model is unable to use the last token in an input sequence,
    so we repeat this token in the next input sequence.

    Args:
        deq: Deque containing tokenized tokens.
        context_size: Size of the context (input sequences).
        bos_token_id: bos_token_id of the tokenizer used.

    Returns:
        list[int]: token sequence.
    """
    sample = [bos_token_id]
    # For the first n-2 elements, pop from the left of the deque
    # and add to the new sample, the (n-1)-th element will be retained
    # in the deque for making the next sample.
    for _ in range(seq_len - 2):
        sample.append(deq.popleft())
    sample.append(deq[0])
    return sample


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
) -> Iterator[list[int]]:
    """
    Tokenizes the input text documents using the provided tokenizer and
    generates token sequences of the specified length.

    Args:
        text_documents: List[str],
        tokenizer,
        context_size,
        batch_size: The size of input into batched tokenization.

    Returns:
        oken sequences of length equal to context_size.
    """
    assert tokenizer.bos_token_id is not None
    deq = deque()
    doc_idx = 0
    # iterate through the text documents and tokenize them
    while doc_idx < len(dataset):
        doc_idx = extend_deque(deq, seq_len, dataset, doc_idx, tokenizer, batch_size)
        yield make_new_sample(deq, seq_len, tokenizer.bos_token_id)
    # We discard the last chunk, so no processing on the remainder of the deque here


def get_tokenized_chunks(
    dataset_split: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
    chunk_size: int,
) -> Iterator[Dataset]:
    seq_it = tokenize_dataset(
        dataset_split,
        tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    while tokens_chunk := tuple(itertools.islice(seq_it, chunk_size)):
        yield Dataset.from_dict({"tokens": tokens_chunk})
