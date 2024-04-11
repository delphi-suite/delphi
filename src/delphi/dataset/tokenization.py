from collections import deque

from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase


def extend_deque(
    dq: deque[int],
    context_size: int,
    text_documents: list[str],
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
    Returns:
        int: Updated index in the text documents dataset.
    """
    while len(dq) < context_size and doc_idx < len(text_documents):
        text_doc = text_documents[doc_idx : doc_idx + batch_size]
        batch_input_ids = tokenizer(
            text_doc, return_attention_mask=False, add_special_tokens=False
        )["input_ids"]
        for input_ids in batch_input_ids:  # type: ignore
            dq.extend(input_ids + [tokenizer.eos_token_id])
        doc_idx += batch_size
    return doc_idx


def make_new_samples(
    dq: deque[int], context_size: int, bos_token_id: int
) -> list[list[int]]:
    """
    Generates new samples for training by creating sequences of tokens
    from the deque until the deque does not hold enough tokens to generate
    another sample.

    Note: the model is unable to use the last token in an input sequence,
    so we repeat this token in the next input sequence.

    Args:
        dq: Deque containing tokenized tokens.
        context_size: Size of the context (input sequences).
        bos_token_id: bos_token_id of the tokenizer used.

    Returns:
        list[list[int]]: List of token sequences of the same length(context_size).
    """

    samples = []
    while len(dq) >= context_size:
        sample = [bos_token_id]

        # For the first (n-1) elements, pop from the left of the deque
        # and add to the new sample, the n-th element will be retained
        # in the deque for making the next sample.
        for _ in range(context_size - 1):
            sample.append(dq.popleft())
        sample.append(dq[0])

        samples.append(sample)
    return samples


def tokenize_dataset(
    text_documents: list[str],
    tokenizer: PreTrainedTokenizerBase,
    context_size: int,
    batch_size: int,
) -> list[list[int]]:
    """
    Tokenizes the input text documents using the provided tokenizer and
    generates token sequences of the specified length.

    Args:
        text_documents: List[str],
        tokenizer,
        context_size,

    Returns:
        list[list[int]]: List of token sequences of length equal to context_size.
    """
    assert tokenizer.bos_token_id is not None
    dq = deque()
    doc_idx = 0
    samples = []

    pbar = tqdm(total=len(text_documents), desc="Tokenizing text documents", leave=True)
    prev_doc_idx = 0
    # iterate through the text documents and tokenize them
    while doc_idx < len(text_documents):
        doc_idx = extend_deque(
            dq, context_size, text_documents, doc_idx, tokenizer, batch_size
        )
        samples.extend(make_new_samples(dq, context_size, tokenizer.bos_token_id))
        # update the tqdm bar
        pbar.update(doc_idx - prev_doc_idx)
        prev_doc_idx = doc_idx

    pbar.close()

    # We discard the last chunk, so no processing on the remainder of the deque here
    return samples
