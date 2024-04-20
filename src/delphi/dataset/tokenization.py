import io
import math
from collections import deque

from datasets import Dataset
from huggingface_hub import HfApi
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
        batch_size: The size of input into batched tokenization.
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


def tokenize_documents(
    documents: list[str],
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
) -> list[list[int]]:
    """
    Tokenizes the input text documents using the provided tokenizer and
    generates token sequences of the specified length.

    Args:
        text_documents: List[str],
        tokenizer,
        context_size,
        batch_size: The size of input into batched tokenization.

    Returns:
        list[list[int]]: List of token sequences of length equal to context_size.
    """
    assert tokenizer.bos_token_id is not None
    dq = deque()
    doc_idx = 0
    samples = []

    prog_bar = tqdm(total=len(documents), desc="Tokenizing text documents", leave=True)
    prev_doc_idx = 0
    # iterate through the text documents and tokenize them
    while doc_idx < len(documents):
        doc_idx = extend_deque(dq, seq_len, documents, doc_idx, tokenizer, batch_size)
        samples.extend(make_new_samples(dq, seq_len, tokenizer.bos_token_id))
        # update the tqdm bar
        prog_bar.update(doc_idx - prev_doc_idx)
        prev_doc_idx = doc_idx
    prog_bar.close()

    # We discard the last chunk, so no processing on the remainder of the deque here
    return samples


def tokenize_and_upload_split(
    dataset_split: Dataset,
    split_name: str,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
    chunk_size: int,
    out_repo_id: str,
    api: HfApi,
):
    print(f"Tokenizing {split_name=}...")
    documents = dataset_split[dataset_split.column_names[0]]
    tokenized_documents = tokenize_documents(
        documents,
        tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    print(f"Done, produced {len(tokenized_documents)} token sequences.")

    n_chunks = math.ceil(len(tokenized_documents) / chunk_size)
    print(f"Uploading {n_chunks} chunks to HuggingFace...")
    for chunk_idx in range(n_chunks):
        ds_chunk = Dataset.from_dict(
            {
                "tokens": tokenized_documents[
                    chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size
                ]
            }
        )

        ds_parquet_chunk = io.BytesIO()
        ds_chunk.to_parquet(ds_parquet_chunk)
        chunk_name = f"{split_name}-{chunk_idx:05}-of-{n_chunks-1:05}.parquet"
        api.upload_file(
            path_or_fileobj=ds_parquet_chunk,
            path_in_repo=f"data/{chunk_name}",
            repo_id=out_repo_id,
            repo_type="dataset",
        )
        print(f"Chunk {chunk_name} done.")
    print("Done.")
