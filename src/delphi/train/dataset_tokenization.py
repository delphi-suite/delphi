from collections import deque

from transformers import PreTrainedTokenizerBase


def extend_deque(
    dq: deque[int],
    context_size: int,
    text_stories: list[str],
    prompt_idx: int,
    tokenizer: PreTrainedTokenizerBase,
) -> int:
    """
    Extends the deque with tokenized text stories until the deque grows large
    enough to reach the context size, or until all text stories are processed.

    The usage of a deque here aims to save the memory as opposed to
    load all the stories and tokenize them at once.

    Args:
        dq (deque[int]): Deque to extend with tokenized tokens.
        context_size (int): Size of the context(input sequences).
        text_stories (list[str]): List of (untokenized) text stories to be tokenized.
        prompt_idx (int): Index of the current text story.
        tokenizer (PreTrainedTokenizerBase): Tokenizer to encode the text strings.

    Returns:
        int: Updated index in the text stories dataset.
    """
    while len(dq) < context_size and prompt_idx < len(text_stories):
        text_story = text_stories[prompt_idx]
        dq.extend(
            tokenizer.encode(text_story, add_special_tokens=False)
            + [tokenizer.eos_token_id]
        )
        prompt_idx += 1
    return prompt_idx


def make_new_samples(
    dq: deque[int], context_size: int, tokenizer: PreTrainedTokenizerBase
) -> list[list[int]]:
    """
    Generates new samples for training by creating sequences of tokens
    from the deque until the deque is empty.

    Note: the model is unable to use the last token in an input sequence,
    so we repeat this token in the next input sequence.

    Args:
        dq (deque[int]): Deque containing tokenized tokens.
        context_size (int): Size of the context (input sequences).
        tokenizer (PreTrainedTokenizerBase): Tokenizer to encode the text strings.

    Returns:
        list[list[int]]: List of token sequences of the same length(context_size).
    """

    samples = []
    while len(dq) >= context_size:
        sample = [tokenizer.bos_token_id]
        for _ in range(context_size - 1):  # peek at and not pop the last element
            sample.append(dq.popleft())
        sample.append(dq[0])
        samples.append(sample)
    return samples


def get_tokenized_batches(
    text_stories: list[str],
    tokenizer: PreTrainedTokenizerBase,
    context_size: int,
) -> list[list[int]]:
    """
    Tokenizes the input text stories using the provided tokenizer and
    generates token sequences of the specified length.

    Args:
        text_stories (list[str]): List of text stories to be tokenized.
        tokenizer (PreTrainedTokenizerBase): Tokenizer to encode the text strings.
        context_size (int): Size of the context (input sequences).

    Returns:
        list[list[int]]: List of token sequences of length equal to context_size.
    """

    dq = deque()
    prompt_idx = 0
    samples = []

    while prompt_idx < len(text_stories):
        prompt_idx = extend_deque(dq, context_size, text_stories, prompt_idx, tokenizer)
        samples.extend(make_new_samples(dq, context_size, tokenizer))

    # We discard the last chunk, so no processing on the remainder of the deque here
    return samples
