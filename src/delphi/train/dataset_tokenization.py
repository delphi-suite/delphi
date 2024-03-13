from collections import deque

from transformers import PreTrainedTokenizerBase


def extend_deque(
    dq: deque[int],
    context_size: int,
    text_stories: list[str],
    prompt_idx: int,
    tokenizer: PreTrainedTokenizerBase,
) -> int:
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
    dq = deque()
    prompt_idx = 0
    samples = []

    while prompt_idx < len(text_stories):
        prompt_idx = extend_deque(dq, context_size, text_stories, prompt_idx, tokenizer)
        samples.extend(make_new_samples(dq, context_size, tokenizer))

    # We discard the last chunk, so no processing on the remainder of the deque here
    return samples
