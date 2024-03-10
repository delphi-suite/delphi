from collections import deque
from typing import Union

from transformers import PreTrainedTokenizerBase


def get_tokenized_batches(
    text_stories: Union[list[str], list[list[int]]],
    tokenizer: PreTrainedTokenizerBase,
    context_size: int,
    input_tokenized=False,
) -> list[list[int]]:
    dq = deque()
    samples = []

    prompt_idx = 0
    while prompt_idx < len(text_stories):
        while len(dq) < context_size:
            text_story = text_stories[prompt_idx]
            if not input_tokenized:
                dq.extend(
                    tokenizer.encode(text_story, add_special_tokens=False)
                    + [tokenizer.eos_token_id]
                )
            else:
                dq.extend(text_story)
                dq.append(tokenizer.eos_token_id)
            prompt_idx += 1

        sample = [tokenizer.bos_token_id]
        for i in range(context_size - 1):  # peek at and not pop the last element
            sample.append(dq.popleft())
        sample.append(dq[0])

        samples.append(sample)

    if dq:
        samples.append([tokenizer.bos_token_id] + list(dq))
    return samples
