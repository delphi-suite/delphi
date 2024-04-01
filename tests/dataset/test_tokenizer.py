import collections
import random

import pytest
from transformers import AutoTokenizer

from delphi.dataset.tokenization import extend_deque, make_new_samples, tokenize_dataset


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("delphi-suite/stories-tokenizer")


def test_extend_deque(tokenizer):
    CTX_SIZE = 10
    BATCH_SIZE = 2
    # generate 100 random stories
    text_stories = [
        " ".join(
            [
                tokenizer.decode(random.randint(3, tokenizer.vocab_size))
                for _ in range(random.randint(100, 800))
            ]
        )
        for _ in range(100)
    ]
    prompt_idx = 0
    dq = collections.deque()

    while prompt_idx < len(text_stories):
        prompt_idx = extend_deque(
            dq, CTX_SIZE, text_stories, prompt_idx, tokenizer, BATCH_SIZE
        )
        if prompt_idx < len(text_stories) - 1:
            # assert that the deque has grown large enough in each round
            assert len(dq) >= CTX_SIZE
        while len(dq) >= CTX_SIZE:
            for _ in range(CTX_SIZE - 1):
                dq.popleft()


def test_make_new_sample(tokenizer):
    for _ in range(100):
        total_tokens = random.randint(100, 1000)
        context_size = random.randint(5, total_tokens // 2)
        dq = collections.deque(random.choices(range(3, 1000), k=total_tokens))
        samples = make_new_samples(dq, context_size, tokenizer.bos_token_id)
        tokens_cnt = 0
        for i, sample in enumerate(samples):
            assert sample[0] == tokenizer.bos_token_id
            if i > 0:
                # assert that there is an overlap of the last token in the previous sample
                # and the first token in its following sample
                assert sample[1] == samples[i - 1][-1]
            tokens_cnt += len(sample)

        # We discard the last chunk so the following lines are only for testing
        tokens_cnt += 1 + len(dq)  # the last batch with BOS in the beginning
        assert tokens_cnt == total_tokens + (
            2 * len(samples) + 1
        )  # BOS for each batch + overlapping of the last tokens in the batches
        assert len(dq) > 0  # always leaving at least one element in the deque


def test_tokenize_dataset(tokenizer):
    CTX_SIZE = 10
    BATCH_SIZE = 2

    text_stories = [
        "Once upon a",
        "Mother woke up alert. She put on her coat",
        "Once upon a time, in a small town, there was a weird",
        "Once upon a time, there was a",
        "Sara and Tom are friends. They like to play in the park.",
    ]
    correct_batches = [
        [1, 432, 440, 261, 2, 367, 501, 1917, 372, 3398, 4037],
        [1, 4037, 341, 577, 359, 342, 1854, 2, 432, 440, 261],
        [1, 261, 403, 4045, 317, 261, 560, 1000, 4045, 406, 286],
        [1, 286, 261, 2567, 2, 432, 440, 261, 403, 4045, 406],
        [1, 406, 286, 261, 2, 787, 269, 396, 484, 415, 4037],
        [1, 4037, 311, 519, 268, 326, 317, 264, 525, 4037, 2],
    ]
    assert (
        tokenize_dataset(text_stories, tokenizer, CTX_SIZE, BATCH_SIZE)
        == correct_batches
    )
