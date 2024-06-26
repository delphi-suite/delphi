import collections
import random

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from delphi.tokenization import extend_deque, make_new_sample, tokenize_dataset


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("delphi-suite/stories-tokenizer")


def make_random_document(tokenizer):
    all_token_ids = range(2, tokenizer.vocab_size)
    n_tokens = random.randint(100, 800)
    random_tokens = random.choices(all_token_ids, k=n_tokens)
    return tokenizer.decode(random_tokens)


def get_random_feature_name():
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=10))


def test_extend_deque(tokenizer):
    CTX_SIZE = 10
    BATCH_SIZE = 2
    # generate 100 random stories
    documents = [make_random_document(tokenizer) for _ in range(100)]
    feature_name = get_random_feature_name()
    dataset = Dataset.from_dict({feature_name: documents})

    prompt_idx = 0
    deq = collections.deque()

    while prompt_idx < len(dataset):
        prompt_idx = extend_deque(
            deq, CTX_SIZE, dataset, prompt_idx, tokenizer, BATCH_SIZE
        )
        if prompt_idx < len(dataset) - 1:
            # assert that the deque has grown large enough in each round
            assert len(deq) >= CTX_SIZE
        while len(deq) >= CTX_SIZE:
            for _ in range(CTX_SIZE - 1):
                deq.popleft()


def test_make_new_sample(tokenizer):
    for _ in range(100):
        total_tokens = random.randint(100, 1000)
        context_size = random.randint(5, total_tokens // 2)
        dq = collections.deque(random.choices(range(3, 1000), k=total_tokens))
        samples = []
        while len(dq) >= context_size:
            samples.append(make_new_sample(dq, context_size, tokenizer.bos_token_id))
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
    SEQ_LEN = 11
    BATCH_SIZE = 2

    documents = [
        "Once upon a",
        "Mother woke up alert. She put on her coat",
        "Once upon a time, in a small town, there was a weird",
        "Once upon a time, there was a",
        "Sara and Tom are friends. They like to play in the park.",
    ]
    feature_name = get_random_feature_name()
    dataset = Dataset.from_dict({feature_name: documents})
    expected = [
        [0, 432, 441, 261, 1, 47, 500, 1946, 369, 3444, 16],
        [0, 16, 341, 577, 356, 338, 1888, 1, 432, 441, 261],
        [0, 261, 400, 14, 315, 261, 561, 1006, 14, 403, 285],
        [0, 285, 261, 2607, 1, 432, 441, 261, 400, 14, 403],
        [0, 403, 285, 261, 1, 1371, 269, 416, 485, 413, 16],
    ]
    actual = [x for x in tokenize_dataset(dataset, tokenizer, SEQ_LEN, BATCH_SIZE)]
    assert actual == expected
