import random
import string

import torch

from delphi.utils import gather_logprobs, hf_split_to_split_name


def random_string(length: int) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=length))


def test_hf_split_to_split_name():
    random_split_name = random_string(5)
    assert hf_split_to_split_name(random_split_name) == random_split_name
    assert hf_split_to_split_name(f"{random_split_name}[:10%]") == random_split_name
    assert hf_split_to_split_name(f"{random_split_name}[10%:]") == random_split_name
    assert hf_split_to_split_name(f"{random_split_name}[10%:20%]") == random_split_name
    assert hf_split_to_split_name(f"{random_split_name}[:200]") == random_split_name
    assert hf_split_to_split_name(f"{random_split_name}[200:]") == random_split_name
    assert hf_split_to_split_name(f"{random_split_name}[200:400]") == random_split_name


def test_gather_logprobs():
    # vocab size = 3
    logprobs = torch.tensor(
        [
            # batch 0
            [
                # seq 0
                [0.00, 0.01, 0.02],
                # seq 1
                [0.10, 0.11, 0.12],
            ],
            # batch 1
            [
                # seq 0
                [1.00, 1.01, 1.02],
                # seq 1
                [1.10, 1.11, 1.12],
            ],
        ]
    )
    tokens = torch.tensor(
        [
            # batch 0
            [0, 2],
            # batch 1
            [1, 2],
        ]
    )
    expected_output = torch.tensor(
        [
            # batch 0
            [0.00, 0.12],
            # batch 1
            [1.01, 1.12],
        ]
    )
    result = gather_logprobs(logprobs, tokens)
    assert torch.allclose(result, expected_output)
