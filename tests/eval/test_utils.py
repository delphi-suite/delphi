import torch

from delphi.eval.utils import gather_logprobs


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
