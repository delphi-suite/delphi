from math import isclose

import pytest
import torch

from delphi.eval.utils import (
    dict_filter_quantile,
    gather_logprobs,
    load_validation_dataset,
)


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


def test_load_validation_dataset():
    text = load_validation_dataset("tinystories-v2-clean")
    tokenized = load_validation_dataset("tinystories-v2-clean-tokenized-v0")


@pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)  # ignore warnings from numpy empty slice
def test_dict_filter_quantile():
    d = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}
    result = dict_filter_quantile(d, 0.2, 0.6)
    expected = {2: 0.2, 3: 0.3}

    # compare keys
    assert result.keys() == expected.keys()
    # compare values
    for k in result:
        assert isclose(result[k], expected[k], rel_tol=1e-6)

    # test with negative values
    d = {1: -0.1, 2: -0.2, 3: -0.3, 4: -0.4, 5: -0.5}
    result = dict_filter_quantile(d, 0.2, 0.6)
    expected = {3: -0.3, 4: -0.4}

    # compare keys
    assert result.keys() == expected.keys()
    # compare values
    for k in result:
        assert isclose(result[k], expected[k], rel_tol=1e-6)

    # test invalid quantile range
    with pytest.raises(ValueError):
        dict_filter_quantile(d, 0.6, 0.2)
    with pytest.raises(ValueError):
        dict_filter_quantile(d, 0.1, 1.1)
    with pytest.raises(ValueError):
        dict_filter_quantile(d, -0.1, 0.6)

    # test empty dict, will raise a warning
    result = dict_filter_quantile({}, 0.2, 0.6)
    assert result == {}
