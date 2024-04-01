from math import isclose
from typing import cast

import pytest
from datasets import Dataset

from delphi.eval.token_positions import *


@pytest.fixture
def mock_data():
    token_ids = Dataset.from_dict(
        {"tokens": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
    ).with_format("torch")
    token_labels = {
        1: {"Is Noun": False, "Is Verb": True},
        2: {"Is Noun": True, "Is Verb": True},
        3: {"Is Noun": False, "Is Verb": False},
        4: {"Is Noun": True, "Is Verb": False},
        5: {"Is Noun": False, "Is Verb": True},
        6: {"Is Noun": True, "Is Verb": True},
        7: {"Is Noun": False, "Is Verb": False},
        8: {"Is Noun": True, "Is Verb": False},
        9: {"Is Noun": False, "Is Verb": True},
    }
    metrics = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    return token_ids, token_labels, metrics


def test_get_all_tok_metrics_in_label(mock_data):
    token_ids, token_labels, metrics = mock_data
    result = get_all_tok_metrics_in_label(
        token_ids["tokens"], token_labels, metrics, "Is Noun"
    )
    expected = {
        (0, 1): 0.2,
        (1, 0): 0.4,
        (1, 2): 0.6,
        (2, 1): 0.8,
    }
    # use isclose to compare floating point numbers
    for k in result:
        assert isclose(cast(float, result[k]), expected[k], rel_tol=1e-6)  # type: ignore

    # test with quantile filtering
    result_q = get_all_tok_metrics_in_label(
        token_ids["tokens"], token_labels, metrics, "Is Noun", q_start=0.3, q_end=1.0
    )
    expected_q = {(1, 2): 0.6, (2, 1): 0.8, (1, 0): 0.4}
    for k in result_q:
        assert isclose(cast(float, result_q[k]), expected_q[k], rel_tol=1e-6)  # type: ignore
