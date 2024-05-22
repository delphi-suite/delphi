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
    selected_tokens = [2, 4, 6, 8]
    metrics = torch.tensor([[-1, 0.45, -0.33], [-1.31, 2.3, 0.6], [0.2, 0.8, 0.1]])
    return token_ids, selected_tokens, metrics


def test_get_all_tok_metrics_in_label(mock_data):
    token_ids, selected_tokens, metrics = mock_data
    result = get_all_tok_metrics_in_label(
        token_ids["tokens"],
        selected_tokens,
        metrics,
    )
    # key: (prompt_pos, tok_pos), value: logprob
    expected = {
        (0, 1): 0.45,
        (1, 0): -1.31,
        (1, 2): 0.6,
        (2, 1): 0.8,
    }

    # compare keys
    assert result.keys() == expected.keys()
    # compare values
    for k in result:
        assert isclose(cast(float, result[k]), expected[k], rel_tol=1e-6)  # type: ignore

    # test with quantile filtering
    result_q = get_all_tok_metrics_in_label(
        token_ids["tokens"], selected_tokens, metrics, q_start=0.6, q_end=1.0
    )
    expected_q = {
        (1, 2): 0.6,
        (2, 1): 0.8,
    }

    # compare keys
    assert result_q.keys() == expected_q.keys()
    # compare values
    for k in result_q:
        assert isclose(cast(float, result_q[k]), expected_q[k], rel_tol=1e-6)  # type: ignore
