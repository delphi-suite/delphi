import pytest
import torch

from delphi.eval.quantile_range import quantile_range_metric_filter


@pytest.fixture
def metrics():
    return {
        0: torch.tensor([0.0, 0.1]),
        1: torch.tensor([0.2, 0.3]),
        2: torch.tensor([0.4, 0.5]),
        3: torch.tensor([0.6, 0.7]),
    }


def test_quantile_range_metric_filter(metrics):
    # first test for errors
    with pytest.raises(AssertionError):
        quantile_range_metric_filter(metrics, q_start=-0.1, q_end=0.5)

    with pytest.raises(AssertionError):
        quantile_range_metric_filter(metrics, q_start=0.1, q_end=1.5)

    with pytest.raises(AssertionError):
        quantile_range_metric_filter(metrics, q_start=0.5, q_end=0.1)

    # now test for the actual functionality
    result = quantile_range_metric_filter(metrics, q_start=0.25, q_end=0.75)
    expected = {
        0: torch.tensor([0.1]),
        1: torch.tensor([0.2, 0.3]),
        2: torch.tensor([0.4, 0.5]),
        3: torch.tensor([0.6]),
    }
    for k, v in result.items():
        assert torch.all(v == expected[k])
