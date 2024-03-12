import torch
from beartype.typing import Mapping
from jaxtyping import Float


def quantile_range_metric_filter(
    metrics: Mapping[int, Float[torch.Tensor, "metric"]],
    q_start: float,
    q_end: float,
    sample: int = 0,
) -> Mapping[int, Float[torch.Tensor, "metric"]]:
    """
    Filter a dictionary of metrics to only include those within a specified quantile range.

    args:
    - metrics: a dictionary of token ids to a tensor of metrics (e.g. {0: torch.tensor([0.0, 0.1]), 1: torch.tensor([0.2, 0.3]), ...})
    - q_start: the start of the quantile range (e.g. 0.25 for the 25th percentile)
    - q_end: the end of the quantile range (e.g. 0.75 for the 75th percentile)
    - sample: how many samples to take from the metrics tensor AFTER filtering. If 0, take all samples. If > 0, take that many samples from the metrics tensor.

    returns:
    - a dictionary of token ids to a tensor of metrics that are within the specified quantile range
    """
    assert (
        q_start >= 0 and q_start <= 1
    ), f"Invalid q_start: {q_start}. Must be between 0 and 1"
    assert q_end >= 0 and q_end <= 1, f"Invalid q_end: {q_end}. Must be between 0 and 1"
    assert (
        q_start < q_end
    ), f"Invalid quantile range: {q_start} to {q_end}. Must be start < end"

    flattened_metrics = torch.cat([v for v in metrics.values()])
    q_start_val = torch.quantile(flattened_metrics, q_start)
    q_end_val = torch.quantile(flattened_metrics, q_end)

    result = {}
    for k, v in metrics.items():
        mask = (v >= q_start_val) & (v <= q_end_val)
        result[k] = v[mask]

    # TODO: sample from the result

    return result
