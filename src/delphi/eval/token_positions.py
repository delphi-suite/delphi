from numbers import Number
from typing import Optional

import torch
from datasets import Dataset


def get_all_tok_metrics_in_label(
    token_ids: Dataset,
    token_labels: dict[int, dict[str, bool]],
    metrics: torch.Tensor,
    label: str,
    q_start: Optional[float] = None,
    q_end: Optional[float] = None,
) -> dict[tuple[int, int], Number]:
    """
    From the token_map, get all the positions of the tokens that have a certain label.
    We don't use the token_map because for sampling purposes, iterating through token_ids is more efficient.
    Optionally, filter the tokens based on the quantile range of the metrics.

    Args:
    - token_ids (Dataset): token_ids dataset e.g. token_ids[0] = {"tokens": [[1, 2, ...], [2, 5, ...], ...]}
    - token_labels (dict[str, bool]): dictionary of token labels e.g. { 0: {"Is Noun": True, "Is Verb": False}, ...}
    - metrics (torch.Tensor): tensor of metrics to search through e.g. torch.tensor([[0.1, 0.2, ...], [0.3, 0.4, ...], ...])
    - label (str): the label to search for
    - q_start (float): the start of the quantile range to filter the metrics e.g. 0.1
    - q_end (float): the end of the quantile range to filter the metrics e.g. 0.9

    Returns:
    - tok_positions (dict[tuple[int, int], Number]): dictionary of token positions and their corresponding metrics
    """

    # check if metrics have the same dimensions as token_ids
    if metrics.shape[0] != len(token_ids["tokens"]):
        raise ValueError(
            "The number of dimensions of the metrics tensor should be the same as the number of prompts in the token_ids dataset."
        )

    if q_start is not None and q_end is not None:
        start_quantile = torch.nanquantile(torch.flatten(metrics), q_start).item()
        end_quantile = torch.nanquantile(torch.flatten(metrics), q_end).item()

    tok_positions = {}
    for prompt_pos, prompt in enumerate(token_ids["tokens"]):
        for tok_pos, tok in enumerate(prompt):
            if (
                token_labels[tok][label] and tok_pos != 0
            ):  # ignore the first token since it's always a start token
                if q_start is not None and q_end is not None:
                    if (
                        metrics[prompt_pos, tok_pos].item() < start_quantile
                        or metrics[prompt_pos, tok_pos].item() > end_quantile
                    ):
                        continue

                tok_positions[(prompt_pos, tok_pos)] = metrics[
                    prompt_pos, tok_pos
                ].item()

    return tok_positions
