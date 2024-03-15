import torch
from jaxtyping import Float


def all_tok_pos_to_metrics_map(
    all_tok: list[tuple[int, int]],
    metrics: list[Float[torch.Tensor, "pos"]],
    quantile_start: float | None = None,
    quantile_end: float | None = None,
) -> dict[tuple[int, int], float]:
    """
    convert a list of all_tok and a list of metrics to a dict
    but only for the metrics that are in the quantile range
    if quantile_start and quantile_end are not None

    args:
    - all_tok: a list of token positions e.g. [(0, 0), (0, 1), (0, 2)]
    - metrics: a list of tensors of metrics e.g. [torch.tensor([0.0, 0.1]), torch.tensor([0.2, 0.3]), torch.tensor([0.4, 0.5])]
    - quantile_start: the start of the quantile range e.g. 0.25
    - quantile_end: the end of the quantile range e.g. 0.75

    return:
    - a dict with the token positions as keys and the metrics as values
    """
    if quantile_start is None or quantile_end is None:
        return {
            all_tok[i]: metrics[all_tok[i][0]][all_tok[i][1]].item()
            for i in range(len(all_tok))
        }

    # get a flattened list of all metrics from all_tok
    # TODO: see if you can estimate the quantiles by sampling from the metrics
    all_metrics = torch.Tensor([metrics[pos[0]][pos[1]].item() for pos in all_tok])
    # get the quantiles
    quantile_start_val = all_metrics.quantile(quantile_start)
    quantile_end_val = all_metrics.quantile(quantile_end)
    # return a dict with only the metrics that are in the quantile range
    return {
        all_tok[i]: metrics[all_tok[i][0]][all_tok[i][1]].item()
        for i in range(len(all_tok))
        if quantile_start_val <= all_metrics[i] <= quantile_end_val
    }


def get_all_tok_pos_in_category(
    labelled_tokens: dict[int, dict[str, bool]],
    category: str,
    tok_pos: dict[int, list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """
    get all token positions that are in a category

    args:
    - labelled_tokens: a dict with the token positions as keys and the token labels as values
        e.g. {0: {"Is Noun": True, "Is Verb": False, "Is Adjective": False}, ...}
    - category: the category to filter by e.g. "Is Noun"
    - tok_pos: a dict with the token positions as keys and the token positions as values
        e.g. {0: [(0, 0), (0, 1)], ...}

    return:
    - a list of token positions that are in the category
    """
    return [
        pos
        for i, token in labelled_tokens.items()
        if token[category]
        for pos in tok_pos[i]
    ]
