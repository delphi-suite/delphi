from collections import defaultdict

import torch
from beartype.typing import Mapping
from jaxtyping import Float

from delphi.eval.constants import CATEGORY_MAP


def get_all_tokens_in_category(
    tlabelled_token_ids_dict: Mapping[int, dict[str, bool]], category: str
) -> list[int]:
    """
    Get all tokens in a category based on a pre-labelled token dictionary.

    args:
    - tlabelled_token_ids_dict: a dictionary of token ids to a dictionary of labels (e.g. {"Is Noun": True, "Is Verb": False, ...})
    - category: the category to filter by (e.g. "nouns", "verbs", "adjectives", "adverbs", "pronouns", "proper_nouns")

    returns:
    - a list of token ids that are in the specified category based on the labels in tlabelled_token_ids_dict
    """
    valid = ["nouns", "verbs", "adjectives", "adverbs", "pronouns", "proper_nouns"]
    assert category in valid, f"Invalid category: {category}. Must be one of {valid}"
    return [
        token_id
        for token_id, labels in tlabelled_token_ids_dict.items()
        if labels[CATEGORY_MAP[category]]
    ]


def get_all_token_positions_in_category(
    tlabelled_token_ids_dict: Mapping[int, dict[str, bool]],
    token_positions: Mapping[int, list[tuple[int, int]]],
    category: str,
) -> Mapping[int, list[tuple[int, int]]]:
    """
    Get all token positions in a category based on a pre-labelled token dictionary and a token mapping to (prompt, position) pairs.

    args:
    - tlabelled_token_ids_dict: a dictionary of token ids to a dictionary of labels (e.g. {"Is Noun": True, "Is Verb": False, ...})
    - token_positions: a dictionary of token ids to a list of (prompt, position) pairs (e.g. {0: [(0, 0), (0, 1)], 1: [(0, 2), (0, 3)], ...})
    - category: the category to filter by (e.g. "nouns", "verbs", "adjectives", "adverbs", "pronouns", "proper_nouns")

    returns:
    - a dictionary of token ids to a list of (prompt, position) pairs that are in the specified category based on the labels in tlabelled_token_ids_dict
    """
    valid = ["nouns", "verbs", "adjectives", "adverbs", "pronouns", "proper_nouns"]
    assert category in valid, f"Invalid category: {category}. Must be one of {valid}"
    all_tokens_in_category = get_all_tokens_in_category(
        tlabelled_token_ids_dict, category
    )
    return {
        token_id: token_positions[token_id]
        for token_id in all_tokens_in_category
        if token_id in token_positions
    }


def get_all_metrics_from_token_positions(
    token_positions: Mapping[int, list[tuple[int, int]]],
    metrics: Float[torch.Tensor, "prompt tok"],
) -> Mapping[int, Float[torch.Tensor, "metric"]]:
    """
    Get all metrics from token positions based on a dictionary of token ids to a list of (prompt, position) pairs and a tensor of metrics.

    args:
    - token_positions: a dictionary of token ids to a list of (prompt, position) pairs (e.g. {0: [(0, 0), (0, 1)], 1: [(0, 2), (0, 3)], ...})
    - metrics: a 2D tensor of metrics (e.g. torch.Tensor([[0.1, 0.2], [0.3, 0.4], ...]))

    returns:
    - a dictionary of token ids to a 1D tensor of metrics that are in the specified category based on the labels in tlabelled_token_ids_dict
    """

    result = {}
    for token_id, positions in token_positions.items():
        # ignore if positions is None
        if positions is None:
            continue
        tensor = torch.tensor(
            [metrics[prompt, position] for prompt, position in positions]
        )
        result[token_id] = tensor

    return result
