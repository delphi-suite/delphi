import math

import pytest
import torch

from delphi.eval.token_position import (
    all_tok_pos_to_metrics_map,
    get_all_tok_pos_in_category,
)


@pytest.fixture
def token_mappings():
    labelled_tokens = {
        0: {
            "Is Noun": True,
            "Is Verb": False,
            "Is Adjective": False,
        },
        1: {
            "Is Noun": False,
            "Is Verb": True,
            "Is Adjective": False,
        },
        2: {
            "Is Noun": False,
            "Is Verb": False,
            "Is Adjective": True,
        },
        3: {
            "Is Noun": True,
            "Is Verb": False,
            "Is Adjective": False,
        },
        4: {
            "Is Noun": False,
            "Is Verb": True,
            "Is Adjective": False,
        },
        5: {
            "Is Noun": False,
            "Is Verb": False,
            "Is Adjective": True,
        },
        6: {
            "Is Noun": True,
            "Is Verb": False,
            "Is Adjective": False,
        },
        7: {
            "Is Noun": False,
            "Is Verb": True,
            "Is Adjective": False,
        },
        8: {
            "Is Noun": False,
            "Is Verb": False,
            "Is Adjective": True,
        },
    }

    tok_pos = {
        0: [],
        1: [(0, 0), (1, 0)],
        2: [(0, 1), (1, 1)],
        3: [(0, 2), (1, 2)],
        4: [(0, 3), (1, 3)],
        5: [(0, 4), (1, 4)],
        6: [(0, 5), (1, 5)],
        7: [(0, 6), (1, 6)],
        8: [(0, 7), (1, 7), (0, 8), (0, 9), (1, 7), (1, 8), (1, 9), (1, 10)],
    }

    raw_tok_ids = [
        torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 8, 8]).int(),
        torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8]).int(),
    ]

    raw_diff_probs = [
        torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8]),
        torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8]),
    ]

    return labelled_tokens, tok_pos, raw_tok_ids, raw_diff_probs


def test_get_all_tok_pos_in_category(token_mappings):
    labelled_tokens, tok_pos, _, _ = token_mappings
    noun_positions = get_all_tok_pos_in_category(labelled_tokens, "Is Noun", tok_pos)
    assert noun_positions == [(0, 2), (1, 2), (0, 5), (1, 5)]

    verb_positions = get_all_tok_pos_in_category(labelled_tokens, "Is Verb", tok_pos)
    assert verb_positions == [(0, 0), (1, 0), (0, 3), (1, 3), (0, 6), (1, 6)]

    adj_positions = get_all_tok_pos_in_category(
        labelled_tokens, "Is Adjective", tok_pos
    )
    # fmt: off
    assert adj_positions == [(0, 1), (1, 1), (0, 4), (1, 4), (0, 7), (1, 7), (0, 8), (0, 9), (1, 7), (1, 8), (1, 9), (1, 10)]
    # fmt: on

    with pytest.raises(KeyError):
        # for now this raises KeyError, might want to try and catch this
        get_all_tok_pos_in_category(labelled_tokens, "Is Adverb", tok_pos)


def test_all_tok_pos_to_metrics_map(token_mappings):
    labelled_tokens, tok_pos, raw_tok_ids, raw_diff_probs = token_mappings
    noun_positions = get_all_tok_pos_in_category(labelled_tokens, "Is Noun", tok_pos)
    noun_metrics = all_tok_pos_to_metrics_map(noun_positions, raw_diff_probs)
    # use math.isclose to compare floats
    for k, v in noun_metrics.items():
        assert math.isclose(v, raw_diff_probs[k[0]][k[1]].item())

    verb_positions = get_all_tok_pos_in_category(labelled_tokens, "Is Verb", tok_pos)
    verb_metrics = all_tok_pos_to_metrics_map(verb_positions, raw_diff_probs)
    for k, v in verb_metrics.items():
        assert math.isclose(v, raw_diff_probs[k[0]][k[1]].item())

    adj_positions = get_all_tok_pos_in_category(
        labelled_tokens, "Is Adjective", tok_pos
    )
    adj_metrics = all_tok_pos_to_metrics_map(
        adj_positions, raw_diff_probs, quantile_start=0.5, quantile_end=0.8
    )
    for k, v in adj_metrics.items():
        assert math.isclose(v, raw_diff_probs[k[0]][k[1]].item())
