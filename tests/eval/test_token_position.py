import pytest
import torch

from delphi.eval.token_position import (
    get_all_metrics_from_token_positions,
    get_all_token_positions_in_category,
    get_all_tokens_in_category,
)


@pytest.fixture
def token_mappings():
    token_labels = {
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
            "Is Adjective": True,
        },
    }

    token_positions = {
        0: [(0, 0), (0, 1)],
        1: [(0, 2), (0, 3)],
        2: [(0, 4), (0, 5)],
        3: [(0, 6), (0, 7)],
    }

    metrics = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        ]
    )

    return token_labels, token_positions, metrics


def test_get_all_tokens_in_category(token_mappings):
    token_labels, _, _ = token_mappings

    result = get_all_tokens_in_category(token_labels, "nouns")
    assert result == [0, 3]
    result = get_all_tokens_in_category(token_labels, "verbs")
    assert result == [1]
    result = get_all_tokens_in_category(token_labels, "adjectives")
    assert result == [2, 3]


def test_get_all_token_positions_in_category(token_mappings):
    token_labels, token_positions, _ = token_mappings

    result = get_all_token_positions_in_category(token_labels, token_positions, "nouns")
    assert result == {
        0: [(0, 0), (0, 1)],
        3: [(0, 6), (0, 7)],
    }
    result = get_all_token_positions_in_category(token_labels, token_positions, "verbs")
    assert result == {
        1: [(0, 2), (0, 3)],
    }
    result = get_all_token_positions_in_category(
        token_labels, token_positions, "adjectives"
    )
    assert result == {
        2: [(0, 4), (0, 5)],
        3: [(0, 6), (0, 7)],
    }


def test_get_all_metrics_from_token_positions(token_mappings):
    _, token_positions, metrics = token_mappings

    result = get_all_metrics_from_token_positions(token_positions, metrics)
    expected = {
        0: [0.0, 0.1],
        1: [0.2, 0.3],
        2: [0.4, 0.5],
        3: [0.6, 0.7],
    }

    for k, v in expected.items():
        # close approximation since we're dealing with floats
        assert torch.allclose(torch.tensor(v), torch.tensor(result[k]), atol=1e-6)
