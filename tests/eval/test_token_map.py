import pytest
from datasets import Dataset

from delphi.eval.token_map import token_map


def test_token_map():
    tokenized_dataset = Dataset.from_dict(
        {
            "tokens": [
                [0, 1, 2, 3, 4, 5, 0, 6, 7],
                [0, 1, 2, 3, 4, 5, 0, 6, 7],
                [0, 1, 2, 3, 4, 5, 0, 6, 7],
            ]
        }
    )
    mapping = token_map(tokenized_dataset)
    assert mapping == {
        0: [(0, 0), (0, 6), (1, 0), (1, 6), (2, 0), (2, 6)],
        1: [(0, 1), (1, 1), (2, 1)],
        2: [(0, 2), (1, 2), (2, 2)],
        3: [(0, 3), (1, 3), (2, 3)],
        4: [(0, 4), (1, 4), (2, 4)],
        5: [(0, 5), (1, 5), (2, 5)],
        6: [(0, 7), (1, 7), (2, 7)],
        7: [(0, 8), (1, 8), (2, 8)],
    }

    # fmt: off
    tokenized_dataset = Dataset.from_dict(
        { # one really long prompt
            "tokens": [
                [0, 1, 2, 3, 4, 5, 0, 6, 7, 0, 1, 2, 3, 4, 5, 0, 6, 7, 0, 1, 2, 3, 4, 5, 0, 6, 7]
            ]
        }
    )
    # fmt: on
    mapping = token_map(tokenized_dataset)
    assert mapping == {
        0: [(0, 0), (0, 6), (0, 9), (0, 15), (0, 18), (0, 24)],
        1: [(0, 1), (0, 10), (0, 19)],
        2: [(0, 2), (0, 11), (0, 20)],
        3: [(0, 3), (0, 12), (0, 21)],
        4: [(0, 4), (0, 13), (0, 22)],
        5: [(0, 5), (0, 14), (0, 23)],
        6: [(0, 7), (0, 16), (0, 25)],
        7: [(0, 8), (0, 17), (0, 26)],
    }
