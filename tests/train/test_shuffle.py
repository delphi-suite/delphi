import random

import pytest

from delphi.train.shuffle import RNG, shuffle_epoch, shuffle_list


def test_rng():
    """
    Compare to the following C++ code:

    #include <iostream>
    #include <random>

    int main() {
        unsigned int seed = 12345;
        std::minstd_rand generator(seed);

        for (int i = 0; i < 5; i++)
            std::cout << generator() << ", ";
    }
    """
    rng = RNG(12345)
    expected = [595905495, 1558181227, 1498755989, 2021244883, 887213142]
    for val in expected:
        assert rng() == val


@pytest.mark.parametrize(
    "input_list, seed",
    [(random.sample(range(100), 10), random.randint(1, 1000)) for _ in range(5)],
)
def test_shuffle_list(input_list, seed):
    original_list = input_list.copy()
    shuffle_list(input_list, seed)
    assert sorted(input_list) == sorted(original_list)


@pytest.mark.parametrize(
    "seed, epoch_nr, expected",
    [
        (1, 1, [2, 5, 1, 3, 4]),
        (2, 5, [2, 1, 4, 5, 3]),
        (3, 10, [1, 4, 3, 5, 2]),
        (4, 100, [3, 4, 5, 1, 2]),
    ],
)
def test_shuffle_epoch(seed, epoch_nr, expected):
    samples = [1, 2, 3, 4, 5]
    shuffle_epoch(samples, seed, epoch_nr)
    assert samples == expected
