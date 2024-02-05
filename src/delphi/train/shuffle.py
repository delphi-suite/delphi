class RNG:
    """Random Number Generator

    Linear Congruential Generator equivalent to minstd_rand in C++11
    https://en.cppreference.com/w/cpp/numeric/random
    """

    a = 48271
    m = 2147483647  # 2^31 - 1

    def __init__(self, seed: int):
        assert 0 <= seed < self.m
        self.state = seed

    def __call__(self) -> int:
        self.state = (self.state * self.a) % self.m
        return self.state


def shuffle_list(in_out: list, seed: int):
    """Deterministically shuffle a list in-place

    Implements Fisher-Yates shuffle with LCG as randomness source
    https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
    """
    rng = RNG(seed)
    n = len(in_out)
    for i in range(n - 1, 0, -1):
        j = rng() % (i + 1)
        in_out[i], in_out[j] = in_out[j], in_out[i]


def shuffle_epoch(samples: list, seed: int, epoch_nr: int):
    """Shuffle the samples in-place for a given training epoch"""
    rng = RNG(10_000 + seed)
    for _ in range(epoch_nr):
        rng()
    shuffle_seed = rng()
    shuffle_list(samples, shuffle_seed)
