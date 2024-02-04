# Number number generation utils without using any library


class Random:
    """
    Permutation class
    Atributes:
    - state: int, seed for the random number generator

    """

    def __init__(self, seed):
        self.state = seed

    def random_uniform(self) -> float:
        """
        Generate a random number between 0 and 1 using minstd_rand from C++11
        """
        self.state = (self.state * 48271) % 4294967296
        return self.state / 4294967296

    def random_number(self, max: int) -> int:
        """
        Generate a random number between 0 and max
        """
        return int(self.random_uniform() * max)


def permute(seed: int, list: list, epoch: int = 0) -> list:
    """
    Permute a list in place using Fisher-Yates shuffle
    """

    random = Random(seed)
    for i in range(len(list)):
        max_index = len(list) - i
        j = random.random_number(max_index) + i
        list[i], list[j] = list[j], list[i]
    return list
