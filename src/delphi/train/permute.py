# Number number generation utils without using any library


class Permutation:
    """
    Permutation class
    Atributes:
    - state: int, seed for the random number generator

    """

    def __init__(self, seed):
        self.state = seed

    def set_seed(self, seed):
        """
        Set the seed
        """
        self.state = seed

    def permute(self, list):
        """
        Permute a list in place using Fisher-Yates shuffle
        """
        for i in range(len(list)):
            j = int(self.random_number_generator() * (i + 1))
            # self.state = self.state + 1
            list[i], list[j] = list[j], list[i]
        return list

    def random_number_generator(self):
        """
        Generate a random number between 0 and 1 using minstd_rand from C++11
        """
        self.state = (self.state * 48271) % 4294967296
        return self.state / 4294967296
