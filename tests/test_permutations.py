from delphi.train.permute import Permutation


def test_permute():

    perm = Permutation(1)

    # Test the random number generator
    for i in range(100000):
        assert perm.random_number_generator() >= 0
        assert perm.random_number_generator() <= 1
    state = perm.state
    assert state == 822144001
    # Test the permutation
    test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    length = len(test_list)
    for i in range(100000):
        perm.permute(test_list)
        assert len(test_list) == length
    assert test_list == [5, 6, 4, 10, 7, 2, 1, 3, 8, 9]
