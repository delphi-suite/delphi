from delphi.train.permute import Random, permute


def test_permute():
    # Test the random number generator
    random = Random(1)
    for i in range(1000):
        assert random.random_uniform() >= 0
        assert random.random_uniform() <= 1
        assert random.random_number(10) >= 0
        assert random.random_number(10) <= 10
    state = random.state
    assert state == 3393775105
    # Test the permutation
    test_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    original_list = test_list.copy()
    length = len(test_list)
    for i in range(100000):
        permute(i, test_list)
        assert len(test_list) == length
        sorted_list = sorted(test_list)
        assert original_list == sorted_list
    assert test_list == [8, 5, 10, 9, 2, 7, 1, 6, 4, 3]
