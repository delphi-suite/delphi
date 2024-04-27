from delphi.utils import hf_split_to_split_name

from .utils import random_string


def test_hf_split_to_split_name():
    random_split_name = random_string(5)
    assert hf_split_to_split_name(random_split_name) == random_split_name
    assert hf_split_to_split_name(f"{random_split_name}[:10%]") == random_split_name
    assert hf_split_to_split_name(f"{random_split_name}[10%:]") == random_split_name
    assert hf_split_to_split_name(f"{random_split_name}[10%:20%]") == random_split_name
    assert hf_split_to_split_name(f"{random_split_name}[:200]") == random_split_name
    assert hf_split_to_split_name(f"{random_split_name}[200:]") == random_split_name
    assert hf_split_to_split_name(f"{random_split_name}[200:400]") == random_split_name
