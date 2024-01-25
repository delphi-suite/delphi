import pytest
import torch
from beartype.roar import BeartypeCallHintViolation

from delphi.dummy import dummy


def test_dummy():
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert torch.allclose(dummy(tensor1), torch.tensor([2.0, 3.0, 4.0]))
    assert torch.allclose(dummy(tensor2), torch.tensor([0.9, 1.9, 2.9]))
    tensor3 = torch.tensor([1, 2, 3])
    with pytest.raises(BeartypeCallHintViolation):
        dummy(tensor3)
