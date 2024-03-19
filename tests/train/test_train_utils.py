import pytest
import torch

from delphi.train.utils import get_device

#######################

# get_device


# Use pytest's monkeypatch fixture to patch `torch.cuda.is_available` and `torch.backends.mps.is_available`
def test_get_device_with_patched_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = get_device("auto")
    assert device.type == "cuda"


def test_get_device_with_patched_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    device = get_device("auto")
    assert device.type == "mps"


def test_get_device_with_patched_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    device = get_device("auto")
    assert device.type == "cpu"


@pytest.mark.parametrize(
    "device_type",
    ["cpu", "mps", "cuda"],
)
def test_get_device_auto(
    monkeypatch,
    device_type,
):
    monkeypatch.setattr(torch.cuda, "is_available", True)
    monkeypatch.setattr(torch.backends.mps, "is_available", True)
    device = get_device(device_type)
    assert device.type == device.type

#######################