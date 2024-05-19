import os

import torch
import transformers

import delphi


def get_auto_device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class RunContext:
    def __init__(self, device_str: str):
        if device_str == "auto":
            device_str = get_auto_device_str()
        self.device = torch.device(device_str)
        if self.device.type == "cuda":
            assert torch.cuda.is_available()
            self.gpu_name = torch.cuda.get_device_name(self.device)
        elif self.device.type == "mps":
            assert torch.backends.mps.is_available()
        self.torch_version = torch.__version__
        self.delphi_version = delphi.__version__
        self.transformers_version = transformers.__version__
        self.os = os.uname().version

    def asdict(self) -> dict:
        asdict = self.__dict__.copy()
        asdict["device"] = str(self.device)
        return asdict
