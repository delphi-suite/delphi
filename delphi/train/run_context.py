import logging
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


def check_set_env_cublas_workspace_config():
    expected_val = ":4096:8"
    actual_val = os.getenv("CUBLAS_WORKSPACE_CONFIG")
    if actual_val is None:
        logging.info(
            f"Environment variable CUBLAS_WORKSPACE_CONFIG not set. Setting to '{expected_val}' to ensure reproducibility."
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_val
    else:
        correct_values = [expected_val, ":16:8"]
        assert actual_val in correct_values, (
            f"Environment variable CUBLAS_WORKSPACE_CONFIG is set to {actual_val}, which is incompatibe with reproducible training. "
            f"Please set it to one of the following values: {correct_values}. "
            f"See https://docs.nvidia.com/cuda/archive/12.4.0/cublas/index.html#results-reproducibility for more information."
        )


class RunContext:
    def __init__(self, device_str: str):
        if device_str == "auto":
            device_str = get_auto_device_str()
        self.device = torch.device(device_str)
        if self.device.type == "cuda":
            assert torch.cuda.is_available()
            check_set_env_cublas_workspace_config()
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
