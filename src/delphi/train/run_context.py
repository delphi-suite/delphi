# get contextual information about a training run

from dataclasses import dataclass

import torch


@dataclass
class RunContext:
    device: torch.device
    torch_version: str
    delphi_version: str
    transformers_version: str
    os: str
