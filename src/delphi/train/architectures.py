from typing import Any

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MambaConfig,
    MambaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

from delphi.constants import ModelTypes
from delphi.train.config.gigaconfig import GigaConfig
from delphi.train.config.model.model_config import ModelConfig, config_to_model


def _get_config_and_model_types(
    arch: str,
) -> tuple[type[PretrainedConfig], type[PreTrainedModel]]:
    if arch == ModelTypes.LLAMA:
        return (LlamaConfig, LlamaForCausalLM)
    elif arch == ModelTypes.MAMBA:
        return (MambaConfig, MambaForCausalLM)
    else:
        raise NotImplementedError(f"Architecture {arch} not yet implemented")


def initalize_model(config: ModelConfig) -> torch.nn.Module:
    return config_to_model(config)


def load_model(config: GigaConfig, checkpoint) -> torch.nn.Module:
    model = config_to_model(config.model_config)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    return model


# TODO: do we need this anymore?
def get_loss(model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Get loss of a model. Model should be a transformers-style *ForCausaLM model."""
    return model(X, labels=Y, return_dict=True).loss
