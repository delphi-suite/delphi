import logging
from typing import Any

import torch
from transformers import LlamaConfig as LlamaConfigHF
from transformers import LlamaForCausalLM as LlamaModelHF
from transformers import (
    MambaConfig,
    MambaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

from delphi.constants import ModelTypes
from delphi.train.config.gigaconfig import GigaConfig

try:
    from delphi.train.mamba import Mamba, MambaArgs
except Exception as e:
    print("no mamba for you")


def _get_config_and_model_types(
    arch: str,
) -> tuple[type[PretrainedConfig], type[PreTrainedModel]]:
    if arch == ModelTypes.LLAMA2HF:
        return (LlamaConfigHF, LlamaModelHF)
    elif arch == ModelTypes.MAMBA:
        return (MambaConfig, MambaForCausalLM)
    else:
        raise NotImplementedError(f"Architecture {arch} not yet implemented")


def initialize_model(arch: str, model_args: dict[str, Any]) -> torch.nn.Module:
    ConfigClass, ModelClass = _get_config_and_model_types(arch)
    return ModelClass(ConfigClass(**model_args))


def load_model(config: GigaConfig, checkpoint) -> torch.nn.Module:
    model = initialize_model(config.architecture, config.model_args)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    return model


# TODO: delete this?
def export_model(model, model_architecture, output_path):
    logging.warning(
        f"Architecture {model_architecture} model export not yet implemented"
    )


# TODO: do we need this anymore?
def get_loss(model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Get loss of a model. Model should be a transformers-style *ForCausaLM model."""
    return model(X, labels=Y, return_dict=True).loss
