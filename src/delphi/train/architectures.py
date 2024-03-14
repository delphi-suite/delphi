import logging
from dataclasses import asdict
from typing import cast

import torch
from transformers import LlamaConfig as LlamaConfigHF
from transformers import LlamaForCausalLM as LlamaModelHF
from transformers import MambaConfig, MambaForCausalLM

from delphi.constants import ModelTypes
from delphi.train.config.gigaconfig import GigaConfig


def initialize_model(config: GigaConfig) -> torch.nn.Module:
    arch = config.architecture
    if arch == ModelTypes.LLAMA2HF:
        return LlamaModelHF(
            cast(
                LlamaConfigHF,
                LlamaConfigHF.from_dict(asdict(config.llama2hf_config)),
            )
        )
    elif arch == ModelTypes.MAMBA:
        return MambaForCausalLM(
            cast(
                MambaConfig,
                MambaConfig.from_dict(asdict(config.mamba_config)),
            )
        )
    else:
        raise NotImplementedError(f"Architecture {arch} not yet implemented")


def load_model(config: GigaConfig, checkpoint) -> torch.nn.Module:
    model = initialize_model(config)
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
