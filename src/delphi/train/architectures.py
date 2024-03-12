import logging
from dataclasses import asdict, fields
from typing import cast

import torch
from llama2c import model_export
from llama2c.model import ModelArgs as Llama2ModelArgs
from llama2c.model import Transformer as Llama2cModel
from transformers import LlamaConfig as LlamaConfigHF
from transformers import LlamaForCausalLM as LlamaModelHF
from transformers import MambaConfig, MambaForCausalLM

from delphi.train.config.llama2_config_data import Llama2ConfigData


class ModelTypes:
    LLAMA2C = "llama2c"
    LLAMA2HF = "llama2-huggingface"
    MAMBA = "mamba"


args_to_load_from_checkpoint = {
    ModelTypes.LLAMA2C: [
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "vocab_size",
        "multiple_of",
        "max_seq_len",
    ],
    ModelTypes.LLAMA2HF: [f.name for f in fields(Llama2ConfigData)],
    ModelTypes.MAMBA: [
        "n_layers",
        "model_dim",
        "vocab_size",
    ],
}


# TODO: fix this once LLAMA2C is deprecated
def initialize_model(config) -> torch.nn.Module:
    arch = config.architecture
    if arch == ModelTypes.LLAMA2C:
        # filter model_args for fields in Llama2ModelArgs
        llama2_arg_names = {f.name for f in fields(Llama2ModelArgs)}
        llama2_args = {k: v for k, v in asdict(config).items() if k in llama2_arg_names}
        return Llama2cModel(Llama2ModelArgs(**llama2_args))  # type: ignore
    elif arch == ModelTypes.LLAMA2HF:
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


def load_model(model_args, checkpoint) -> torch.nn.Module:
    arch = model_args["architecture"]
    checkpoint_model_args = checkpoint["model_args"]
    for k in args_to_load_from_checkpoint[arch]:
        model_args[k] = checkpoint_model_args[k]
    if arch == ModelTypes.LLAMA2C:
        # create the model
        gptconf = Llama2ModelArgs(**model_args)
        model = Llama2cModel(gptconf)
    elif arch == ModelTypes.LLAMA2HF:
        model = initialize_model(**model_args)
    else:
        raise NotImplementedError(f"Architecture {arch} not yet implemented")
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    return model


def export_model(model, model_architecture, output_path):
    if model_architecture == ModelTypes.LLAMA2C:
        model_export(
            model,
            output_path,
            version=0,
        )
    else:
        logging.warning(
            f"Architecture {model_architecture} model export not yet implemented"
        )


def get_loss(
    model: torch.nn.Module, model_arch: str, X: torch.Tensor, Y: torch.Tensor
) -> torch.Tensor:
    if model_arch == ModelTypes.LLAMA2C:
        _logits = model(X, Y)
        loss = cast(torch.Tensor, model.last_loss)
    elif model_arch in (ModelTypes.LLAMA2HF, ModelTypes.MAMBA):
        loss = model(X, labels=Y, return_dict=True).loss
    else:
        raise NotImplementedError(f"Architecture {model_arch} loss not yet implemented")
    return loss
