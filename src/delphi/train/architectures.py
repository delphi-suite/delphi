from dataclasses import fields

import torch
from llama2c import model_export
from llama2c.model import ModelArgs as Llama2ModelArgs
from llama2c.model import Transformer as Llama2Model


class ModelTypes:
    LLAMA2C = "llama2c"
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
    ModelTypes.MAMBA: [
        "n_layers",
        "model_dim",
        "vocab_size",
    ],
}


def initialize_model(**model_args) -> torch.nn.Module:
    if model_args["architecture"] == ModelTypes.LLAMA2C:
        # filter model_args for fields in Llama2ModelArgs
        llama2_arg_names = {f.name for f in fields(Llama2ModelArgs)}
        llama2_args = {k: v for k, v in model_args.items() if k in llama2_arg_names}
        return Llama2Model(Llama2ModelArgs(**llama2_args))
    else:
        raise NotImplementedError(
            f"Architecture {model_args['architecture']} not yet implemented"
        )


def load_model(model_args, checkpoint) -> torch.nn.Module:
    arch = model_args["architecture"]
    checkpoint_model_args = checkpoint["model_args"]
    for k in args_to_load_from_checkpoint[arch]:
        model_args[k] = checkpoint_model_args[k]
    if arch == ModelTypes.LLAMA2C:
        # create the model
        gptconf = Llama2ModelArgs(**model_args)
        model = Llama2Model(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model
    else:
        raise NotImplementedError(f"Architecture {arch} not yet implemented")


def export_model(model, model_architecture, output_path):
    if model_architecture == ModelTypes.LLAMA2C:
        model_export(
            model,
            output_path,
            version=0,
        )
    else:
        raise NotImplementedError("only llama2c model export is supported for now")
