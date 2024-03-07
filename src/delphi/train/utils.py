import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from llama2c import model_export
from llama2c.model import ModelArgs as Llama2ModelArgs
from llama2c.model import Transformer as Llama2Model
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from delphi.train.gigaconfig import GigaConfig


def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)


def get_device() -> str:
    # cuda if available; else mps if apple silicon; else cpu
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@dataclass
class ModelMidTrain:
    # hack for packing the values touched by resume_model in a single object
    model: Llama2Model
    iter_num: int
    best_val_loss: float
    checkpoint: Any


def initialize_model(**model_args) -> Llama2Model:
    return Llama2Model(Llama2ModelArgs(**model_args))


def resume_model(resume_from_path: Path, device: str, **model_args) -> ModelMidTrain:
    ckpt_path = resume_from_path / "ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in [
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "vocab_size",
        "multiple_of",
        "max_seq_len",
    ]:
        model_args[k] = checkpoint_model_args[k]
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
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    return ModelMidTrain(
        model=model,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
        checkpoint=checkpoint,
    )


def get_optimizer(
    model: Llama2Model,
    config: GigaConfig,
    device: str,
    checkpoint=None,
) -> AdamW:
    device_type = "cuda" if "cuda" in device else "cpu"
    optimizer = model.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        (config.beta1, config.beta2),
        device_type,
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return optimizer


@torch.no_grad()
def estimate_loss(
    model: Llama2Model,
    eval_iters: int,
    batch_size: int,
    split_to_ds: dict[str, Dataset],
) -> dict[str, float]:
    """helps estimate an arbitrarily accurate loss over either split using many batches"""
    out = {}
    model.eval()
    for split, ds in split_to_ds.items():
        batch_iter = iter(DataLoader(ds, batch_size=batch_size))  # type: ignore
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(min(eval_iters, len(ds) // batch_size)):  # type: ignore
            X, Y = next(batch_iter)
            # forward pass, which will also compute the loss
            _logits = model(X, Y)
            loss = cast(Tensor, model.last_loss)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def set_lr(lr_decay_iters: int, config: GigaConfig, optimizer: AdamW, iter_num: int):
    lr = (
        get_lr(
            iter_num,
            config.warmup_iters,
            config.learning_rate,
            lr_decay_iters,
            config.min_lr,
        )
        if config.decay_lr
        else config.learning_rate
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


@dataclass
class EvalData:
    # values we expose to eval callback functions
    iter_num: int
    tokens_per_iter: int
    running_mfu: float
    lr: float
    losses: dict[str, float]
    best_val_loss: float
    new_best_val_loss: bool
    model: torch.nn.Module
    model_args: Any
    optimizer: torch.optim.Optimizer
    config: GigaConfig


def save_checkpoint_if_needed(eval_data: EvalData):
    if eval_data.iter_num == 0:
        return
    if not (eval_data.new_best_val_loss or eval_data.config.always_save_checkpoint):
        return
    checkpoint = {
        "model": eval_data.model.state_dict(),
        "optimizer": eval_data.optimizer.state_dict(),
        "model_args": eval_data.model_args,
        "iter_num": eval_data.iter_num,
        "best_val_loss": eval_data.best_val_loss,
        "config": asdict(eval_data.config),
    }
    print(f"saving checkpoint to {eval_data.config.out_dir}")
    torch.save(checkpoint, os.path.join(eval_data.config.out_dir, "ckpt.pt"))
    model_export(
        eval_data.model, os.path.join(eval_data.config.out_dir, "model.bin"), version=0
    )
