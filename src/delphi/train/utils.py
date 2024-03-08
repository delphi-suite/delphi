import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from delphi import constants
from delphi.eval.utils import load_delphi_dataset
from delphi.train.architectures import export_model, initialize_model, load_model
from delphi.train.gigaconfig import GigaConfig


def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get torch device specified by device_str. May pass "auto" to set torch device automatically.
    """
    # cuda if available; else mps if apple silicon; else cpu
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    return torch.device(device_str)


@dataclass
class ModelMidTrain:
    # hack for packing the values touched by resume_model in a single object
    model: torch.nn.Module
    iter_num: int
    best_val_loss: float
    checkpoint: Any


def resume_model(
    resume_from_path: Path, device: torch.device, **model_args
) -> ModelMidTrain:
    ckpt_path = resume_from_path / "ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = load_model(model_args, checkpoint)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    return ModelMidTrain(
        model=model,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
        checkpoint=checkpoint,
    )


def get_optimizer(
    model: torch.nn.Module,
    config: GigaConfig,
    device: torch.device,
    checkpoint=None,
) -> AdamW:
    device_type = device.type
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
    model: torch.nn.Module,
    eval_iters: int,
    batch_size: int,
    split_to_ds: dict[str, Dataset],
    device: torch.device,
) -> dict[str, float]:
    """helps estimate an arbitrarily accurate loss over either split using many batches"""
    out = {}
    model.eval()
    for split, ds in split_to_ds.items():
        batch_iter = iter(DataLoader(ds, batch_size=batch_size))  # type: ignore
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(min(eval_iters, len(ds) // batch_size)):  # type: ignore
            X, Y = get_next_xy(batch_iter, device)
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


def set_lr(
    lr_decay_iters: int,
    config: GigaConfig,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
):
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
    # we save if it's not the first iter AND at least one of:
    # 1) we have a new best validation loss
    # 2) always_save_checkpoint is set
    if eval_data.iter_num == 0:
        return
    if (not eval_data.new_best_val_loss) and (
        not eval_data.config.always_save_checkpoint
    ):
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
    export_model(
        eval_data.model,
        eval_data.model_args["architecture"],
        os.path.join(eval_data.config.out_dir, "model.bin"),
    )


@dataclass
class ModelTrainingState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    model_args: Any
    iter_num: int
    local_iter_num: int
    best_val_loss: float
    running_mfu: float
    t0: float


def load_model_training_state(
    config: GigaConfig, device: torch.device
) -> ModelTrainingState:
    iter_num = 0
    local_iter_num = 0
    best_val_loss = 1e9
    running_mfu = -1.0
    t0 = time.time()
    model_args = dict(
        architecture=config.architecture,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        vocab_size=config.vocab_size,
        multiple_of=config.multiple_of,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    )  # start with model_args from command line
    if config.init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        model = initialize_model(**model_args)
        checkpoint = None
    # TODO: resume from huggingface model
    elif config.init_from == "resume":
        print(f"Resuming training from {config.out_dir}")
        model_mid_train = resume_model(Path(config.out_dir), device, **model_args)
        model = model_mid_train.model
        iter_num = model_mid_train.iter_num
        best_val_loss = model_mid_train.best_val_loss
        checkpoint = model_mid_train.checkpoint
    model.to(device)
    # optimizer
    optimizer = get_optimizer(
        model=model,
        config=config,
        device=device,
        checkpoint=checkpoint
        if checkpoint is not None and "optimizer" in checkpoint
        else None,
    )
    checkpoint = None  # free up memory
    return ModelTrainingState(
        model=model,
        optimizer=optimizer,
        model_args=model_args,
        iter_num=iter_num,
        local_iter_num=local_iter_num,
        best_val_loss=best_val_loss,
        running_mfu=running_mfu,
        t0=t0,
    )


def load_delphi_training_dataset(split: str, limit: int = -1):
    """For training, we want (X, Y) pairs, where X is a chunk of text and Y is the next token.)
    To construct this, we take the original tokenized dataset, break it into max_seq_len+1 length chunks,
    and then take [:-1] as X and [1:] as Y.
    """
    if limit == -1:
        ds = load_delphi_dataset(constants.TOKENIZED_CORPUS_DATASET, split)
    else:
        ds = load_delphi_dataset(constants.TOKENIZED_CORPUS_DATASET, split).select(
            range(limit)
        )
    ds.set_format("torch")
    return ds


def get_next_xy(train_batch_iter, device):
    data = cast(torch.Tensor, next(train_batch_iter)["tokens"].to(device))
    # X and Y NEED to be contigious. llama2c's implementation involves
    # calling .view on them, which breaks if they're not contigious
    X, Y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
    return X, Y
