import json
import math
import os
import time
from collections.abc import Generator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader

from delphi import constants
from delphi.eval.utils import load_delphi_dataset
from delphi.train.architectures import (
    export_model,
    get_loss,
    initialize_model,
    load_model,
)
from delphi.train.config.gigaconfig import GigaConfig
from delphi.train.run_context import RunContext
from delphi.train.shuffle import shuffle_list


@dataclass
class ModelTrainingState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iter_num: int
    local_iter_num: int
    best_val_loss: float
    running_mfu: float
    t0: float
    epoch: int
    step: int
    lr: float = 1.0e-5


@dataclass
class EvalData:
    # values we expose to eval callback functions
    tokens_per_iter: int
    losses: dict[str, float]
    new_best_val_loss: bool
    config: GigaConfig
    model_training_state: ModelTrainingState


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


def get_optimizer(
    model: torch.nn.Module,
    config: GigaConfig,
    checkpoint=None,
) -> AdamW:
    optimizer = AdamW(
        lr=config.optimizer.learning_rate,
        params=model.parameters(),
        weight_decay=config.optimizer.weight_decay,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return optimizer


def get_lr(
    iter_num: int,
    warmup_iters: int,
    learning_rate: float,
    lr_decay_iters: int,
    min_lr: float,
):
    # 1) linear warmup for warmup_iters steps
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iter_num > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
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
            iter_num=iter_num,
            warmup_iters=config.optimizer.warmup_iters,
            learning_rate=config.optimizer.learning_rate,
            lr_decay_iters=lr_decay_iters,
            min_lr=config.optimizer.min_lr,
        )
        if config.optimizer.decay_lr
        else config.optimizer.learning_rate
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint_if_needed(eval_data: EvalData):
    mts = eval_data.model_training_state
    # we save if it's not the first iter AND at least one of:
    # 1) we have a new best validation loss
    # 2) always_save_checkpoint is set
    if mts.iter_num == 0:
        return
    if (not eval_data.new_best_val_loss) and (
        not eval_data.config.always_save_checkpoint
    ):
        return
    checkpoint = {
        "model": mts.model.state_dict(),
        "optimizer": mts.optimizer.state_dict(),
        "iter_num": mts.iter_num,
        "best_val_loss": mts.best_val_loss,
        "config": asdict(eval_data.config),
    }
    run_output_dir = get_run_output_dir(eval_data.config)
    print(f"saving checkpoint to {run_output_dir}")
    torch.save(checkpoint, os.path.join(run_output_dir, "ckpt.pt"))
    export_model(
        mts.model,
        eval_data.config.architecture,
        os.path.join(run_output_dir, "model.bin"),
    )


def load_model_training_state(
    config: GigaConfig, device: torch.device
) -> ModelTrainingState:
    iter_num = 0
    local_iter_num = 0
    best_val_loss = 1e9
    running_mfu = -1.0
    t0 = time.time()
    if config.init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        model = initialize_model(config.architecture, config.model_args)
        checkpoint = None
    # TODO: resume from huggingface model
    elif config.init_from == "resume":
        run_output_dir = get_run_output_dir(config)
        print(f"Resuming training from {run_output_dir}")
        checkpoint = torch.load(Path(run_output_dir) / "ckpt.pt", map_location=device)
        model = load_model(config, checkpoint)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    model.to(device)
    # optimizer
    optimizer = get_optimizer(
        model=model,
        config=config,
        checkpoint=checkpoint
        if checkpoint is not None and "optimizer" in checkpoint  # type: ignore
        else None,
    )
    epoch = checkpoint.get("epoch", 0) if checkpoint is not None else 0
    step = checkpoint.get("step", 0) if checkpoint is not None else 0
    checkpoint = None  # free up memory
    return ModelTrainingState(
        model=model,
        optimizer=optimizer,
        iter_num=iter_num,
        local_iter_num=local_iter_num,
        best_val_loss=best_val_loss,
        running_mfu=running_mfu,
        t0=t0,
        epoch=epoch,
        step=step,
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


def get_next_xy(
    train_batch_iter: Generator,
    device: torch.device
    # train_batch_iter: Generator[dict[str, list[int]], None, None], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    data = next(train_batch_iter).to(device)
    # X and Y NEED to be contigious. llama2c's implementation involves
    # calling .view on them, which breaks if they're not contigious
    X, Y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
    return X, Y


def batch_generator(
    dataset: Dataset, batch_size: int, epoch: int, ordering_seed: int
) -> Generator[torch.Tensor, None, None]:
    sampler = list(range(len(dataset)))  # type: ignore
    shuffle_list(sampler, seed=ordering_seed + epoch)
    sampler = torch.Tensor(sampler)
    for samples in sampler.split(batch_size):
        yield dataset[samples]["tokens"]


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    eval_iters: int,
    batch_size: int,
    split_to_ds: dict[str, Dataset],
    device: torch.device,
    model_arch: str,
) -> dict[str, float]:
    """helps estimate an arbitrarily accurate loss over either split using many batches"""
    out = {}
    model.eval()
    for split, ds in split_to_ds.items():
        # batch_iter = iter(DataLoader(ds, batch_size=batch_size))  # type: ignore
        batch_iter = iter(batch_generator(ds, batch_size, 0, 0))
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(min(eval_iters, len(ds) // batch_size)):  # type: ignore
            X, Y = get_next_xy(batch_iter, device)
            loss = get_loss(model, X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_run_output_dir(config: GigaConfig) -> str:
    return os.path.join(config.output_dir, config.run_name)


def save_results(
    config: GigaConfig,
    train_results: ModelTrainingState,
    run_context: RunContext,
    results_path: str,
):
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, "config.json"), "w") as file:
        json.dump(asdict(config), file)
    torch.save(train_results.model.state_dict(), os.path.join(results_path, "model.pt"))
    torch.save(
        train_results.optimizer.state_dict(), os.path.join(results_path, "opt.pt")
    )
    with open(os.path.join(results_path, "training_state.json"), "w") as file:
        training_state_dict = {
            "iter_num": train_results.iter_num,
            "local_iter_num": train_results.local_iter_num,
            "best_val_loss": train_results.best_val_loss,
            "running_mfu": train_results.running_mfu,
            "epoch": train_results.epoch,
            "lr": train_results.lr,
        }
        json.dump(training_state_dict, file, indent=2)
    with open(os.path.join(results_path, "run_context.json"), "w") as file:
        run_context_dict = asdict(run_context)
        run_context_dict["device"] = str(run_context.device)
        json.dump(run_context_dict, file, indent=2)
