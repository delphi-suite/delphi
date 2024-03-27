import json
import logging
import math
import os
import time
from collections.abc import Generator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import safetensors.torch as st
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from torch.optim import AdamW
from transformers import PreTrainedModel

from delphi import constants
from delphi.eval.utils import load_delphi_dataset

from .config import GigaConfig
from .run_context import RunContext
from .shuffle import shuffle_list


@dataclass
class ModelTrainingState:
    """mutable training state - stuff that changes over the course of training"""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iter_num: int = field(
        metadata={"help": "total iterations so far across all epochs"}
    )
    local_iter_num: int = field(
        metadata={"help": "total iterations on this instance so far"}
    )
    best_val_loss: float = field(metadata={"help": "best validation loss so far"})
    last_training_step_time: float = field(
        metadata={"help": "time last iteration ended"}
    )
    epoch: int = field(metadata={"help": "current epoch"})
    step: int = field(metadata={"help": "step within current epoch"})
    lr: float = field(default=1.0e-5, metadata={"help": "learning rate"})
    train_loss: float = field(
        default=0.0, metadata={"help": "loss on most recent train step"}
    )


@dataclass
class EvalData:
    # values we expose to eval callback functions
    tokens_per_iter: int
    losses: dict[str, float]
    new_best_val_loss: bool
    config: GigaConfig
    model_training_state: ModelTrainingState
    run_context: RunContext


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
    """
    Set the learning rate (calculated by get_lr) on the optimizer
    """
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
    results_path = os.path.join(eval_data.config.output_dir, f"iter_{mts.iter_num:06d}")
    logging.info(f"saving checkpoint to {results_path}")
    save_results(
        config=eval_data.config,
        train_results=mts,
        run_context=eval_data.run_context,
        results_path=results_path,
    )


def initialize_model_training_state(
    config: GigaConfig, device: torch.device
) -> ModelTrainingState:
    t0 = time.time()
    model = config.model_config.get_model()
    model.to(device)  # type: ignore
    optimizer = AdamW(
        lr=config.optimizer.learning_rate,
        params=model.parameters(),
        weight_decay=config.optimizer.weight_decay,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
    )
    training_state_vals = dict()
    if config.init_from == "scratch":
        logging.info(f"  initialized model and optimizer from scratch")
    # TODO: resume from huggingface model
    elif config.init_from == "resume":
        logging.info(f"Resuming training from {config.output_dir}")
        checkpoint = config.output_dir
        st.load_model(
            model, os.path.join(config.output_dir, "model", "model.safetensors")
        )
        with open(os.path.join(checkpoint, "training_state.json"), "r") as f:
            training_state_vals = json.load(f)
        opt_state_dict_path = Path(os.path.join(config.output_dir, "opt.pt"))
        if opt_state_dict_path.exists():
            with open(opt_state_dict_path, "rb") as f:
                logging.info("  Loading optimizer state from {state_dict_path}")
                optimizer.load_state_dict(torch.load(f))
    else:
        raise ValueError(
            f"{config.init_from} is not one of (scratch, resume), which are the two valid initialization methods. Unable to initialize model."
        )
    return ModelTrainingState(
        model=model,
        optimizer=optimizer,
        last_training_step_time=t0,
        iter_num=training_state_vals.get("iter_num", 0),
        local_iter_num=training_state_vals.get("local_iter_num", 0),
        best_val_loss=training_state_vals.get("best_val_loss", 1e9),
        epoch=training_state_vals.get("epoch", 0),
        step=training_state_vals.get("step", 0),
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


def get_indices_for_epoch(
    dataset_size: int, batch_size: int, epoch: int, ordering_seed: int
) -> list[int]:
    """ """
    num_indices = dataset_size // batch_size
    indices = list(range(num_indices))
    shuffle_list(indices, seed=ordering_seed + epoch)
    return indices


def get_xy_batch(
    batch_size: int,
    dataset: Dataset,
    indices: list[int],
    step: int,
    microstep: int,
    gradient_accumulation_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a batch of data from a dataset given a batch number and indices

    Imagine dataset is functionally split into batches of size batch_size. If batch_size=3, then
    the split each sample belongs to would go: [0, 0, 0, 1, 1, 1, 2, 2, 2, ... n_batches-1, n_batches-1, n_batches-1]
    We can refer to these splits by indices (0...n_batches-1), each of which is a contiguous chunk of size batch_size.
    At the start of each epoch, we make a list of indices (range(n_batches)) and shuffle it deterministically
    so that we get a different ordering of the dataset each epoch. Here, we want to get the split-of-size-batch_size
    corresponding to the current batch number within this batch.
    """
    batch_num = step * gradient_accumulation_steps + microstep
    index = indices[batch_num]
    start = index * batch_size
    end = (index + 1) * batch_size
    data = dataset[start:end]["tokens"].to(device)
    return data[:, :-1], data[:, 1:]


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    eval_iters: int,
    batch_size: int,
    split_to_ds: dict[str, Dataset],
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """helps estimate an arbitrarily accurate loss over either split using many batches"""
    out = {}
    model.eval()
    for split, ds in split_to_ds.items():
        indices = get_indices_for_epoch(
            dataset_size=len(ds),
            batch_size=batch_size,
            epoch=epoch,
            ordering_seed=1234,
        )
        num_losses = min(eval_iters, len(ds) // batch_size)
        losses = torch.zeros(num_losses)  # keep on CPU
        for k in range(num_losses):  # type: ignore
            X, Y = get_xy_batch(
                batch_size=batch_size,
                dataset=ds,
                indices=indices,
                step=k,
                microstep=0,
                gradient_accumulation_steps=1,
                device=device,
            )
            loss = model(X, labels=Y, return_dict=True).loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def save_results(
    config: GigaConfig,
    train_results: ModelTrainingState,
    run_context: RunContext,
    results_path: str,
):
    """
    save results to disk, and to huggingface if configured to do so.

    Saves everything required to replicate the current state of training, including optimizer state,
    config, context (e.g. hardware), training step, etc
    """
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, "config.json"), "w") as file:
        json.dump(asdict(config), file, indent=2)
    model = train_results.model
    if isinstance(model, PreTrainedModel):
        model = cast(PreTrainedModel, model)
        model.save_pretrained(
            save_directory=os.path.join(results_path, "model"),
        )
    else:
        st.save_model(
            train_results.model,
            os.path.join(results_path, "model", "model.safetensors"),
        )
    with open(os.path.join(results_path, "opt.pt"), "wb") as f:
        torch.save(train_results.optimizer.state_dict(), f)
    with open(os.path.join(results_path, "training_state.json"), "w") as file:
        training_state_dict = {
            "iter_num": train_results.iter_num,
            "local_iter_num": train_results.local_iter_num,
            "best_val_loss": train_results.best_val_loss,
            "lr": train_results.lr,
            "epoch": train_results.epoch,
            "step": train_results.step,
        }
        json.dump(training_state_dict, file, indent=2)
    with open(os.path.join(results_path, "run_context.json"), "w") as file:
        run_context_dict = asdict(run_context)
        run_context_dict["device"] = str(run_context.device)
        json.dump(run_context_dict, file, indent=2)
    if config.huggingface.push_checkpoints_to_hub:
        api = HfApi()
        api.upload_folder(
            folder_path=results_path,
            repo_id=str(config.huggingface.repo_id),
            path_in_repo=f"iter_{train_results.iter_num}/",
        )
