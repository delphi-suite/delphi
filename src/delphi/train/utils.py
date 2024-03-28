import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, cast

import datasets
import safetensors.torch as st
import torch
import transformers
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from torch.optim import AdamW
from transformers import PreTrainedModel

from .config import GigaConfig
from .run_context import RunContext
from .shuffle import shuffle_list


@dataclass
class ModelTrainingState:
    """mutable training state - stuff that changes over the course of training"""

    model: PreTrainedModel
    optimizer: torch.optim.Optimizer
    iter_num: int = field(
        metadata={"help": "total iterations so far across all epochs"}
    )
    last_training_step_time: float = field(
        metadata={"help": "time last iteration ended"}
    )
    epoch: int = field(metadata={"help": "current epoch"})
    step: int = field(metadata={"help": "step within current epoch"})
    lr: float = field(default=1.0e-5, metadata={"help": "learning rate"})
    train_loss: float = field(
        default=0.0, metadata={"help": "loss on most recent train step"}
    )


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


def initialize_model_training_state(
    config: GigaConfig, device: torch.device
) -> ModelTrainingState:
    t0 = time.time()
    model = get_model(config.model_config)
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
    elif config.init_from == "resume":
        logging.info(f"Resuming training from {config.resume_from_path}")
        st.load_model(
            model, os.path.join(config.resume_from_path, "model", "model.safetensors")
        )
        with open(
            os.path.join(config.resume_from_path, "training_state.json"), "r"
        ) as f:
            training_state_vals = json.load(f)
        opt_state_dict_path = Path(os.path.join(config.resume_from_path, "opt.pt"))
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
        epoch=training_state_vals.get("epoch", 0),
        step=training_state_vals.get("step", 0),
    )


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
            revision=f"iter_{train_results.iter_num}",
        )


def load_tokens_dataset_from_huggingface(
    dataset: str,
    split: str,
    tokens_feature: str,
    limit: Optional[int] = None,
) -> Dataset:
    """Load a dataset from huggingface"""
    ds = cast(
        Dataset,
        load_dataset(
            dataset,
            split=split,
            features=datasets.Features(
                {tokens_feature: datasets.Sequence(datasets.Value("int32"))}
            ),
        ),
    )
    if limit is not None and limit > 0:
        ds = ds.select(range(limit))
    ds.set_format("torch")
    return ds


def count_tokens_so_far(config: GigaConfig, mts: ModelTrainingState) -> int:
    tokens_per_iter = (
        config.batch_size
        * config.optimizer.gradient_accumulation_steps
        * config.max_seq_len
    )

    return mts.iter_num * tokens_per_iter


def get_model(model_config_dict: dict[str, Any]) -> PreTrainedModel:
    """
    Get a model from a model config dictionary
    """
    model_class = getattr(transformers, model_config_dict["model_class"])
    config_class = cast(Type[transformers.PretrainedConfig], model_class.config_class)
    model_params_dict = model_config_dict.copy()
    model_params_dict.pop("model_class")
    return model_class(config_class(**(model_params_dict)))
