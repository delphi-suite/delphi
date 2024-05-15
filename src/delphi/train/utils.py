import json
import logging
import math
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Type, cast

import safetensors.torch as st
import torch
import transformers
from datasets import Dataset
from huggingface_hub import HfApi
from torch.optim import AdamW
from transformers import PreTrainedModel

from .config import TrainingConfig
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


def setup_determinism(seed: int):
    logging.debug(f"Setting up torch determinism (seed={seed})...")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


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
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
):
    """
    Set the learning rate (calculated by get_lr) on the optimizer
    """
    lr = (
        get_lr(
            iter_num=iter_num,
            warmup_iters=config.adam.warmup_iters,
            learning_rate=config.adam.learning_rate,
            lr_decay_iters=lr_decay_iters,
            min_lr=config.adam.min_lr,
        )
        if config.adam.decay_lr
        else config.adam.learning_rate
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def initialize_model_training_state(
    config: TrainingConfig, device: torch.device
) -> ModelTrainingState:
    t0 = time.time()
    model = init_model(config.model_config, seed=config.torch_seed)
    model.to(device)  # type: ignore
    optimizer = AdamW(
        lr=config.adam.learning_rate,
        params=model.parameters(),
        weight_decay=config.adam.weight_decay,
        betas=(config.adam.beta1, config.adam.beta2),
    )
    training_state_vals = dict()
    if config.resume_from_path is not None:
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
    indices = list(range(dataset_size))
    shuffle_list(indices, seed=ordering_seed + epoch)
    return indices


def gen_minibatches(
    dataset: Dataset,
    batch_size: int,
    num_minibatches: int,
    step: int,
    indices: list[int],
    device: torch.device,
    feature_name: str,
) -> Iterator[torch.Tensor]:
    """
    Generate minibatches from a dataset given a step and indices
    """
    minibatch_size = batch_size // num_minibatches
    first_minibatch_num = num_minibatches * step
    for batch_num in range(first_minibatch_num, first_minibatch_num + num_minibatches):
        start = batch_num * minibatch_size
        end = (batch_num + 1) * minibatch_size
        batch_indices = indices[start:end]
        yield dataset[batch_indices][feature_name].to(device)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    eval_iters: int,
    batch_size: int,
    split_to_ds: dict[str, Dataset],
    device: torch.device,
    epoch: int,
    feature_name: str,
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
        eval_iters = min(eval_iters, len(ds) // batch_size)
        losses = torch.zeros(eval_iters)  # keep on CPU
        minibatches = gen_minibatches(
            dataset=ds,
            batch_size=batch_size,
            num_minibatches=eval_iters,
            step=0,
            indices=indices,
            device=device,
            feature_name=feature_name,
        )
        for k, X in enumerate(minibatches):
            loss = model(X, labels=X, return_dict=True).loss
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_results(
    config: TrainingConfig,
    train_results: ModelTrainingState,
    run_context: RunContext,
    results_path: str,
    final: bool = False,
):
    """
    save results to disk, and to huggingface if configured to do so.

    Saves everything required to replicate the current state of training, including optimizer state,
    config, context (e.g. hardware), training step, etc
    """
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, "training_config.json"), "w") as file:
        json.dump(asdict(config), file, indent=2)
    model = train_results.model
    if isinstance(model, PreTrainedModel):
        model.save_pretrained(
            save_directory=results_path,
        )
    else:
        st.save_model(
            model,
            os.path.join(results_path, "model.safetensors"),
        )
    if config.save_optimizer:
        with open(os.path.join(results_path, "optimizer.pt"), "wb") as f:
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
        json.dump(run_context.asdict(), file, indent=2)
    if config.out_repo_id:
        api = HfApi()
        api.create_repo(config.out_repo_id, exist_ok=True)
        branch_name = f"iter{train_results.iter_num}"
        api.create_branch(config.out_repo_id, branch=branch_name)
        api.upload_folder(
            folder_path=results_path,
            repo_id=config.out_repo_id,
            revision=branch_name,
        )
        if final:
            api.upload_folder(
                folder_path=results_path,
                repo_id=config.out_repo_id,
                revision="main",
            )


def count_tokens_so_far(config: TrainingConfig, mts: ModelTrainingState) -> int:
    tokens_per_iter = config.batch_size * config.max_seq_len
    return mts.iter_num * tokens_per_iter


def init_model(model_config_dict: dict[str, Any], seed: int) -> PreTrainedModel:
    """
    Get a model from a model config dictionary
    """
    # reseed torch to ensure reproducible results in case other torch calls are different up to this point
    torch.random.manual_seed(seed)
    model_class = getattr(transformers, model_config_dict["model_class"])
    config_class = cast(Type[transformers.PretrainedConfig], model_class.config_class)
    model_params_dict = model_config_dict.copy()
    model_params_dict.pop("model_class")
    return model_class(config_class(**(model_params_dict)))
