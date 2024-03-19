import logging
from dataclasses import dataclass

from datasets import Dataset

from .config import GigaConfig


@dataclass
class IterationParams:
    num_batches: int
    num_steps: int
    eval_iters: int
    lr_decay_iters: int
    tokens_per_iter: int


def set_iteration_params(
    config: GigaConfig, train_ds: Dataset, validation_ds: Dataset
) -> IterationParams:
    num_batches = len(train_ds) // config.batch_size
    # we take gradient_accumulation_steps batches per step (one in each microstep)
    num_steps = num_batches // config.optimizer.gradient_accumulation_steps
    eval_iters = min(12, len(validation_ds) // config.batch_size)
    lr_decay_iters = (
        config.max_epochs * num_batches
    )  # should be ~=max_iters per Chinchilla
    tokens_per_iter = (
        config.optimizer.gradient_accumulation_steps
        * config.batch_size
        * config.max_seq_len
    )
    logging.debug(f"tokens per iteration will be: {tokens_per_iter:,}")
    logging.debug(
        f"breaks down as: {config.optimizer.gradient_accumulation_steps} grad accum steps * {config.batch_size} batch size * {config.max_seq_len} max seq len"
    )
    return IterationParams(
        num_batches, num_steps, eval_iters, lr_decay_iters, tokens_per_iter
    )
