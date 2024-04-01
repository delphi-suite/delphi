import logging
from dataclasses import dataclass

from datasets import Dataset

from .config import TrainingConfig


@dataclass
class IterationParams:
    num_batches: int
    num_steps: int
    eval_iters: int
    lr_decay_iters: int
    tokens_per_iter: int


def set_iteration_params(
    config: TrainingConfig, train_ds: Dataset, validation_ds: Dataset
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
    logging.info("Iteration setup:")
    logging.info(f"  batch size: {config.batch_size}")
    logging.info(f"  training set size: {len(train_ds)}")
    logging.info(f"  training batches: {num_batches}")
    logging.info(
        f"  gradient accumulations per step (=batches per step): {config.optimizer.gradient_accumulation_steps}"
    )
    logging.info(f"  steps per batch: {num_steps}")
    logging.info(f"  tokens per sequence: {config.max_seq_len}")
    logging.info(f"  tokens per training step will be: {tokens_per_iter:,}")
    logging.info(
        f"    breaks down as: {config.optimizer.gradient_accumulation_steps} grad accum steps * {config.batch_size} batch size * {config.max_seq_len} tokens per sequence"
    )
    logging.info(f"  validation set size: {len(validation_ds)}")
    logging.info(f"  batches per validation step: {eval_iters}")
    logging.info(
        f"  tokens per validation step: {eval_iters * config.batch_size * config.max_seq_len:,}"
    )
    return IterationParams(
        num_batches, num_steps, eval_iters, lr_decay_iters, tokens_per_iter
    )
