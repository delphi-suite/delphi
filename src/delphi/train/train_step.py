import logging
from collections.abc import Iterable

import torch
from datasets import Dataset
from transformers import PreTrainedModel

from .config import GigaConfig
from .utils import ModelTrainingState, get_xy_batch


def train_step(
    model_training_state: ModelTrainingState,
    train_ds: Dataset,
    config: GigaConfig,
    device: torch.device,
    indices: list[int],
):
    """
    Runs a training step, updating (mutating in place) model_training_state
    """
    model = model_training_state.model
    optimizer = model_training_state.optimizer

    if config.debug_config.no_training:
        total_loss = 0.0
        logging.debug("no_training set, skipping forward backward pass")
    else:
        batches = (
            get_xy_batch(
                dataset=train_ds,
                indices=indices,
                batch_size=config.batch_size,
                step=model_training_state.step,
                microstep=microstep,
                gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps,
                device=device,
            )
            for microstep in range(config.optimizer.gradient_accumulation_steps)
        )
        total_loss = accumulate_gradients(
            model=model,
            batches=batches,
            num_batches=config.optimizer.gradient_accumulation_steps,
        )
        # clip the gradient
        if config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)  # type: ignore
        optimizer.step()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
    model_training_state.train_loss = total_loss


def accumulate_gradients(
    model: PreTrainedModel,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    num_batches: int,
) -> float:
    """
    Accumulate gradients over multiple batches as if they were a single batch
    """
    total_loss = 0.0
    for X, Y in batches:
        loss = model(X, labels=Y, return_dict=True).loss / num_batches
        total_loss += loss.item()
        loss.backward()
    return total_loss
