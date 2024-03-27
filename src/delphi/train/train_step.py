import logging

import torch
from datasets import Dataset

from .config import GigaConfig
from .run_context import RunContext
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

    loss = torch.Tensor([0.0]).to(device)
    total_loss = loss.item()
    if config.debug_config.no_training:
        logging.debug("no_training set, skipping forward backward pass")
    else:
        for micro_step in range(config.optimizer.gradient_accumulation_steps):
            X, Y = get_xy_batch(
                dataset=train_ds,
                indices=indices,
                batch_size=config.batch_size,
                step=model_training_state.step,
                microstep=micro_step,
                gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps,
                device=device,
            )
            loss = (
                model(X, labels=Y, return_dict=True).loss
                / config.optimizer.gradient_accumulation_steps
            )
            total_loss += loss.item()
            loss.backward()
        # clip the gradient
        if config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)  # type: ignore
        optimizer.step()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
    model_training_state.train_loss = total_loss
