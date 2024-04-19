import logging
from collections.abc import Iterable

import torch
from datasets import Dataset
from transformers import PreTrainedModel

from .config import TrainingConfig
from .utils import ModelTrainingState, gen_minibatches


def train_step(
    model_training_state: ModelTrainingState,
    train_ds: Dataset,
    config: TrainingConfig,
    device: torch.device,
    ds_indices: list[int],
):
    """
    Runs a training step, updating (mutating in place) model_training_state:
    - generate gradient_accumulation_steps batches (each batch is batch_size/gradient_accumulation_steps items)
    - forward pass, accumulating gradient/gradient_accumulation_steps over gradient_accumulation_steps batches
    - clip gradient where gradient exceeds grad_clip (if configured)
    - backward pass, updating model weights
    - reset grad
    """
    model = model_training_state.model
    optimizer = model_training_state.optimizer

    if config.debug_config.no_training:
        total_loss = 0.0
        logging.debug("no_training set, skipping forward backward pass")
    else:
        minibatches = gen_minibatches(
            dataset=train_ds,
            indices=ds_indices,
            batch_size=config.batch_size,
            num_minibatches=config.gradient_accumulation_steps,
            step=model_training_state.step,
            device=device,
            feature_name=config.dataset.feature,
        )
        total_loss = accumulate_gradients(
            model=model,
            batches=minibatches,
            num_batches=config.gradient_accumulation_steps,
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
