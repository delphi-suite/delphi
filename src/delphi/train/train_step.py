import logging
import time
from collections.abc import Callable, Generator

import torch
from datasets import Dataset

from .config import GigaConfig
from .iteration_params import IterationParams
from .run_context import RunContext
from .utils import EvalData, ModelTrainingState, estimate_loss, get_next_xy, set_lr


def train_step(
    model_training_state: ModelTrainingState,
    train_ds: Dataset,
    validation_ds: Dataset,
    iteration_params: IterationParams,
    eval_callbacks: list[Callable],
    config: GigaConfig,
    train_batch_iter: Generator,
    run_context: RunContext,
):
    """
    Runs a training step, updating (mutating in place) model_training_state
    returns true if training should break, false otherwise
    """
    model = model_training_state.model
    optimizer = model_training_state.optimizer

    # here's how each train step works:
    # 1. Set learning rate
    # 2. (every eval_interval steps) evaluate, log to wandb, save checkpoint
    # 3. forward backward update
    # 4. log timing

    # 1. determine and set the learning rate for this iteration
    model_training_state.lr = set_lr(
        iteration_params.lr_decay_iters,
        config,
        optimizer,
        model_training_state.iter_num,
    )

    # 2. evaluate the loss on train/val sets and write checkpoints
    if model_training_state.iter_num % config.eval_interval == 0:
        if config.debug_config.no_eval:
            logging.debug("no_eval=True, skipping evaluation and using dummy losses")
            losses = {"train": 42.0, "val": 43.0}
        else:
            losses = estimate_loss(
                model=model,
                eval_iters=iteration_params.eval_iters,
                batch_size=config.batch_size,
                split_to_ds={"train": train_ds, "val": validation_ds},
                device=run_context.device,
                epoch=model_training_state.epoch,
            )
        new_best_val_loss = False
        if losses["val"] < model_training_state.best_val_loss:
            model_training_state.best_val_loss = float(losses["val"])
            new_best_val_loss = True
        eval_data = EvalData(
            tokens_per_iter=iteration_params.tokens_per_iter,
            losses=losses,
            new_best_val_loss=new_best_val_loss,
            config=config,
            model_training_state=model_training_state,
            run_context=run_context,
        )
        logging.info(
            f"step {model_training_state.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        for callback in eval_callbacks:
            callback(eval_data)

    # 3. forward backward update, with optional gradient accumulation to simulate larger batch size
    if config.debug_config.no_training:
        logging.debug("no_training set, skipping forward backward pass")
        loss = torch.Tensor([42.1]).to(run_context.device)
    else:
        for micro_step in range(config.optimizer.gradient_accumulation_steps):
            X, Y = get_next_xy(train_batch_iter, run_context.device)
            loss = (
                model(X, labels=Y, return_dict=True).loss
                / config.optimizer.gradient_accumulation_steps
            )
            loss.backward()
        # clip the gradient
        if config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)  # type: ignore
        optimizer.step()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

    # 4. log timing
    t1 = time.time()
    dt = t1 - model_training_state.last_training_step_time
    model_training_state.last_training_step_time = t1
    if model_training_state.iter_num % config.log_interval == 0:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * config.optimizer.gradient_accumulation_steps
        logging.debug(
            (
                f"{model_training_state.iter_num} | loss {lossf:.4f} | lr {model_training_state.lr:e} | "
                f"{dt*1000:.2f}ms"
            )
        )
    model_training_state.iter_num += 1
    model_training_state.local_iter_num += 1
