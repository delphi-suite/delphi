import time

import torch
from beartype.typing import Callable
from datasets import Dataset
from torch.utils.data.dataloader import _BaseDataLoaderIter

from delphi.train.architectures import get_loss
from delphi.train.gigaconfig import GigaConfig
from delphi.train.iteration_params import IterationParams
from delphi.train.utils import (
    EvalData,
    ModelTrainingState,
    estimate_loss,
    get_next_xy,
    set_lr,
)


def train_step(
    model_training_state: ModelTrainingState,
    train_ds: Dataset,
    validation_ds: Dataset,
    iteration_params: IterationParams,
    eval_callbacks: list[Callable],
    config: GigaConfig,
    train_batch_iter: _BaseDataLoaderIter,
    device: torch.device,
) -> bool:
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
        losses = estimate_loss(
            model=model,
            eval_iters=iteration_params.eval_iters,
            batch_size=config.batch_size,
            split_to_ds={"train": train_ds, "val": validation_ds},
            device=device,
            model_arch=config.architecture,
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
        )
        print(
            f"step {model_training_state.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        for callback in eval_callbacks:
            callback(eval_data)

    if model_training_state.iter_num == 0 and config.eval_only:
        return True

    # 3. forward backward update, with optional gradient accumulation to simulate larger batch size
    print(
        f"gradient accumulation steps: {config.gradient_accumulation_steps}, "
        f"num_steps: {iteration_params.num_steps}, iter_num: {model_training_state.iter_num}"
    )
    for micro_step in range(config.gradient_accumulation_steps):
        X, Y = get_next_xy(train_batch_iter, device)
        loss = (
            get_loss(model, config.architecture, X, Y)
            / config.gradient_accumulation_steps
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
    dt = t1 - model_training_state.t0
    model_training_state.t0 = t1
    if model_training_state.iter_num % config.log_interval == 0:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * config.gradient_accumulation_steps
        if (
            model_training_state.local_iter_num >= 5
        ):  # let the training loop settle a bit
            mfu = model.estimate_mfu(
                config.batch_size * config.gradient_accumulation_steps, dt
            )
            model_training_state.running_mfu = (
                mfu
                if model_training_state.running_mfu == -1.0
                else 0.9 * model_training_state.running_mfu + 0.1 * mfu
            )
        print(
            (
                f"{model_training_state.iter_num} | loss {lossf:.4f} | lr {model_training_state.lr:e} | "
                f"{dt*1000:.2f}ms | mfu {model_training_state.running_mfu*100:.2f}%"
            )
        )
    model_training_state.iter_num += 1
    model_training_state.local_iter_num += 1
    return False
