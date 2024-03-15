import logging
import time
from collections.abc import Callable, Generator

import torch
from datasets import Dataset

from delphi.constants import ModelTypes
from delphi.train.architectures import get_loss
from delphi.train.config.gigaconfig import GigaConfig
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
    train_batch_iter: Generator,
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
        f"gradient accumulation steps: {config.optimizer.gradient_accumulation_steps}, "
        f"num_steps: {iteration_params.num_steps}, iter_num: {model_training_state.iter_num}"
    )
    for micro_step in range(config.optimizer.gradient_accumulation_steps):
        X, Y = get_next_xy(train_batch_iter, device)
        loss = get_loss(model, X, Y) / config.optimizer.gradient_accumulation_steps
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
        lossf = loss.item() * config.optimizer.gradient_accumulation_steps
        if (
            model_training_state.local_iter_num >= 5
        ):  # let the training loop settle a bit
            mfu = estimate_mfu(
                config=config, model=model_training_state.model, timedelta=dt
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


def estimate_mfu(config: GigaConfig, model: torch.nn.Module, timedelta: float):
    """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
    # first estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    N = sum(p.numel() for p in model.parameters())
    if config.architecture == ModelTypes.LLAMA2HF:
        cfg = model.config
        L, H, Q, T = (
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.hidden_size // cfg.num_attention_heads,
            cfg.max_position_embeddings,
        )
    elif config.architecture == ModelTypes.MAMBA:
        logging.warn("MAMBA MFU estimate not implemented")
        return -1.0
    flops_per_token = 6 * N + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    fwdbwd_per_iter = config.batch_size * config.optimizer.gradient_accumulation_steps
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0 / timedelta)  # per second
    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu
