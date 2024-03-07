import time

import torch

from delphi.train.utils import EvalData, ModelTrainingState, estimate_loss, set_lr


def train_step(
    model_training_state: ModelTrainingState,
    train_ds,
    validation_ds,
    iteration_params,
    eval_callbacks,
    config,
    train_batch_iter,
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
    # TODO: move lr to ModelTrainingState
    lr = set_lr(
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
        )
        new_best_val_loss = False
        if losses["val"] < model_training_state.best_val_loss:
            model_training_state.best_val_loss = float(losses["val"])
            new_best_val_loss = True
        # TODO: refactor EvalData to use ModelTrainingState
        eval_data = EvalData(
            iter_num=model_training_state.iter_num,
            tokens_per_iter=iteration_params.tokens_per_iter,
            running_mfu=model_training_state.running_mfu,
            lr=lr,
            losses=losses,
            best_val_loss=model_training_state.best_val_loss,
            new_best_val_loss=new_best_val_loss,
            model=model,
            model_args=model_training_state.model_args,
            optimizer=optimizer,
            config=config,
        )
        print(
            f"step {model_training_state.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        for callback in eval_callbacks:
            callback(eval_data)

    if model_training_state.iter_num == 0 and config.eval_only:
        return True

    # 3. forward backward update, with optional gradient accumulation to simulate larger batch size
    X, Y = next(train_batch_iter)
    print(
        f"gradient accumulation steps: {config.gradient_accumulation_steps}, "
        f"num_steps: {iteration_params.num_steps}, iter_num: {model_training_state.iter_num}"
    )
    for micro_step in range(
        min(
            config.gradient_accumulation_steps,
            iteration_params.num_steps - model_training_state.iter_num + 1,
        )
    ):
        logits = model(X, Y)
        loss = model.last_loss / config.gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
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
                f"{model_training_state.iter_num} | loss {lossf:.4f} | lr {lr:e} | "
                f"{dt*1000:.2f}ms | mfu {model_training_state.running_mfu*100:.2f}%"
            )
        )
    model_training_state.iter_num += 1
    model_training_state.local_iter_num += 1
    return False
