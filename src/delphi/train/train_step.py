import time

import torch

from delphi.train.utils import EvalData, estimate_loss, set_lr


def train_step(
    train_ds,
    validation_ds,
    lr_decay_iters,
    tokens_per_iter,
    iter_num,
    best_val_loss,
    model_args,
    model,
    optimizer,
    eval_callbacks,
    running_mfu,
    t0,
    local_iter_num,
    config,
    train_batch_iter,
    num_steps,
):
    # here's how each train step works:
    # 1. Set learning rate
    # 2. (every eval_interval steps) evaluate, log to wandb, save checkpoint
    # 3. forward backward update
    # 4. log timing

    # 1. determine and set the learning rate for this iteration
    lr = set_lr(lr_decay_iters, config, optimizer, iter_num)

    # 2. evaluate the loss on train/val sets and write checkpoints
    if iter_num % config.eval_interval == 0:
        losses = estimate_loss(
            model=model,
            eval_iters=config.eval_iters,
            batch_size=config.batch_size,
            split_to_ds={"train": train_ds, "val": validation_ds},
        )
        new_best_val_loss = False
        if losses["val"] < best_val_loss:
            best_val_loss = float(losses["val"])
            new_best_val_loss = True
        eval_data = EvalData(
            iter_num=iter_num,
            tokens_per_iter=tokens_per_iter,
            running_mfu=running_mfu,
            lr=lr,
            losses=losses,
            best_val_loss=best_val_loss,
            new_best_val_loss=new_best_val_loss,
            model=model,
            model_args=model_args,
            optimizer=optimizer,
            config=config,
        )
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        for callback in eval_callbacks:
            callback(eval_data)

    if iter_num == 0 and config.eval_only:
        return True, None, None, None, None

    # 3. forward backward update, with optional gradient accumulation to simulate larger batch size
    X, Y = next(train_batch_iter)
    print(
        f"gradient accumulation steps: {config.gradient_accumulation_steps}, num_steps: {num_steps}, iter_num: {iter_num}"
    )
    for micro_step in range(
        min(config.gradient_accumulation_steps, num_steps - iter_num + 1)
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
    dt = t1 - t0
    t0 = t1
    if iter_num % config.log_interval == 0:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * config.gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = model.estimate_mfu(
                config.batch_size * config.gradient_accumulation_steps, dt
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1
    return False, t0, iter_num, local_iter_num, best_val_loss
