"""

"""

import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from llama2c import model_export
from torch.utils.data import DataLoader
from tqdm import tqdm

from delphi import constants
from delphi.eval.utils import load_delphi_dataset
from delphi.train import wandb_utils
from delphi.train.gigaconfig import jai_config as config
from delphi.train.tokenized_chunks_dataset import TokenizedChunksDataset
from delphi.train.utils import (
    EvalData,
    estimate_loss,
    get_device,
    get_lr,
    get_optimizer,
    initialize_model,
    resume_model,
    save_checkpoint_if_needed,
    set_lr,
)

# system
device = get_device()

# load data
train_docs_ds = load_delphi_dataset(constants.TOKENIZED_CORPUS_DATASET, "train").select(
    range(256)  # TODO: remove when done debugging
)
validation_docs_ds = load_delphi_dataset(
    constants.TOKENIZED_CORPUS_DATASET, "validation"
)

train_ds = TokenizedChunksDataset(train_docs_ds, config.max_seq_len, device)
validation_ds = TokenizedChunksDataset(validation_docs_ds, config.max_seq_len, device)

train_ds.initialize_samples()
validation_ds.initialize_samples()

# fixing some hyperparams to sensible defaults

num_batches = len(train_ds) // config.batch_size
print(f"num_batches: {num_batches}")
num_steps = num_batches // config.gradient_accumulation_steps
config.eval_iters = min(12, len(validation_ds) // config.batch_size)

lr_decay_iters = (
    config.max_epochs * num_batches
)  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# validating checks
assert config.vocab_source in ["llama2", "custom"]
assert (
    config.vocab_source == "custom" or config.vocab_size == 32000
), "The vocab from Meta has 32K tokens"

# various inits, derived attributes, I/O setup
seed = 1337

tokens_per_iter = (
    config.gradient_accumulation_steps * config.batch_size * config.max_seq_len
)
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(
    f"breaks down as: {config.gradient_accumulation_steps} grad accum steps * {config.batch_size} batch size * {config.max_seq_len} max seq len"
)

os.makedirs(config.out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=config.dim,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    n_kv_heads=config.n_kv_heads,
    vocab_size=config.vocab_size,
    multiple_of=config.multiple_of,
    max_seq_len=config.max_seq_len,
    dropout=config.dropout,
)  # start with model_args from command line
if config.init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = initialize_model(**model_args)
    checkpoint = None
elif config.init_from == "resume":
    print(f"Resuming training from {config.out_dir}")
    model_mid_train = resume_model(Path(config.out_dir), device, **model_args)
    model = model_mid_train.model
    iter_num = model_mid_train.iter_num
    best_val_loss = model_mid_train.best_val_loss
    checkpoint = model_mid_train.checkpoint
model.to(device)


# optimizer
optimizer = get_optimizer(
    model=model,
    config=config,
    device=device,
    checkpoint=checkpoint
    if checkpoint is not None and "optimizer" in checkpoint
    else None,
)
checkpoint = None  # free up memory


eval_callbacks = [save_checkpoint_if_needed]
if config.wandb_log:
    wandb_utils.init_wandb(config)
    eval_callbacks.append(wandb_utils.log_to_wandb)


# training loop
t0 = time.time()


local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
epoch = 0


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
        if losses["val"] < best_val_loss or config.always_save_checkpoint:
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
        return True, None, None, None

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
    return False, t0, iter_num, local_iter_num


for epoch in range(config.max_epochs):
    train_ds.shuffle(epoch)
    train_batch_iter = iter(DataLoader(train_ds, batch_size=config.batch_size))  # type: ignore
    for _ in tqdm(range(num_steps)):
        breaknow, t0, iter_num, local_iter_num = train_step(
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
        )
        if breaknow:
            break
