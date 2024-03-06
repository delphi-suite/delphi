"""

"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import torch
from llama2c import Task, model_export
from llama2c.model import ModelArgs as Llama2ModelArgs
from llama2c.model import Transformer as Llama2Model
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from delphi import constants
from delphi.eval.utils import load_delphi_dataset
from delphi.train.tokenized_chunks_dataset import TokenizedChunksDataset
from delphi.train.utils import (
    estimate_loss,
    get_lr,
    get_optimizer,
    initialize_model,
    resume_model,
)

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'
# wandb logging
wandb_log = True  # disabled by default
# wandb_entity = "jannik-brinkmann"
wandb_entity = "jaiwithani"
wandb_project = "delphi"
wandb_run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data
batch_size = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 256
vocab_source = (
    "llama2"  # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
)
vocab_size = 32000  # the Llama 2 tokenizer has 32K tokens
# model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0
# adamw optimizer
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_epochs = 10  # total number of training epochs
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for


# Jai Overrides TODO: remove these
vocab_source = "custom"
vocab_size = 4096
max_seq_len = 512
dim = 48
n_layers = 2
n_heads = 2
n_kv_heads = 2
max_epochs = 2
eval_interval = 500
eval_iters = 10


# system
# TODO: when fixing all of this, also dynamically detect env
# and set the device accordingly (e.g. cuda when available, mps on macbooks, cpu otherwise)
device = (
    # "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    "mps"  # TODO: remove, this is for debugging on macbooks
)
dtype = "float32"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
# TODO: replace this with something sane
# exec(
#     open("./llama2c/configurator.py").read()
# )  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging


# -----------------------------------------------------------------------------

# load data
train_docs_ds = load_delphi_dataset(constants.TOKENIZED_CORPUS_DATASET, "train").select(
    range(256)
)
validation_docs_ds = load_delphi_dataset(
    constants.TOKENIZED_CORPUS_DATASET, "validation"
)

train_ds = TokenizedChunksDataset(train_docs_ds, max_seq_len, device)
validation_ds = TokenizedChunksDataset(validation_docs_ds, max_seq_len, device)

train_ds.initialize_samples()
validation_ds.initialize_samples()

# fixing some hyperparams to sensible defaults

num_batches = len(train_ds) // batch_size
print(f"num_batches: {num_batches}")
num_steps = num_batches // gradient_accumulation_steps
eval_iters = min(12, len(validation_ds) // batch_size)

lr_decay_iters = max_epochs * num_batches  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# validating checks
assert vocab_source in ["llama2", "custom"]
assert (
    vocab_source == "custom" or vocab_size == 32000
), "The vocab from Meta has 32K tokens"

# various inits, derived attributes, I/O setup
seed = 1337
tokens_per_iter = gradient_accumulation_steps * batch_size * max_seq_len
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(
    f"breaks down as: {gradient_accumulation_steps} grad accum steps * {batch_size} batch size * {max_seq_len} max seq len"
)

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = initialize_model(**model_args)
    checkpoint = None
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    model_mid_train = resume_model(Path(out_dir), device, **model_args)
    model = model_mid_train.model
    iter_num = model_mid_train.iter_num
    best_val_loss = model_mid_train.best_val_loss
    checkpoint = model_mid_train.checkpoint
model.to(device)


# optimizer
optimizer = get_optimizer(
    model=model,
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    beta_1=beta1,
    beta_2=beta2,
    device_type=device_type,
    checkpoint=checkpoint
    if checkpoint is not None and "optimizer" in checkpoint
    else None,
)
checkpoint = None  # free up memory

# wrap model into DDP container


# logging
if wandb_log:
    import wandb

    wandb.init(
        entity=wandb_entity, project=wandb_project, name=wandb_run_name, config=config
    )


# training loop
t0 = time.time()


local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
epoch = 0
for epoch in range(max_epochs):
    train_ds.shuffle(epoch)
    train_batch_iter = iter(DataLoader(train_ds, batch_size=batch_size))  # type: ignore
    # get the first batch
    X, Y = next(train_batch_iter)

    for _ in tqdm(range(num_steps)):
        # determine and set the learning rate for this iteration
        lr = (
            get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr)
            if decay_lr
            else learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses = estimate_loss(
                model=model,
                eval_iters=eval_iters,
                batch_size=batch_size,
                split_to_ds={"train": train_ds, "val": validation_ds},
            )
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if wandb_log:
                try:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "tokens": iter_num * tokens_per_iter,
                            "loss/train": losses["train"],
                            "loss/val": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        },
                        step=iter_num,
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                    model_export(model, os.path.join(out_dir, "model.bin"), version=0)
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(min(gradient_accumulation_steps, num_steps - iter_num)):
            logits = model(X, Y)
            loss = model.last_loss / gradient_accumulation_steps
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = next(train_batch_iter)
            # backward pass, with gradient scaling if training in fp16
            loss.backward()
        # clip the gradient
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # type: ignore
        # step the optimizer and scaler if training in fp16
        optimizer.step()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1
