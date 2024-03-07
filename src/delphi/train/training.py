"""

"""

import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from delphi import constants
from delphi.eval.utils import load_delphi_dataset
from delphi.train import wandb_utils
from delphi.train.gigaconfig import jai_config as config
from delphi.train.tokenized_chunks_dataset import TokenizedChunksDataset
from delphi.train.train_step import train_step
from delphi.train.utils import (
    get_device,
    get_optimizer,
    initialize_model,
    resume_model,
    save_checkpoint_if_needed,
)

# system
device = get_device()


def load_dataset(split: str, device, limit: int = -1):
    if limit == -1:
        ds = load_delphi_dataset(constants.TOKENIZED_CORPUS_DATASET, split)
    else:
        ds = load_delphi_dataset(constants.TOKENIZED_CORPUS_DATASET, split).select(
            range(limit)
        )
    token_ds = TokenizedChunksDataset(ds, config.max_seq_len, device)
    token_ds.initialize_samples()
    return token_ds


# load data
train_ds = load_dataset("train", device, limit=256)
validation_ds = load_dataset("validation", device)

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


local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0

# training loop
t0 = time.time()

for epoch in range(config.max_epochs):
    train_ds.shuffle(epoch)
    train_batch_iter = iter(DataLoader(train_ds, batch_size=config.batch_size))  # type: ignore
    for _ in tqdm(range(num_steps)):
        breaknow, t0, iter_num, local_iter_num, best_val_loss = train_step(
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
        )
        if breaknow:
            break
