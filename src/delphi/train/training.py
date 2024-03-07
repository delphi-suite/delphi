import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from delphi.train import wandb_utils
from delphi.train.gigaconfig import jai_config as config
from delphi.train.train_step import train_step
from delphi.train.utils import (
    get_device,
    load_dataset,
    load_model_training_state,
    save_checkpoint_if_needed,
)

# system
device = get_device()


# load data

train_ds = load_dataset("train", config.max_seq_len, device, limit=256)
validation_ds = load_dataset("validation", config.max_seq_len, device)

# fixing some hyperparams to sensible defaults

num_batches = len(train_ds) // config.batch_size
print(f"num_batches: {num_batches}")
num_steps = num_batches // config.gradient_accumulation_steps
config.eval_iters = min(12, len(validation_ds) // config.batch_size)

lr_decay_iters = (
    config.max_epochs * num_batches
)  # should be ~= max_iters per Chinchilla

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

# model init
model_training_state = load_model_training_state(config, device)
iter_num = model_training_state.iter_num
best_val_loss = model_training_state.best_val_loss
model = model_training_state.model
optimizer = model_training_state.optimizer
model_args = model_training_state.model_args


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
