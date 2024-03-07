import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from delphi.train import wandb_utils
from delphi.train.gigaconfig import assert_config_sanity
from delphi.train.gigaconfig import jai_config as config
from delphi.train.iteration_params import set_iteration_params
from delphi.train.train_step import train_step
from delphi.train.utils import (
    get_device,
    load_delphi_training_dataset,
    load_model_training_state,
    save_checkpoint_if_needed,
)

# validating checks
assert_config_sanity(config)

# system
device = get_device()

# load data
train_ds = load_delphi_training_dataset("train", config.max_seq_len, device, limit=256)
validation_ds = load_delphi_training_dataset("validation", config.max_seq_len, device)

# derive iteration params (num_batches, num_steps, etc)
iteration_params = set_iteration_params(config, train_ds, validation_ds)


os.makedirs(config.out_dir, exist_ok=True)
torch.manual_seed(config.seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


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
    for _ in tqdm(range(iteration_params.num_steps)):
        breaknow, t0, iter_num, local_iter_num, best_val_loss = train_step(
            train_ds,
            validation_ds,
            iteration_params.lr_decay_iters,
            iteration_params.tokens_per_iter,
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
            iteration_params.num_steps,
            iteration_params.eval_iters,
        )
        if breaknow:
            break
