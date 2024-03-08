import os
import time
from dataclasses import fields

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from delphi.train import wandb_utils
from delphi.train.gigaconfig import GigaConfig, debug_config
from delphi.train.iteration_params import set_iteration_params
from delphi.train.shuffle import shuffle_list
from delphi.train.train_step import train_step
from delphi.train.utils import (
    get_device,
    load_delphi_training_dataset,
    load_model_training_state,
    save_checkpoint_if_needed,
)


def run_training(config: GigaConfig):
    print("Starting training...")
    print()
    print("Config:")
    for field in fields(config):
        print(f"  {field.name}: {getattr(config, field.name)}")
    # system
    device = get_device(config.device)
    print("Device:", device)

    # load data
    print("Loading data...")
    train_ds = load_delphi_training_dataset("train", limit=config.train_sample_limit)
    validation_ds = load_delphi_training_dataset(
        "validation", limit=config.val_sample_limit
    )

    # derive iteration params (num_batches, num_steps, etc)
    iteration_params = set_iteration_params(config, train_ds, validation_ds)

    # setup
    print("Setting up...")
    os.makedirs(config.out_dir, exist_ok=True)
    torch.manual_seed(config.torch_seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # model init
    model_training_state = load_model_training_state(config, device)

    # setup eval callbacks
    eval_callbacks = [save_checkpoint_if_needed]
    if config.wandb_log:
        wandb_utils.init_wandb(config)
        eval_callbacks.append(wandb_utils.log_to_wandb)

    # training loop
    print("Starting training...")
    for epoch in range(config.max_epochs):
        sampler = shuffle_list(
            list(range(len(train_ds))), seed=config.batch_ordering_seed + epoch
        )
        train_batch_iter = iter(DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler))  # type: ignore
        for _ in tqdm(range(iteration_params.num_steps)):
            breaknow = train_step(
                model_training_state,
                train_ds,
                validation_ds,
                iteration_params,
                eval_callbacks,
                config,
                train_batch_iter,
                device,
            )
            if breaknow:
                break
