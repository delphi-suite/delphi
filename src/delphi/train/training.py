import os
from dataclasses import fields
from typing import cast

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataloader import _BaseDataLoaderIter
from tqdm import tqdm

from delphi.train import wandb_utils
from delphi.train.config.gigaconfig import GigaConfig
from delphi.train.iteration_params import set_iteration_params
from delphi.train.shuffle import shuffle_list
from delphi.train.train_step import train_step
from delphi.train.utils import (
    ModelTrainingState,
    get_device,
    load_delphi_training_dataset,
    load_model_training_state,
    save_checkpoint_if_needed,
)


def run_training(config: GigaConfig) -> ModelTrainingState:
    print("Starting training...")
    print("Setting torch.use_deterministic_algorithms(True)")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.torch_seed)
    print()
    print("Config:")
    for field in fields(config):
        print(f"  {field.name}: {getattr(config, field.name)}")
    # system
    device = get_device(config.device)
    print("Device:", device)

    # load data
    print("Loading data...")
    train_ds = cast(
        Dataset, load_delphi_training_dataset("train", limit=config.train_sample_limit)
    )
    validation_ds = cast(
        Dataset,
        load_delphi_training_dataset("validation", limit=config.val_sample_limit),
    )

    # derive iteration params (num_batches, num_steps, etc)
    iteration_params = set_iteration_params(config, train_ds, validation_ds)

    # setup
    print("Setting up...")
    os.makedirs(config.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # model init
    model_training_state = load_model_training_state(config, device)
    print(model_training_state.model.config.to_json_string())

    # setup eval callbacks
    eval_callbacks = [save_checkpoint_if_needed]
    if config.wandb_config.log:
        wandb_utils.init_wandb(config)
        eval_callbacks.append(wandb_utils.log_to_wandb)

    # training loop
    print("Starting training...")
    for epoch in range(config.max_epochs):
        sampler = list(range(len(train_ds)))  # type: ignore
        shuffle_list(sampler, seed=config.batch_ordering_seed + epoch)
        train_batch_iter = cast(
            _BaseDataLoaderIter,
            iter(
                DataLoader(
                    cast(TorchDataset, train_ds),
                    batch_size=config.batch_size,
                    sampler=sampler,
                    pin_memory=True,
                    drop_last=True,
                )
            ),
        )
        for i, _ in enumerate(tqdm(range(iteration_params.num_steps))):
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
    return model_training_state
