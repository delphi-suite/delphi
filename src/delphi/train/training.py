import os
from dataclasses import fields
from typing import cast

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import __version__ as transformers_version

from delphi import __version__ as delphi_version
from delphi.train import wandb_utils
from delphi.train.config.gigaconfig import GigaConfig
from delphi.train.iteration_params import set_iteration_params
from delphi.train.run_context import RunContext
from delphi.train.train_step import train_step
from delphi.train.utils import (
    ModelTrainingState,
    batch_generator,
    get_device,
    get_run_output_dir,
    initialize_model_training_state,
    load_delphi_training_dataset,
    save_checkpoint_if_needed,
)


def run_training(config: GigaConfig) -> tuple[ModelTrainingState, RunContext]:
    print("Starting training...")
    print("Setting torch.use_deterministic_algorithms(True)")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(config.torch_seed)
    print()
    print("Config:")
    for field in fields(config):
        print(f"  {field.name}: {getattr(config, field.name)}")
    # system
    run_context = RunContext(
        device=get_device(config.device),
        torch_version=torch.__version__,
        delphi_version=delphi_version,
        transformers_version=transformers_version,
        os=os.uname().version,
    )
    print(f"Run context: {run_context}")

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
    run_dir = get_run_output_dir(config)
    os.makedirs(run_dir, exist_ok=True)
    print("  Run dir:", run_dir)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # model init
    model_training_state = initialize_model_training_state(config, run_context.device)

    # setup eval callbacks
    eval_callbacks = [save_checkpoint_if_needed]
    if config.wandb_config.log:
        wandb_utils.init_wandb(config)
        eval_callbacks.append(wandb_utils.log_to_wandb)

    # training loop
    print("Starting training...")
    for epoch in range(config.max_epochs):
        train_batch_iter = iter(
            batch_generator(
                train_ds, config.batch_size, epoch, config.batch_ordering_seed
            )
        )
        model_training_state.epoch = epoch
        for step in tqdm(range(iteration_params.num_steps)):
            model_training_state.step = step
            breaknow = train_step(
                model_training_state,
                train_ds,
                validation_ds,
                iteration_params,
                eval_callbacks,
                config,
                train_batch_iter,
                run_context.device,
            )
            if breaknow:
                break
    return model_training_state, run_context
