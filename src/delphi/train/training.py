import logging
import os
from dataclasses import fields
from typing import cast

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import __version__ as transformers_version

from delphi import __version__ as delphi_version

from . import wandb_utils
from .config import GigaConfig
from .iteration_params import set_iteration_params
from .run_context import RunContext
from .iteration_step import iteration_step
from .utils import (
    ModelTrainingState,
    batch_generator,
    get_device,
    initialize_model_training_state,
    load_delphi_training_dataset,
    save_checkpoint_if_needed,
)


def run_training(config: GigaConfig) -> tuple[ModelTrainingState, RunContext]:
    logging.info("Starting training...")
    logging.debug("Setting torch.use_deterministic_algorithms(True)")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(config.torch_seed)
    logging.info("Config:")
    for field in fields(config):
        logging.info(f"  {field.name}: {getattr(config, field.name)}")
    # system
    run_context = RunContext(
        device=get_device(config.device),
        torch_version=torch.__version__,
        delphi_version=delphi_version,
        transformers_version=transformers_version,
        os=os.uname().version,
    )
    logging.debug(f"Run context: {run_context}")

    # load data
    logging.info("Loading data...")
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
    logging.info("Setting up...")
    os.makedirs(config.output_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # model init
    model_training_state = initialize_model_training_state(config, run_context.device)

    # setup eval callbacks
    logging.info("Setting eval step callbacks...")
    eval_callbacks = [save_checkpoint_if_needed]
    logging.info(f"  added save_checkpoint_if_needed eval callback")
    if config.wandb_config.log:
        if config.wandb_config.silence:
            wandb_utils.silence_wandb()
        wandb_utils.init_wandb(config)
        eval_callbacks.append(wandb_utils.log_to_wandb)
        logging.info(f"  added log_to_wandb callback")

    # training loop
    logging.info("Starting training...")
    for epoch in range(config.max_epochs):
        logging.info(f"Epoch: {epoch} / {config.max_epochs - 1}")
        train_batch_iter = iter(
            batch_generator(
                train_ds, config.batch_size, epoch, config.batch_ordering_seed
            )
        )
        model_training_state.epoch = epoch
        for step in tqdm(range(iteration_params.num_steps)):
            model_training_state.step = step
            iteration_step(
                model_training_state,
                train_ds,
                validation_ds,
                iteration_params,
                eval_callbacks,
                config,
                train_batch_iter,
                run_context,
            )
    return model_training_state, run_context
