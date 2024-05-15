import logging
import os
import time
from dataclasses import fields

import torch
from tqdm import tqdm

from .checkpoint_step import log_and_save_checkpoint, should_save_checkpoint
from .config import TrainingConfig
from .run_context import RunContext
from .train_step import train_step
from .utils import (
    ModelTrainingState,
    get_indices_for_epoch,
    initialize_model_training_state,
    set_lr,
    setup_determinism,
)
from .wandb_utils import init_wandb


def setup_training(config: TrainingConfig):
    logging.info("Setting up training...")
    os.makedirs(config.output_dir, exist_ok=True)

    # torch misc - TODO: check if this is actually needed
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # determinism
    setup_determinism(config.torch_seed)

    # wandb setup
    if config.wandb:
        init_wandb(config=config)


def check_set_env_cublas_workspace_config():
    expected_val = ":4096:8"
    actual_val = os.getenv("CUBLAS_WORKSPACE_CONFIG")
    if actual_val is None:
        # https://docs.nvidia.com/cuda/archive/12.4.0/cublas/index.html#results-reproducibility
        logging.info(
            f"Environment variable CUBLAS_WORKSPACE_CONFIG not set. Setting to '{expected_val}' to ensure reproducibility."
        )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_val
    elif actual_val != expected_val:
        logging.warning(
            f"Environment variable CUBLAS_WORKSPACE_CONFIG is set to {actual_val}, which may affect reproducibility. "
            f"We recommend setting it to '{expected_val}' to ensure reproducibility."
        )


def run_training(config: TrainingConfig) -> tuple[ModelTrainingState, RunContext]:
    if torch.cuda.is_available():
        check_set_env_cublas_workspace_config()
    setup_training(config)
    logging.info("Starting training...")
    logging.info("Config:")
    for field in fields(config):
        logging.info(f"  {field.name}: {getattr(config, field.name)}")
    run_context = RunContext(config.device)
    logging.debug(f"Run context: {run_context.asdict()}")

    # load data
    logging.info("Loading data...")
    train_ds = config.dataset.load_train()
    validation_ds = config.dataset.load_validation()
    logging.info(f"Train dataset: {len(train_ds)} samples")
    logging.info(f"Validation dataset: {len(validation_ds)} samples")

    # derive iteration params
    steps_per_epoch = len(train_ds) // config.batch_size
    lr_decay_iters = (
        config.max_epochs * steps_per_epoch
    )  # should be ~=max_iters per Chinchilla

    # model init
    model_training_state = initialize_model_training_state(config, run_context.device)

    # training loop
    logging.info("Starting training...")
    for epoch in range(config.max_epochs):
        logging.info(f"Epoch: {epoch} / {config.max_epochs - 1}")
        train_data_indices = get_indices_for_epoch(
            dataset_size=len(train_ds),
            batch_size=config.batch_size,
            epoch=epoch,
            ordering_seed=config.batch_ordering_seed,
        )
        model_training_state.epoch = epoch
        for step in tqdm(range(steps_per_epoch)):
            model_training_state.step = step
            if should_save_checkpoint(config, model_training_state):
                log_and_save_checkpoint(
                    config=config,
                    mts=model_training_state,
                    train_ds=train_ds,
                    validation_ds=validation_ds,
                    run_context=run_context,
                )
            model_training_state.lr = set_lr(
                lr_decay_iters=lr_decay_iters,
                config=config,
                optimizer=model_training_state.optimizer,
                iter_num=model_training_state.iter_num,
            )
            train_step(
                model_training_state=model_training_state,
                train_ds=train_ds,
                config=config,
                device=run_context.device,
                ds_indices=train_data_indices,
            )
            t1 = time.time()
            dt = t1 - model_training_state.last_training_step_time
            model_training_state.last_training_step_time = t1
            if model_training_state.iter_num % config.log_interval == 0:
                logging.debug(
                    (
                        f"{model_training_state.iter_num} | loss {model_training_state.train_loss:.4f} | lr {model_training_state.lr:e} | "
                        f"{dt*1000:.2f}ms"
                    )
                )
            model_training_state.iter_num += 1
    return model_training_state, run_context
