import logging
import os
import time
from dataclasses import fields

import torch
from tqdm import tqdm
from transformers import __version__ as transformers_version

from delphi import __version__ as delphi_version
from delphi import constants

from .checkpoint_step import run_checkpoint, should_run_checkpoint
from .config import GigaConfig
from .iteration_params import set_iteration_params
from .run_context import RunContext
from .train_step import train_step
from .utils import (
    ModelTrainingState,
    get_device,
    get_indices_for_epoch,
    initialize_model_training_state,
    load_tokens_dataset_from_huggingface,
    set_lr,
)
from .wandb_utils import init_wandb


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
    train_ds = load_tokens_dataset_from_huggingface(
        dataset=config.data_config.train_dataset,
        split=config.data_config.train_split,
        tokens_feature=config.data_config.train_feature,
        limit=config.data_config.train_sample_limit,
    )
    validation_ds = load_tokens_dataset_from_huggingface(
        dataset=(
            config.data_config.validation_dataset or config.data_config.train_dataset
        ),
        split=config.data_config.validation_split,
        tokens_feature=(
            config.data_config.validation_feature or config.data_config.train_feature
        ),
        limit=config.data_config.validation_sample_limit,
    )

    # derive iteration params (num_batches, num_steps, etc)
    iteration_params = set_iteration_params(config, train_ds, validation_ds)

    # setup
    logging.info("Setting up...")
    os.makedirs(config.output_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # wandb setup
    if config.wandb_config.log:
        init_wandb(config=config)

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
        for step in tqdm(range(iteration_params.num_steps)):
            model_training_state.step = step
            if should_run_checkpoint(config, model_training_state):
                run_checkpoint(
                    config=config,
                    mts=model_training_state,
                    iteration_params=iteration_params,
                    train_ds=train_ds,
                    validation_ds=validation_ds,
                    run_context=run_context,
                )
            model_training_state.lr = set_lr(
                lr_decay_iters=iteration_params.lr_decay_iters,
                config=config,
                optimizer=model_training_state.optimizer,
                iter_num=model_training_state.iter_num,
            )
            train_step(
                model_training_state=model_training_state,
                train_ds=train_ds,
                config=config,
                device=run_context.device,
                indices=train_data_indices,
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
