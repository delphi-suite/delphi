import logging
from collections.abc import Callable

from datasets import Dataset

from .config import TrainingConfig
from .iteration_params import IterationParams
from .run_context import RunContext
from .utils import (
    CheckpointData,
    ModelTrainingState,
    estimate_loss,
    save_checkpoint_if_needed,
)
from .wandb_utils import log_to_wandb


def should_save_checkpoint(config: TrainingConfig, mts: ModelTrainingState):
    return mts.iter_num % config.eval_interval == 0


def log_and_save_checkpoint(
    config: TrainingConfig,
    mts: ModelTrainingState,
    iteration_params: IterationParams,
    train_ds: Dataset,
    validation_ds: Dataset,
    run_context: RunContext,
):
    """
    Save a checkpoint of the current model + training state, evaluate, and optionally upload to huggingface and log to wandb (if configured)
    """
    model = mts.model
    if config.debug_config.no_eval:
        logging.debug("no_eval=True, skipping evaluation and using dummy losses")
        losses = {"train": 42.0, "val": 43.0}
    else:
        losses = estimate_loss(
            model=model,
            eval_iters=iteration_params.eval_iters,
            batch_size=config.batch_size,
            split_to_ds={"train": train_ds, "val": validation_ds},
            device=run_context.device,
            epoch=mts.epoch,
            feature_names={
                "train": config.data_config.train_feature,
                "val": (
                    config.data_config.validation_feature
                    or config.data_config.train_feature
                ),
            },
        )
    new_best_val_loss = False
    if losses["val"] < mts.best_val_loss:
        mts.best_val_loss = float(losses["val"])
        new_best_val_loss = True
    checkpoint_data = CheckpointData(
        tokens_per_iter=iteration_params.tokens_per_iter,
        losses=losses,
        new_best_val_loss=new_best_val_loss,
        config=config,
        model_training_state=mts,
        run_context=run_context,
    )
    logging.info(
        f"step {mts.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
    )
    save_checkpoint_if_needed(checkpoint_data)
    if config.wandb_config.log:
        log_to_wandb(checkpoint_data)
