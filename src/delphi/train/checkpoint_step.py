import logging
import os

from datasets import Dataset

from .config import TrainingConfig
from .run_context import RunContext
from .utils import ModelTrainingState, count_tokens_so_far, estimate_loss, save_results
from .wandb_utils import log_to_wandb


def should_save_checkpoint(config: TrainingConfig, mts: ModelTrainingState):
    return (
        mts.iter_num % config.checkpoint_interval == 0
        and mts.iter_num > 0
        or mts.iter_num in config.extra_checkpoint_iters
    )


def log_and_save_checkpoint(
    config: TrainingConfig,
    mts: ModelTrainingState,
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
            eval_iters=config.eval_iters,
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
    logging.info(
        f"step {mts.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
    )
    results_path = os.path.join(config.output_dir, f"iter_{mts.iter_num:06d}")
    logging.info(f"saving checkpoint to {results_path}")
    save_results(
        config=config,
        train_results=mts,
        run_context=run_context,
        results_path=results_path,
    )
    if config.wandb_config.log:
        log_to_wandb(
            mts=mts,
            losses=losses,
            tokens_so_far=count_tokens_so_far(config, mts),
        )
