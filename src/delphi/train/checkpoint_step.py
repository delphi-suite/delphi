import logging
import os
from collections.abc import Callable

from datasets import Dataset

from .config import GigaConfig
from .iteration_params import IterationParams
from .run_context import RunContext
from .utils import CheckpointData, ModelTrainingState, estimate_loss, save_results
from .wandb_utils import log_to_wandb


def should_run_checkpoint(config: GigaConfig, mts: ModelTrainingState):
    return mts.iter_num % config.checkpoint_interval == 0 and mts.iter_num > 0


def run_checkpoint(
    config: GigaConfig,
    mts: ModelTrainingState,
    iteration_params: IterationParams,
    train_ds: Dataset,
    validation_ds: Dataset,
    run_context: RunContext,
):
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
        )
    if losses["val"] < mts.best_val_loss:
        mts.best_val_loss = float(losses["val"])
    checkpoint_data = CheckpointData(
        tokens_per_iter=iteration_params.tokens_per_iter,
        losses=losses,
        config=config,
        model_training_state=mts,
        run_context=run_context,
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
        log_to_wandb(checkpoint_data)
