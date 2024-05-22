import logging
import os
from dataclasses import asdict

import wandb

from .config import TrainingConfig
from .utils import ModelTrainingState


def silence_wandb():
    logging.info("silencing wandb output")
    os.environ["WANDB_SILENT"] = "true"


def init_wandb(config: TrainingConfig):
    # if log level < debug, silence wandb
    assert config.wandb is not None
    if logging.getLogger().level > logging.INFO or config.wandb.silence:
        silence_wandb()
    wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        name=config.run_name,
        config=asdict(config),
    )


def log_to_wandb(mts: ModelTrainingState, losses: dict[str, float], tokens_so_far: int):
    try:
        wandb.log(
            {
                "epoch": mts.epoch,
                "epoch_iter": mts.step,
                "global_iter": mts.iter_num,
                "tokens": tokens_so_far,
                "loss/train": losses["train"],
                "loss/val": losses["val"],
                "lr": mts.lr,
            },
            step=mts.iter_num,
        )
    except Exception as e:
        logging.error(f"logging to wandb failed: {e}")
