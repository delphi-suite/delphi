import logging
import os
from dataclasses import asdict

import wandb

from .config import GigaConfig
from .utils import CheckpointData


def silence_wandb():
    logging.info("silencing wandb output")
    os.environ["WANDB_SILENT"] = "true"


def init_wandb(config: GigaConfig):
    # if log level < debug, silence wandb
    if logging.getLogger().level > logging.INFO or config.wandb_config.silence:
        silence_wandb()
    wandb.init(
        entity=config.wandb_config.entity,
        project=config.wandb_config.project,
        name=config.run_name,
        config=asdict(config),
    )


def log_to_wandb(eval_data: CheckpointData):
    mts = eval_data.model_training_state
    try:
        wandb.log(
            {
                "iter": mts.iter_num,
                "tokens": mts.iter_num * eval_data.tokens_per_iter,
                "loss/train": eval_data.losses["train"],
                "loss/val": eval_data.losses["val"],
                "lr": mts.lr,
            },
            step=mts.iter_num,
        )
    except Exception as e:
        logging.error(f"logging to wandb failed: {e}")
