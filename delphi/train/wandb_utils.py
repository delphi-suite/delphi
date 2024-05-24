import logging
from dataclasses import asdict

import wandb

from .config import TrainingConfig
from .utils import ModelTrainingState


def init_wandb(config: TrainingConfig):
    assert "/" in config.wandb, "wandb should be in the 'entity/project' form"
    wandb_entity, wandb_project = config.wandb.split("/")
    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
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
