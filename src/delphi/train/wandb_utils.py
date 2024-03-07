from dataclasses import asdict

import torch
import wandb

from delphi.train.gigaconfig import GigaConfig
from delphi.train.utils import EvalData


def init_wandb(config: GigaConfig):
    wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config),
    )


def log_to_wandb(eval_data: EvalData):
    try:
        wandb.log(
            {
                "iter": eval_data.iter_num,
                "tokens": eval_data.iter_num * eval_data.tokens_per_iter,
                "loss/train": eval_data.losses["train"],
                "loss/val": eval_data.losses["val"],
                "lr": eval_data.lr,
                "mfu": eval_data.running_mfu * 100,  # convert to percentage
            },
            step=eval_data.iter_num,
        )
    except Exception as e:
        print(f"logging to wandb failed: {e}")
