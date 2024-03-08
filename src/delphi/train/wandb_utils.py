from dataclasses import asdict

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
    mts = eval_data.model_training_state
    try:
        wandb.log(
            {
                "iter": mts.iter_num,
                "tokens": mts.iter_num * eval_data.tokens_per_iter,
                "loss/train": eval_data.losses["train"],
                "loss/val": eval_data.losses["val"],
                "lr": mts.lr,
                "mfu": mts.running_mfu * 100,  # convert to percentage
            },
            step=mts.iter_num,
        )
    except Exception as e:
        print(f"logging to wandb failed: {e}")
