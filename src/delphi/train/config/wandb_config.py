from dataclasses import dataclass
from datetime import datetime


@dataclass
class WandbConfig:
    log: bool = False
    project: str = "delphi"
    entity: str = "set_wandb.entity_to_your_wandb_username_to_make_wandb_logging_work"
    silence: bool = False
