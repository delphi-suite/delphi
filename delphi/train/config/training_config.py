import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import platformdirs
from beartype import beartype

from .adam_config import AdamConfig
from .dataset_config import DatasetConfig
from .debug_config import DebugConfig


@beartype
@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    # model config; class_name=name of model class in transformers, everything else is kwargs for the corresponding model config
    model_config: dict[str, Any]

    max_seq_len: int
    run_name: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_dir: str = os.path.join(platformdirs.user_data_dir(appname="delphi"), run_name)

    # device to use (cuda, mps, cpu)
    device: str = "auto"

    # checkpoint every N iters
    checkpoint_interval: int = 2000

    # manually list iterations to save checkpoints on
    extra_checkpoint_iters: list[int] = field(default_factory=list)

    # log every N iters
    log_interval: int = 1

    # use N iters for each eval
    eval_iters: int = 100

    # path to a checkpoint to resume from (if init_from=='resume')
    resume_from_path: Optional[str] = None

    # number of samples used to compute the gradient for a single optimizer step
    batch_size: int = 64

    # total number of training epochs
    max_epochs: int = 10

    # clip gradients at this value, or disable if == 0.0
    grad_clip: float = 1.0

    # if > 1 reduces memory usage by computing gradient in microbatches
    gradient_accumulation_steps: int = 1

    # (adamw) optimizer
    adam: AdamConfig = field(default_factory=AdamConfig)

    # seed used for pseudorandomly sampling data during training
    batch_ordering_seed: int

    # seed used for torch
    torch_seed: int

    # whether to save the optimizer state with each checkpoint
    # this is twice as large as the model, but allows to resume training in a reproducible way
    save_optimizer: bool = True

    # specify training and validation data
    dataset: DatasetConfig

    # HF repo id or local directory containing the tokenizer. Used only to upload it to HF with the model, not for training
    tokenizer: str = ""

    # wandb config in 'entity/project' form. Set to empty string to not use wandb.
    wandb: str

    # HF repo id. Set to empty string to not push to repo.
    out_repo: str

    # debug config
    debug_config: DebugConfig = field(default_factory=DebugConfig)
