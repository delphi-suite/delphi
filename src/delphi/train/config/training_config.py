import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import platformdirs
from beartype import beartype

from .adam_config import AdamConfig
from .dataset_config import DatasetConfig
from .debug_config import DebugConfig
from .huggingface_config import HuggingfaceConfig
from .wandb_config import WandbConfig


@beartype
@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    model_config: dict[str, Any] = field(
        metadata={
            "help": "model config; class_name=name of model class in transformers, everything else is kwargs for the corresponding model config"
        },
    )
    max_seq_len: int = field(metadata={"help": "max sequence length"})
    # meta
    run_name: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir: str = field(
        default=os.path.join(platformdirs.user_data_dir(appname="delphi"), run_name),
        metadata={"help": "output directory"},
    )

    # device
    device: str = field(
        default="auto", metadata={"help": "device to use (cuda, mps, cpu)"}
    )

    # checkpoints, logging, eval
    checkpoint_interval: int = field(
        default=2000, metadata={"help": "checkpoint every N iters"}
    )
    extra_checkpoint_iters: list[int] = field(
        default_factory=list,
        metadata={"help": "manually list iterations to save checkpoints on"},
    )
    log_interval: int = field(default=1, metadata={"help": "log every N iters"})
    eval_iters: int = field(default=100, metadata={"help": "use N iters for each eval"})

    # resume from checkpoint
    resume_from_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to a checkpoint to resume from (if init_from=='resume')"
        },
    )

    # data
    batch_size: int = field(
        default=64,
        metadata={
            "help": "number of samples used to compute the gradient for a single optimizer step"
        },
    )

    # training
    max_epochs: int = field(
        default=10, metadata={"help": "total number of training epochs"}
    )
    grad_clip: float = field(
        default=1.0,
        metadata={"help": "clip gradients at this value, or disable if == 0.0"},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "if > 1 reduces memory usage by computing gradient in microbatches"
        },
    )
    # (adamw) optimizer
    adam: AdamConfig = field(default_factory=AdamConfig)

    # reproducibility
    batch_ordering_seed: int = field(
        metadata={"help": "seed used for pseudorandomly sampling data during training"},
    )
    torch_seed: int = field(metadata={"help": "seed used for torch"})

    # data
    dataset: DatasetConfig = field(
        metadata={"help": "specify training and validation data"},
    )

    # third party
    wandb: Optional[WandbConfig] = None
    hf: Optional[HuggingfaceConfig] = None

    # debug
    debug_config: DebugConfig = field(default_factory=DebugConfig)
