import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import platformdirs
from beartype import beartype

from .data_config import DataConfig
from .debug_config import DebugConfig
from .huggingface_config import HuggingfaceConfig
from .optimizer_config import OptimizerConfig
from .wandb_config import WandbConfig


@beartype
@dataclass(frozen=True)
class GigaConfig:
    model_config: dict[str, Any] = field(
        metadata={
            "help": "dictionary specifying model_class in transformers and arguments of the corresponding model config"
        }
    )
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
        default_factory=lambda: [],
        metadata={"help": "manually list iterations to save checkpoints on"},
    )
    log_interval: int = field(default=1, metadata={"help": "log every N iters"})
    eval_iters: int = field(default=100, metadata={"help": "use N iters for each eval"})

    # initialization
    init_from: str = field(
        default="scratch",
        metadata={
            "help": "'scratch' for a new model, 'resume' to resume from output_dir"
        },
    )
    resume_from_path: str = field(
        default=".",
        metadata={
            "help": "path to a checkpoint to resume from (if init_from=='resume')"
        },
    )

    # data
    batch_size: int = field(
        default=64,
        metadata={
            "help": "if gradient_accumulation_steps > 1, this is the micro-batch size"
        },
    )

    # model config
    max_seq_len: int = field(default=512, metadata={"help": "max sequence length"})

    # training
    max_epochs: int = field(
        default=10, metadata={"help": "total number of training epochs"}
    )
    grad_clip: float = field(
        default=1.0,
        metadata={"help": "clip gradients at this value, or disable if == 0.0"},
    )
    # (adamw) optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # reproducibility
    batch_ordering_seed: int = field(
        default=1337,
        metadata={"help": "seed used for pseudorandomly sampling data during training"},
    )
    torch_seed: int = field(default=42, metadata={"help": "seed used for torch"})

    # data
    data_config: DataConfig = field(
        default_factory=DataConfig,
        metadata={"help": "specify training and validation data"},
    )

    # third party
    wandb_config: WandbConfig = field(default_factory=WandbConfig)
    huggingface: HuggingfaceConfig = field(default_factory=HuggingfaceConfig)

    # debug
    debug_config: DebugConfig = field(default_factory=DebugConfig)
