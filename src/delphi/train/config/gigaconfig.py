import os
from dataclasses import dataclass, field
from datetime import datetime

import platformdirs
from beartype import beartype

from .huggingface_config import HuggingfaceConfig
from .models import ModelConfig
from .optimizer_config import OptimizerConfig
from .wandb_config import WandbConfig


@beartype
@dataclass(frozen=True)
class GigaConfig:
    model_config: ModelConfig
    # meta
    run_name: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir: str = field(
        default=os.path.join(platformdirs.user_data_dir(appname="delphi"), run_name),
        metadata={
            "help": "output directory; also directory to resume from if init_from=='resume'"
        },
    )
    huggingface: HuggingfaceConfig = field(default_factory=HuggingfaceConfig)

    # device
    device: str = field(
        default="auto", metadata={"help": "device to use (cuda, mps, cpu)"}
    )

    # I/O
    eval_interval: int = field(default=2000, metadata={"help": "eval every N iters"})
    log_interval: int = field(default=1, metadata={"help": "log every N iters"})
    eval_iters: int = field(default=100, metadata={"help": "use N iters for each eval"})
    always_save_checkpoint: bool = field(
        default=False,  # if True, always save a checkpoint after each eval
        metadata={"help": "if True, always save a checkpoint after each eval"},
    )
    init_from: str = field(
        default="scratch",
        metadata={
            "help": "'scratch' for a new model, 'resume' to resume from output_dir"
        },
    )
    # wandb logging
    wandb_config: WandbConfig = field(default_factory=WandbConfig)
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
    # debugging
    train_sample_limit: int = field(
        default=-1,
        metadata={
            "help": "for debugging: limit size of the training set.# -1 implies no limit"
        },
    )
    val_sample_limit: int = field(
        default=-1,
        metadata={
            "help": "for debugging: limit size of the validation set. -1 implies no limit"
        },
    )
