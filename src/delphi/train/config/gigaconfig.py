from dataclasses import dataclass, field

from beartype import beartype

from delphi.constants import ModelTypes
from delphi.train.config.llama2_config_data import Llama2ConfigData
from delphi.train.config.mamba_config_data import MambaConfigData
from delphi.train.config.optimizer_config import OptimizerConfig
from delphi.train.config.wandb_config import WandbConfig


@beartype
@dataclass
class GigaConfig:
    # device
    device: str = "auto"

    # model architecture
    architecture: str = ModelTypes.LLAMA2HF

    # I/O
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 100
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = (
        False  # if True, always save a checkpoint after each eval
    )
    init_from: str = "scratch"  # 'scratch' or 'resume'
    # wandb logging
    wandb_config: WandbConfig = field(default_factory=WandbConfig)
    # data
    batch_size: int = (
        64  # if gradient_accumulation_steps > 1, this is the micro-batch size
    )
    # model config
    max_seq_len: int = 512
    llama2hf_config: Llama2ConfigData = field(default_factory=Llama2ConfigData)
    mamba_config: MambaConfigData = field(default_factory=MambaConfigData)
    # training
    max_epochs: int = 10  # total number of training epochs
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0
    # (adamw) optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    # reproducibility
    batch_ordering_seed = 1337
    torch_seed = 42
    # debugging
    train_sample_limit: int = -1  # -1 implies no limit
    val_sample_limit: int = -1
