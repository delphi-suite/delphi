from dataclasses import dataclass, field

from beartype import beartype

from delphi.train.architectures import ModelTypes
from delphi.train.config.llama2_config_data import Llama2ConfigData
from delphi.train.config.wandb_config import WandbConfig


@beartype
@dataclass
class GigaConfig:
    """This is a terrible hack to get usable config objects to pass around
    It's way too big and ties way too many things together. This should be broken
    into several smaller configs.
    """

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
    # model
    dim: int = 288
    max_seq_len: int = 512
    vocab_size: int = 4096
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6
    multiple_of: int = 32
    dropout: float = 0.0
    # llama2hf model
    llama2hf_config: Llama2ConfigData = field(default_factory=Llama2ConfigData)
    # adamw optimizer
    gradient_accumulation_steps: int = 4  # used to simulate larger batch sizes
    learning_rate: float = 5e-4  # max learning rate
    max_epochs: int = 10  # total number of training epochs
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 1000  # how many steps to warm up for
    min_lr: float = 0.0  # should be ~learning_rate/10 per Chinchill
    # reproducibility
    batch_ordering_seed = 1337
    torch_seed = 42
    # debugging
    train_sample_limit: int = -1  # -1 implies no limit
    val_sample_limit: int = -1
