from dataclasses import dataclass
from datetime import datetime


@dataclass
class GigaConfig:
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
    wandb_log: bool = True  # disabled by default
    wandb_entity: str = "jannik-brinkmann"
    wandb_project: str = "delphi"
    wandb_run_name: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # data
    batch_size: int = (
        64  # if gradient_accumulation_steps > 1, this is the micro-batch size
    )
    max_seq_len: int = 256
    vocab_source: str = (
        "llama2"  # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
    )
    vocab_size: int = 32000  # the Llama 2 tokenizer has 32K tokens
    # model
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6
    multiple_of: int = 32
    dropout: float = 0.0
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


# Jai Overrides TODO: remove these
jai_config = GigaConfig(
    wandb_entity="jaiwithani",
    vocab_source="custom",
    vocab_size=4096,
    max_seq_len=512,
    dim=48,
    n_layers=2,
    n_heads=2,
    n_kv_heads=2,
    max_epochs=2,
    eval_interval=500,
    eval_iters=10,
)
