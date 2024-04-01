from dataclasses import dataclass


@dataclass
class AdamConfig:
    # adamw optimizer
    learning_rate: float = 5e-4  # max learning rate
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 1000  # how many steps to warm up for
    min_lr: float = 0.0  # should be ~learning_rate/10 per Chinchill
