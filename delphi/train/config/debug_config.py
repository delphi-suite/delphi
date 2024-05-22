from dataclasses import dataclass, field

from beartype import beartype


@beartype
@dataclass(frozen=True)
class DebugConfig:
    no_training: bool = field(
        default=False, metadata={"help": "skip all actual training, do everything else"}
    )
    no_eval: bool = field(default=False, metadata={"help": "skip actual evaluation"})
